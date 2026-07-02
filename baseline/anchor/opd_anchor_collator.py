"""Data collation for Evidence Anchor OPD.

Evidence Anchor OPD is the hidden-hint/GHD prompt pair plus an explicit prompt
anchor marker. The student sees ``image + question + anchor``; the teacher sees
``image + question + hidden bbox hint + anchor`` and scores the same rollout. The
anchor is never part of the completion target, but later completion tokens can
attend to it under the causal mask, so its hidden state is a small latent
bottleneck for the privileged visual cue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from baseline.hint.opd_hint_collator import (
    HINT_TEMPLATE,
    crop_to_bbox,
    format_bbox_hint,
)
from baseline.opd_data_collator import (
    OPD_SYSTEM_PROMPT,
    OPDDataCollator,
    _safe_rgb_image,
    format_opd_student_prompt,
)
from baseline.probe.saliency_data import parse_bbox_norm
from vigos.answer_utils import normalize_reference_answer
from vigos.data_collator import _format_reasoning_reference


DEFAULT_ANCHOR_TOKEN = "<EVID>"
DEFAULT_ANCHOR_ANSWER_CUE = "Now answer the question."


def build_anchor_text(
    *,
    anchor_token: str = DEFAULT_ANCHOR_TOKEN,
    num_anchor_tokens: int = 1,
    indexed_tokens: bool = True,
) -> str:
    """Render the visible prompt marker used as the latent anchor span."""
    n = max(1, int(num_anchor_tokens))
    token = str(anchor_token).strip() or DEFAULT_ANCHOR_TOKEN
    if n == 1:
        return token
    if indexed_tokens:
        if token.endswith(">"):
            stem = token[:-1]
            return " ".join(f"{stem}_{idx}>" for idx in range(1, n + 1))
        return " ".join(f"{token}_{idx}" for idx in range(1, n + 1))
    return " ".join(token for _ in range(n))


def format_anchor_prompt(
    problem: Any,
    *,
    suffix: str,
    anchor_text: str,
    answer_cue: str,
    hint: str = "",
    hint_after_anchor: bool = False,
) -> str:
    """Question text with optional hidden hint around the causal anchor."""
    text = format_opd_student_prompt(problem, suffix)
    if hint and not hint_after_anchor:
        text = f"{text}\n{hint}"
    text = f"{text}\n{anchor_text}"
    if hint and hint_after_anchor:
        text = f"{text}\n{hint}"
    if answer_cue:
        text = f"{text}\n{answer_cue}"
    return text


def build_anchor_messages(
    problem: Any,
    image: Any,
    *,
    system_prompt: str,
    suffix: str,
    anchor_text: str,
    answer_cue: str,
    hint: str = "",
    hint_after_anchor: bool = False,
) -> list[dict[str, Any]]:
    """``[system, user(image + question + anchor/hint + cue)]``."""
    content: list[dict[str, Any]] = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append(
        {
            "type": "text",
            "text": format_anchor_prompt(
                problem,
                suffix=suffix,
                anchor_text=anchor_text,
                answer_cue=answer_cue,
                hint=hint,
                hint_after_anchor=hint_after_anchor,
            ),
        }
    )
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append(
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
        )
    messages.append({"role": "user", "content": content})
    return messages


def _plain_token_ids(tokenizer: Any, text: str | None) -> list[int]:
    if not text:
        return []
    if hasattr(tokenizer, "encode"):
        try:
            return list(tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            return list(tokenizer.encode(text))
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def _find_subsequence(row: torch.Tensor, pattern: list[int]) -> tuple[int, int] | None:
    if not pattern:
        return None
    pat = torch.tensor(pattern, dtype=row.dtype, device=row.device)
    max_start = int(row.shape[0]) - int(pat.shape[0])
    for start in range(max_start + 1):
        end = start + int(pat.shape[0])
        if torch.equal(row[start:end], pat):
            return start, end
    return None


def _anchor_span_from_text(
    *,
    prompt_text: str,
    row: torch.Tensor,
    attention: torch.Tensor,
    tokenizer: Any,
    anchor_text: str,
) -> tuple[int, int] | None:
    """Map the anchor's character span in the rendered chat prompt to token ids."""
    char_start = prompt_text.find(anchor_text)
    if char_start < 0:
        return None
    char_end = char_start + len(anchor_text)
    start = len(_plain_token_ids(tokenizer, prompt_text[:char_start]))
    end = len(_plain_token_ids(tokenizer, prompt_text[:char_end]))
    if end <= start:
        return None

    nonpad = attention.nonzero(as_tuple=False).flatten()
    pad_offset = int(nonpad[0].item()) if int(nonpad.numel()) else 0
    start += pad_offset
    end += pad_offset
    if start < 0 or end > int(row.shape[0]):
        return None
    if not bool(attention[start:end].to(dtype=torch.bool).all()):
        return None
    return start, end


def _anchor_span_from_patterns(
    *,
    row: torch.Tensor,
    attention: torch.Tensor,
    tokenizer: Any,
    anchor_text: str,
) -> tuple[int, int] | None:
    """Fallback token-pattern search when rendered-text token counting diverges."""
    variants = [
        anchor_text,
        f"\n{anchor_text}",
        f"{anchor_text}\n",
        f"\n{anchor_text}\n",
        f" {anchor_text}",
        f" {anchor_text} ",
    ]
    for variant in variants:
        pattern = _plain_token_ids(tokenizer, variant)
        span = _find_subsequence(row, pattern)
        if span is None:
            continue
        start, end = span
        if bool(attention[start:end].to(dtype=torch.bool).all()):
            return start, end
    return None


def anchor_position_tensors(
    *,
    prompt_texts: list[str],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer: Any,
    anchor_text: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return padded absolute anchor-token positions and a validity mask.

    Positions index into the padded prompt tensor produced by ``_encode``. The
    anchor can tokenize to multiple ids; all ids in the rendered marker span are
    aligned by the trainer.
    """
    rows: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for text, row, attention in zip(
        prompt_texts, input_ids, attention_mask, strict=True
    ):
        span = _anchor_span_from_text(
            prompt_text=text,
            row=row,
            attention=attention,
            tokenizer=tokenizer,
            anchor_text=anchor_text,
        )
        if span is None:
            span = _anchor_span_from_patterns(
                row=row,
                attention=attention,
                tokenizer=tokenizer,
                anchor_text=anchor_text,
            )
        if span is None:
            rows.append(torch.zeros(1, dtype=torch.long, device=input_ids.device))
            masks.append(torch.zeros(1, dtype=torch.bool, device=input_ids.device))
            continue
        start, end = span
        positions = torch.arange(start, end, dtype=torch.long, device=input_ids.device)
        rows.append(positions)
        masks.append(torch.ones_like(positions, dtype=torch.bool))

    max_len = max(1, max(int(row.numel()) for row in rows))
    pos_rows = []
    mask_rows = []
    for pos, mask in zip(rows, masks, strict=True):
        pad = max_len - int(pos.numel())
        if pad > 0:
            pos = torch.cat(
                [pos, torch.zeros(pad, dtype=pos.dtype, device=pos.device)], dim=0
            )
            mask = torch.cat(
                [mask, torch.zeros(pad, dtype=mask.dtype, device=mask.device)], dim=0
            )
        pos_rows.append(pos[:max_len])
        mask_rows.append(mask[:max_len])
    return torch.stack(pos_rows, dim=0), torch.stack(mask_rows, dim=0)


@dataclass
class OPDAnchorDataCollator(OPDDataCollator):
    """Plain student prompt + hidden-hint teacher prompt + anchor positions."""

    teacher_privilege_mode: str = "hint"
    teacher_system_prompt: str | None = None
    bbox_field: str = "bbox"
    hint_template: str = HINT_TEMPLATE
    hint_coord_decimals: int = 2
    crop_padding: float = 0.0
    anchor_token: str = DEFAULT_ANCHOR_TOKEN
    num_anchor_tokens: int = 1
    anchor_indexed_tokens: bool = True
    anchor_answer_cue: str = DEFAULT_ANCHOR_ANSWER_CUE
    hint_after_anchor: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.teacher_privilege_mode not in {"hint", "crop"}:
            raise ValueError(
                f"Unknown teacher_privilege_mode {self.teacher_privilege_mode!r}; "
                "use 'hint' (text coordinates) or 'crop' (cropped evidence image)."
            )
        if int(self.num_anchor_tokens) <= 0:
            raise ValueError("num_anchor_tokens must be >= 1.")
        self.anchor_text = build_anchor_text(
            anchor_token=self.anchor_token,
            num_anchor_tokens=self.num_anchor_tokens,
            indexed_tokens=self.anchor_indexed_tokens,
        )

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        student_messages: list[list[dict[str, Any]]] = []
        teacher_messages: list[list[dict[str, Any]]] = []
        student_prompt_texts: list[str] = []
        teacher_prompt_texts: list[str] = []
        student_images: list[Any] = []
        problems: list[str] = []
        references: list[str] = []
        answers: list[str] = []
        sample_ids: list[int] = []
        hint_texts: list[str] = []
        has_hint: list[int] = []
        bbox_values: list[list[float]] = []
        bbox_masks: list[int] = []

        for local_idx, feature in enumerate(features):
            image = _safe_rgb_image(feature.get("images", feature.get("image")))
            problem = str(feature["problem"]).strip()
            reference = _format_reasoning_reference(feature, self.answer_field)
            answer = normalize_reference_answer(feature.get(self.answer_field))
            sample_id = int(feature.get("problem_id", local_idx))
            bbox = parse_bbox_norm(feature.get(self.bbox_field))
            if bbox is None:
                bbox_values.append([0.0, 0.0, 0.0, 0.0])
                bbox_masks.append(0)
            else:
                bbox_values.append([float(v) for v in bbox])
                bbox_masks.append(1)

            student_message = build_anchor_messages(
                problem,
                image,
                system_prompt=self.system_prompt,
                suffix=self.opd_prompt_suffix,
                anchor_text=self.anchor_text,
                answer_cue=self.anchor_answer_cue,
            )

            if self.teacher_privilege_mode == "crop":
                teacher_image = (
                    _safe_rgb_image(crop_to_bbox(image, bbox, padding=self.crop_padding))
                    if bbox is not None
                    else image
                )
                hint = ""
                privileged = bbox is not None
            else:
                teacher_image = image
                hint = (
                    format_bbox_hint(
                        bbox, self.hint_template, decimals=self.hint_coord_decimals
                    )
                    if bbox is not None
                    else ""
                )
                privileged = bool(hint)

            teacher_message = build_anchor_messages(
                problem,
                teacher_image,
                system_prompt=self.teacher_system_prompt or self.system_prompt,
                suffix=self.opd_prompt_suffix,
                anchor_text=self.anchor_text,
                answer_cue=self.anchor_answer_cue,
                hint=hint,
                hint_after_anchor=self.hint_after_anchor,
            )

            student_messages.append(student_message)
            teacher_messages.append(teacher_message)
            student_prompt_texts.append(
                self._apply_chat_template(
                    student_message, tokenize=False, add_generation_prompt=True
                )
            )
            teacher_prompt_texts.append(
                self._apply_chat_template(
                    teacher_message, tokenize=False, add_generation_prompt=True
                )
            )
            student_images.append(image)
            problems.append(problem)
            references.append(reference)
            answers.append(answer)
            sample_ids.append(sample_id)
            hint_texts.append(hint)
            has_hint.append(1 if privileged else 0)

        result: dict[str, Any] = {}
        result.update(self._encode("student", student_messages))
        result.update(self._encode("teacher", teacher_messages))

        s_pos, s_mask = anchor_position_tensors(
            prompt_texts=student_prompt_texts,
            input_ids=result["student_prompt_input_ids"],
            attention_mask=result["student_prompt_attention_mask"],
            tokenizer=tokenizer,
            anchor_text=self.anchor_text,
        )
        t_pos, t_mask = anchor_position_tensors(
            prompt_texts=teacher_prompt_texts,
            input_ids=result["teacher_prompt_input_ids"],
            attention_mask=result["teacher_prompt_attention_mask"],
            tokenizer=tokenizer,
            anchor_text=self.anchor_text,
        )
        result["student_anchor_positions"] = s_pos
        result["student_anchor_attention_mask"] = s_mask
        result["teacher_anchor_positions"] = t_pos
        result["teacher_anchor_attention_mask"] = t_mask

        result["student_prompt_texts"] = student_prompt_texts
        result["teacher_prompt_texts"] = teacher_prompt_texts
        result["student_images"] = student_images
        result["vigos_problems"] = problems
        result["vigos_references"] = references
        result["vigos_answers"] = answers
        result["sample_ids"] = torch.tensor(sample_ids, dtype=torch.long)
        result["hint_texts"] = hint_texts
        result["has_hint"] = torch.tensor(has_hint, dtype=torch.long)
        result["bbox_norm"] = torch.tensor(bbox_values, dtype=torch.float32)
        result["bbox_attention_mask"] = torch.tensor(bbox_masks, dtype=torch.bool)
        result["anchor_texts"] = [self.anchor_text for _ in features]
        return result
