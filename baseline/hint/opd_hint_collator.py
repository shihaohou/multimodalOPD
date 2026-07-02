"""Data collation for Grounding-Hint Distillation (GHD).

GHD = vanilla OPD with a *spatially privileged* teacher. The student is always
scored on the plain ``(image, question)`` prompt (exactly :class:`OPDDataCollator`);
the frozen teacher is additionally handed the GT evidence bounding box through one
of two privilege channels (``teacher_privilege_mode``):

* ``hint`` (default) — **direction.** Full image + the box as a *text coordinate
  hint* appended to the question ("pay attention to region ``[x1,y1,x2,y2]``"). The
  image is not cropped/upsampled, so the KL gap is attributable to *where to look*,
  not extra pixels.
* ``crop`` — **zoom.** The image is cropped to the box (no text hint), so the
  teacher sees the evidence region at higher effective resolution — genuinely more
  *information* on high-res inputs (the V*Bench regime). Uses the GT box, so the
  crop is fixed and the student forward backprops normally (no RL needed).

This collator builds **two** prompts per sample:

* ``student_prompt_*`` — system + user(full image + question). What the policy
  rolls out from and what the student forward scores. Identical in both modes.
* ``teacher_prompt_*`` — system + user(privileged image + question [+ hint]). The
  frozen teacher scores the student's completion under this privileged prefix.

Samples whose ``bbox`` field is missing / unparseable degrade to the plain student
prompt for that row (vanilla OPD) instead of crashing — the ``has_hint`` flag
records which rows were actually privileged (``hint_coverage`` curve).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from baseline.opd_data_collator import (
    OPD_SYSTEM_PROMPT,
    OPDDataCollator,
    _safe_rgb_image,
    format_opd_student_prompt,
)
from baseline.probe.saliency_data import BoxNorm, parse_bbox_norm

# The exact privileged hint appended to the teacher's question. ``{bbox}`` is
# filled with the per-sample normalized box, e.g. ``[0.12, 0.34, 0.55, 0.78]``.
# Spells out the coordinate convention (normalized, top-left origin, x1y1x2y2) so a
# format-following VLM teacher can localize, states *why* the region matters (the
# evidence is there), and — critically — forbids the teacher from VERBALIZING the
# box/coordinates/hint in its output.
#
# Why the no-verbalize clause: without it, a region-aware teacher (e.g.
# CapCurriculum) writes its CoT as "the hint directs me to bbox [X]… in the crop…".
# Reverse-KL then forces the hint-FREE student to reproduce that text on its own
# rollouts — it must emit coordinate tokens it cannot know → unmatchable per-token KL
# → on-policy mode collapse into token salad (observed: loss_opd crashes, clip_ratio
# spikes, agreement craters ~step 60-100). Keeping the box out of the teacher's
# *output* leaves only the benign "more grounded answer" signal we actually want.
# Override via ``--hint_template`` for ablations.
HINT_TEMPLATE = (
    "Hint: the evidence needed to answer the question is inside the bounding box "
    "{bbox} (normalized to [0,1], top-left origin, [x1, y1, x2, y2]). Use this only "
    "to decide where to look in the image, then answer the question directly. Do NOT "
    "mention the bounding box, the coordinates, this hint, or a crop in your "
    "reasoning or your answer."
)


def format_bbox_hint(
    bbox: BoxNorm,
    template: str = HINT_TEMPLATE,
    *,
    decimals: int = 2,
) -> str:
    """Render a normalized ``(x1,y1,x2,y2)`` box into the hint sentence."""
    coords = "[" + ", ".join(f"{v:.{decimals}f}" for v in bbox) + "]"
    return template.format(bbox=coords)


def crop_to_bbox(image: Any, bbox: BoxNorm, *, padding: float = 0.0) -> Any:
    """Crop a PIL image to a normalized ``(x1,y1,x2,y2)`` box (top-left origin).

    ``padding`` (fraction of the box's own width/height) expands the crop on every
    side for a little context — 0.0 is a tight crop. Returns the original image
    unchanged if the box collapses to <1px after rounding (degenerate → no zoom).
    The caller re-runs ``_safe_rgb_image`` on the result so a tiny crop is padded
    up to the processor's minimum side instead of crashing it.
    """
    width, height = image.size
    x1, y1, x2, y2 = bbox
    left, top, right, bottom = x1 * width, y1 * height, x2 * width, y2 * height
    if padding > 0:
        pad_w = (right - left) * padding
        pad_h = (bottom - top) * padding
        left, top, right, bottom = left - pad_w, top - pad_h, right + pad_w, bottom + pad_h
    left = max(0, int(round(left)))
    top = max(0, int(round(top)))
    right = min(width, int(round(right)))
    bottom = min(height, int(round(bottom)))
    if right <= left or bottom <= top:
        return image
    return image.crop((left, top, right, bottom))


def build_hint_teacher_messages(
    problem: Any,
    image: Any,
    hint: str,
    *,
    system_prompt: str = OPD_SYSTEM_PROMPT,
    suffix: str = "",
) -> list[dict[str, Any]]:
    """``[system, user(image + question + hint)]`` — the privileged teacher prompt.

    Mirrors :func:`baseline.opd_data_collator.build_opd_messages` exactly (same
    system prompt, same image-then-text content order) and appends ``hint`` to the
    end of the user text so the teacher reads question first, then where to look.
    An empty ``hint`` reproduces the student prompt verbatim.
    """
    text = format_opd_student_prompt(problem, suffix)
    if hint:
        text = f"{text}\n{hint}"
    content: list[dict[str, Any]] = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": text})
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append(
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
        )
    messages.append({"role": "user", "content": content})
    return messages


@dataclass
class OPDHintDataCollator(OPDDataCollator):
    """OPD student prompt + a privileged teacher prompt (text hint or image crop)."""

    # How the teacher is privileged with the box: "hint" = full image + text coords
    # (direction); "crop" = image cropped to the box, no text (zoom).
    teacher_privilege_mode: str = "hint"
    # System prompt for the privileged TEACHER turn. ``None`` (default) reuses the
    # student's ``system_prompt`` verbatim — the original GHD behaviour, so the spine
    # is byte-for-byte unchanged. Set it to decouple the two turns: the Locate-Once
    # fork gives the student a "find a <box>" prompt but keeps the teacher on the
    # plain think prompt (it must NOT be told to locate — it is silently handed the
    # box via the hint and forbidden from verbalizing it).
    teacher_system_prompt: str | None = None
    # Column on the dataset row holding the evidence box. saliency-r1-8k ships it
    # as a string ``"[x1, y1, x2, y2]"`` normalized to [0,1]; parse_bbox_norm also
    # accepts a list/tuple and order-normalizes / clamps / drops degenerate boxes.
    bbox_field: str = "bbox"
    hint_template: str = HINT_TEMPLATE
    hint_coord_decimals: int = 2
    # crop mode only: context padding around the box (fraction of box w/h per side).
    crop_padding: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.teacher_privilege_mode not in {"hint", "crop"}:
            raise ValueError(
                f"Unknown teacher_privilege_mode {self.teacher_privilege_mode!r}; "
                "use 'hint' (text coordinates) or 'crop' (cropped evidence image)."
            )

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        # Build everything the vanilla OPD collator does (student prompt + the
        # rollout/answer metadata) first, then add the privileged teacher prompt.
        result = super().__call__(features)
        images = result["student_images"]
        problems = result["vigos_problems"]

        teacher_messages: list[list[dict[str, Any]]] = []
        teacher_prompt_texts: list[str] = []
        hint_texts: list[str] = []
        has_hint: list[int] = []
        for feature, image, problem in zip(features, images, problems):
            bbox = parse_bbox_norm(feature.get(self.bbox_field))
            if self.teacher_privilege_mode == "crop":
                # ZOOM: teacher sees the box cropped from the full image, no hint
                # text. Re-run _safe_rgb_image so a tiny crop is padded, not crashed.
                teacher_image = (
                    _safe_rgb_image(crop_to_bbox(image, bbox, padding=self.crop_padding))
                    if bbox is not None
                    else image
                )
                hint = ""
                privileged = bbox is not None
            else:
                # DIRECTION: full image + the box as a text coordinate hint.
                teacher_image = image
                hint = (
                    format_bbox_hint(
                        bbox, self.hint_template, decimals=self.hint_coord_decimals
                    )
                    if bbox is not None
                    else ""
                )
                privileged = bool(hint)
            message = build_hint_teacher_messages(
                problem,
                teacher_image,
                hint,
                system_prompt=self.teacher_system_prompt or self.system_prompt,
                suffix=self.opd_prompt_suffix,
            )
            teacher_messages.append(message)
            teacher_prompt_texts.append(
                self._apply_chat_template(
                    message, tokenize=False, add_generation_prompt=True
                )
            )
            hint_texts.append(hint)
            has_hint.append(1 if hint else 0)

        # Encode the privileged teacher prompt under the "teacher" prefix; the
        # trainer pulls it back with _prompt_inputs(inputs, "teacher"). No assistant
        # prefill (matches the student encode) so the completion appends at the same
        # <|im_start|>assistant boundary for both forwards.
        result.update(self._encode("teacher", teacher_messages))
        result["teacher_prompt_texts"] = teacher_prompt_texts
        result["hint_texts"] = hint_texts
        result["has_hint"] = torch.tensor(has_hint, dtype=torch.long)
        return result
