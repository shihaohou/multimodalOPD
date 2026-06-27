"""Data collation for Grounding-Hint Distillation (GHD).

GHD = vanilla OPD with a *spatially privileged* teacher. The student is scored on
the plain ``(image, question)`` prompt (exactly :class:`OPDDataCollator`); the
frozen teacher is additionally handed the GT **evidence bounding box as a text
coordinate hint** appended to the question — "pay attention to region
``[x1,y1,x2,y2]``". The image itself is **not** cropped or upsampled: the teacher
gets *direction* (where to look), not *information* (a sharper view), so any KL
gap it opens up is attributable to grounding, not to extra pixels.

This collator therefore builds **two** prompts per sample:

* ``student_prompt_*`` — system + user(image + question). Identical to
  :class:`~baseline.opd_data_collator.OPDDataCollator`; this is what the policy
  rolls out from and what the student forward scores.
* ``teacher_prompt_*`` — system + user(image + question + **bbox hint**). The
  frozen teacher scores the student's completion under *this* privileged prefix.

Both encode the **same** ``_safe_rgb_image`` at the same resolution, so the only
difference fed to the two forwards is the extra hint text. Samples whose ``bbox``
field is missing / unparseable get an empty hint, so the teacher prompt degrades
to the student prompt for that row (vanilla OPD) instead of crashing — the
``has_hint`` flag records which rows were actually privileged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from baseline.opd_data_collator import (
    OPD_SYSTEM_PROMPT,
    OPDDataCollator,
    format_opd_student_prompt,
)
from baseline.probe.saliency_data import BoxNorm, parse_bbox_norm

# The exact privileged hint appended to the teacher's question. ``{bbox}`` is
# filled with the per-sample normalized box, e.g. ``[0.12, 0.34, 0.55, 0.78]``.
# Deliberately spells out the coordinate convention (normalized, top-left origin,
# x1y1x2y2) so a format-following VLM teacher can localize without guessing, and
# states *why* the region matters (the evidence is there) to bias its attention,
# not its answer. Override via ``--hint_template`` for ablations.
HINT_TEMPLATE = (
    "Hint: pay special attention to the region of the image inside the bounding "
    "box {bbox} (normalized to [0,1], top-left origin, [x1, y1, x2, y2]). "
    "The evidence needed to answer the question is located there."
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
    """OPD student prompt + a privileged teacher prompt carrying the bbox hint."""

    # Column on the dataset row holding the evidence box. saliency-r1-8k ships it
    # as a string ``"[x1, y1, x2, y2]"`` normalized to [0,1]; parse_bbox_norm also
    # accepts a list/tuple and order-normalizes / clamps / drops degenerate boxes.
    bbox_field: str = "bbox"
    hint_template: str = HINT_TEMPLATE
    hint_coord_decimals: int = 2

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
            hint = (
                format_bbox_hint(
                    bbox, self.hint_template, decimals=self.hint_coord_decimals
                )
                if bbox is not None
                else ""
            )
            message = build_hint_teacher_messages(
                problem,
                image,
                hint,
                system_prompt=self.system_prompt,
                suffix=self.opd_prompt_suffix,
            )
            teacher_messages.append(message)
            teacher_prompt_texts.append(
                self.processor.apply_chat_template(
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
