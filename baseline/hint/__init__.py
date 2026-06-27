"""Grounding-Hint Distillation (GHD) — privileged-bbox OPD.

On-Policy Distillation where the frozen teacher is handed the GT evidence
bounding box as a **text coordinate hint** ("pay attention to region
``[x1,y1,x2,y2]``") appended to the question, while the student is scored on the
plain ``(image, question)`` prompt and never sees the box. The image is not
cropped or upsampled — the teacher gets *direction* (where to look), not extra
*information* — so the per-token reverse KL ``KL(student‖teacher)`` distils
grounding, not pixels.

This is the "spine" experiment: does an implicit, privileged where-to-look signal
on the teacher move the student's visual-search accuracy (V*Bench)? It is purely
additive — ``vigos/`` and the vanilla OPD files are untouched; GHD subclasses
:class:`~baseline.opd_trainer.OPDTrainer` (teacher-prompt swap only) and
:class:`~baseline.opd_data_collator.OPDDataCollator` (adds the privileged
``teacher_prompt_*``).

Modules:
* :mod:`baseline.hint.opd_hint_collator` — ``OPDHintDataCollator`` + the hint text.
* :mod:`baseline.hint.opd_hint_trainer`  — ``OPDHintTrainer``.

Entry point: ``baseline/train_opd_hint.py`` · launcher: ``scripts/train_opd_hint_qwen3_2b.sh``.
"""

from baseline.hint.opd_hint_collator import (
    HINT_TEMPLATE,
    OPDHintDataCollator,
    build_hint_teacher_messages,
    format_bbox_hint,
)
from baseline.hint.opd_hint_trainer import OPDHintTrainer

__all__ = [
    "HINT_TEMPLATE",
    "OPDHintDataCollator",
    "OPDHintTrainer",
    "build_hint_teacher_messages",
    "format_bbox_hint",
]
