"""Grounding-Hint Distillation (GHD) — privileged-bbox OPD.

On-Policy Distillation where the frozen teacher is privileged with the GT evidence
bounding box while the student is scored on the plain ``(image, question)`` prompt
and never sees the box. Two privilege channels (``teacher_privilege_mode``):

* ``hint`` — full image + the box as a **text coordinate hint** (*direction*:
  where to look). The spine experiment.
* ``crop`` — the image **cropped to the box** (*zoom*: a higher-resolution view of
  the evidence; more information on high-res inputs / the V*Bench regime).

Both distil the teacher's per-token distribution into the student via reverse KL
``KL(student‖teacher)`` on the student's on-policy rollout. The GT box is fixed, so
the student forward backprops normally (no RL). Purely additive — ``vigos/`` and the
vanilla OPD files are untouched; GHD subclasses
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
    crop_to_bbox,
    format_bbox_hint,
)
from baseline.hint.opd_hint_trainer import OPDHintTrainer

__all__ = [
    "HINT_TEMPLATE",
    "OPDHintDataCollator",
    "OPDHintTrainer",
    "build_hint_teacher_messages",
    "crop_to_bbox",
    "format_bbox_hint",
]
