"""Data collation for Locate-Once Grounding (LOG).

``OPDLocateDataCollator`` is :class:`~baseline.hint.opd_hint_collator.OPDHintDataCollator`
with two additions on top of the hidden-hint spine:

1. **Asymmetric prompts.** The *student* gets a **locate-once** system prompt
   (:data:`LOCATE_SYSTEM_PROMPT`): begin ``<think>`` by emitting a single
   ``<box>[x1,y1,x2,y2]</box>``, then reason and ``\\boxed{}`` the answer. The
   *teacher* keeps the plain think prompt and is handed the GT box silently through
   the hint (``teacher_system_prompt`` decoupling) — it is the privileged, more
   grounded distribution the student is pulled toward, and it must never be told to
   locate or it would verbalize coordinates the un-hinted student cannot reproduce.

2. **Group rollout.** Each feature is replicated ``group_size`` times (contiguous
   blocks) so the on-policy rollout draws ``G`` samples per prompt — the group GRPO
   needs to compute a baseline. ``group_ids`` marks the blocks; ``locate_gt_boxes``
   carries the per-row GT evidence box (normalized) for the IoU reward.

The student NEVER sees the box (no hint, no crop) — the ``<box>`` it emits is its own
commitment, scored by RL, not an input. Everything else (image safety, teacher hint
construction, ``has_hint`` coverage) is inherited unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from baseline.hint.opd_hint_collator import OPDHintDataCollator
from baseline.locate.prompts import LOCATE_SYSTEM_PROMPT
from baseline.probe.saliency_data import parse_bbox_norm

# Re-exported for callers that import it from the collator (entry point, sanity check).
__all__ = ["LOCATE_SYSTEM_PROMPT", "OPDLocateDataCollator"]


@dataclass
class OPDLocateDataCollator(OPDHintDataCollator):
    """Locate-once student prompt + hidden-hint teacher prompt + group rollout."""

    # Rollouts sampled per prompt for the GRPO baseline. The data loader yields
    # `per_device_train_batch_size` PROMPTS; this collator expands each to `group_size`
    # contiguous rows, so a micro-batch holds whole groups (the GRPO advantage is
    # computed within `group_ids`). Effective rollouts/step = per_device_bs *
    # group_size * grad_accum * world_size.
    group_size: int = 8

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {self.group_size}.")
        # The student must be told to emit a <box>; if the caller forgot to swap the
        # system prompt the whole RL handle is missing. Don't silently train box-free.
        if "<box>" not in (self.system_prompt or ""):
            raise ValueError(
                "OPDLocateDataCollator.system_prompt must instruct the student to emit "
                "a <box>...</box> (use LOCATE_SYSTEM_PROMPT). The teacher prompt is set "
                "separately via teacher_system_prompt."
            )

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        # Replicate each prompt into a contiguous group BEFORE building anything, so
        # the student/teacher encodes, images and metadata all line up row-for-row.
        if self.group_size > 1:
            features = [feature for feature in features for _ in range(self.group_size)]

        result = super().__call__(features)

        batch_size = len(features)
        # Contiguous group blocks: rows [g*G : (g+1)*G] share a prompt. Robust even
        # when sample_ids collide / are absent (the base collator derives sample_id
        # from problem_id-or-local-index, which differs across replicas).
        result["group_ids"] = torch.arange(batch_size, dtype=torch.long) // self.group_size
        # Per-row GT evidence box (normalized) for the IoU reward; None where absent.
        result["locate_gt_boxes"] = [
            parse_bbox_norm(feature.get(self.bbox_field)) for feature in features
        ]
        return result
