"""Trainer for Locate-Once Grounding (LOG).

``OPDLocateTrainer`` is :class:`~baseline.hint.opd_hint_trainer.OPDHintTrainer` (the
verified hidden-hint OPD spine) plus an explicit, student-generated evidence box
trained by RL. One on-policy step, two **span-decoupled** gradient sources on a
single student forward:

```
L = lambda_opd * L_OPD(answer/reasoning span, box span MASKED)
  + lambda_rl  * L_RL(box coordinate span)
```

* **L_OPD** — per-token ``KL(student||teacher)`` exactly as GHD, but the student's
  ``<box>...</box>`` span is removed from the loss mask. The teacher runs the
  *hidden-hint* prompt (silently handed the GT box, told not to verbalize it) and so
  emits **no** box; scoring the student's box tokens under it would push the student
  to *stop* emitting boxes — directly fighting the RL term. Masking the box span
  leaves OPD to teach "how to answer as if you knew where to look".

* **L_RL** — GRPO. The reward is ``IoU(student_box, GT_box)`` gated by answer
  correctness (DeepEyes-style conditional tool reward: only a correct rollout earns
  its localization credit). Group-normalized over the ``group_size`` rollouts of each
  prompt → advantage; policy gradient ``-A * log pi`` on the box coordinate tokens
  only. This teaches "where to look". The box does not change the pixels the student
  sees (no crop), so the only thing RL can move is the model's internal attention.

The two *supervised output positions* are disjoint (OPD on non-box tokens, RL on
box-coordinate tokens; the literal ``<box>``/``</box>`` tags get neither), so no token
receives a direct OPD and RL gradient at once. (They are not fully independent: the
answer/reasoning tokens still attend back to the box text in context, so OPD's
later-token loss is implicitly conditioned on the student's box — the box is kept in
context and only removed from the *loss*.) ``local_hf`` teacher only (reverse KL needs
full teacher logits).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from baseline.hint.opd_hint_trainer import OPDHintTrainer
from baseline.locate.locate_rl import (
    BOX_CLOSE,
    BOX_OPEN,
    group_normalize_advantage,
    iou_norm,
    sampled_token_logprobs,
)
from baseline.opd_losses import masked_topk_kl_loss
from baseline.probe.saliency_data import parse_bbox_norm


class OPDLocateTrainer(OPDHintTrainer):
    def __init__(
        self,
        *args: Any,
        lambda_rl: float = 0.5,
        rl_reward: str = "gated_iou",
        rl_ungated_weight: float = 0.0,
        rl_normalize_adv: bool = True,
        rl_adv_eps: float = 1e-6,
        group_size: int = 8,
        kl_position_gate: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if rl_reward not in {"gated_iou", "iou"}:
            raise ValueError(
                f"Unknown rl_reward {rl_reward!r}; use 'gated_iou' (IoU gated by answer "
                "correctness, the default) or 'iou' (ungated, the ablation control)."
            )
        if group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {group_size}.")
        self.lambda_rl = float(lambda_rl)
        self.rl_reward = rl_reward
        # Early-training warmup (gated_iou only): add `rl_ungated_weight * IoU` so a box
        # with good overlap still earns a (small) signal even when the answer is wrong —
        # mitigates reward sparsity before answer accuracy rises. 0.0 = pure gated.
        self.rl_ungated_weight = float(rl_ungated_weight)
        self.rl_normalize_adv = bool(rl_normalize_adv)
        self.rl_adv_eps = float(rl_adv_eps)
        self.group_size = int(group_size)
        # Deferred (off by default, per plan): apply the OPD KL only where the teacher
        # assigns the sampled token higher logprob than the student — the
        # evidence-dependent tokens. Available for ablation once the spine shows signal.
        self.kl_position_gate = bool(kl_position_gate)

    # ------------------------------------------------------------------ box spans
    def _locate_row_box(
        self, row: torch.Tensor, valid_length: int
    ) -> tuple[tuple[int, int] | None, tuple[int, int] | None, Any, bool]:
        """Find the first ``<box>...</box>`` in one completion row.

        Returns ``(full_span, coord_span, box_norm, late)``:
        * ``full_span``  — token range to MASK from the OPD loss (``<box>`` tag, plus
          coords + ``</box>`` when the box closes), or None if no ``<box>`` at all;
        * ``coord_span`` — token range the RL term reinforces (the coordinates only),
          or None when the box is malformed (no close) or *late* (no RL handle);
        * ``box_norm``   — the parsed normalized ``(x1,y1,x2,y2)`` (None if unparseable
          or late);
        * ``late``       — True when the box appears at/after ``\\boxed{}`` ("answer then
          locate", not "locate then answer"). A late box is still masked from OPD but
          earns no RL/reward, so RL can't reinforce locating *after* the fact.

        A box with an open tag but no close still masks the tag from OPD (the
        hidden-hint teacher never emits it) but yields no RL handle / box value.
        """
        open_span = self._find_tag_span(row, valid_length, BOX_OPEN)
        if open_span is None:
            return None, None, None, False
        close_span = self._find_tag_span(
            row, valid_length, BOX_CLOSE, start=open_span[1]
        )
        if close_span is None or close_span[0] < open_span[1]:
            return (open_span[0], open_span[1]), None, None, False
        full_span = (open_span[0], close_span[1])
        answer_span = self._find_tag_span(row, valid_length, "\\boxed{")
        if answer_span is not None and open_span[0] >= answer_span[0]:
            return full_span, None, None, True  # late box: mask from OPD, no RL
        coord_span = (open_span[1], close_span[0])
        inner_text = self._decode_token_ids(
            row[coord_span[0] : coord_span[1]], skip_special_tokens=False
        )
        return full_span, coord_span, parse_bbox_norm(inner_text), False

    def _locate_box_masks(
        self,
        completion_ids: torch.Tensor,
        completion_attention: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[Any], torch.Tensor, torch.Tensor]:
        """Per-row box masks + parsed boxes + diagnostics.

        Returns ``(box_full_mask, box_coord_mask, student_boxes, box_present, box_late)``:
        ``box_full_mask`` is masked OUT of OPD; ``box_coord_mask`` is what RL reinforces
        (both restricted to attended tokens); ``box_present`` flags rows that emitted a
        ``<box>`` tag (vs. no box at all — distinguishes "didn't emit" from "malformed"
        in the metrics); ``box_late`` flags boxes after ``\\boxed{}``.
        """
        batch_size, completion_length = completion_ids.shape
        device = completion_ids.device
        box_full_mask = torch.zeros(
            (batch_size, completion_length), dtype=torch.bool, device=device
        )
        box_coord_mask = torch.zeros(
            (batch_size, completion_length), dtype=torch.bool, device=device
        )
        box_present = torch.zeros(batch_size, dtype=torch.bool, device=device)
        box_late = torch.zeros(batch_size, dtype=torch.bool, device=device)
        student_boxes: list[Any] = []
        for row_idx in range(batch_size):
            valid_length = int(completion_attention[row_idx].sum().item())
            full_span, coord_span, box, late = self._locate_row_box(
                completion_ids[row_idx], valid_length
            )
            if full_span is not None:
                box_full_mask[row_idx, full_span[0] : full_span[1]] = True
                box_present[row_idx] = True
            if coord_span is not None and coord_span[1] > coord_span[0]:
                box_coord_mask[row_idx, coord_span[0] : coord_span[1]] = True
            box_late[row_idx] = late
            student_boxes.append(box)
        box_full_mask &= completion_attention
        box_coord_mask &= completion_attention
        return box_full_mask, box_coord_mask, student_boxes, box_present, box_late

    # --------------------------------------------------------------------- reward
    def _locate_rewards(
        self,
        inputs: dict[str, Any],
        student_boxes: list[Any],
        answer_correct: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(rewards, iou_vals, has_box)`` per rollout (all 1-D, length B).

        ``rewards`` is the RL signal: IoU(student_box, GT_box), gated by answer
        correctness when ``rl_reward='gated_iou'``. ``iou_vals`` is the raw IoU (0
        where there is no student box or no GT box) for logging; ``has_box`` flags
        rollouts that emitted a parseable box.
        """
        batch_size = len(student_boxes)
        gt_boxes = self._metadata_values(inputs.get("locate_gt_boxes"), batch_size)
        rewards = torch.zeros(batch_size, dtype=torch.float32, device=device)
        iou_vals = torch.zeros(batch_size, dtype=torch.float32, device=device)
        has_box = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for row_idx in range(batch_size):
            student_box = student_boxes[row_idx]
            if student_box is None:
                continue
            has_box[row_idx] = True
            gt_box = gt_boxes[row_idx]
            iou = iou_norm(student_box, gt_box) if gt_box is not None else 0.0
            iou_vals[row_idx] = iou
            if self.rl_reward == "gated_iou":
                gated = iou if bool(answer_correct[row_idx]) else 0.0
                # Optional warmup: a small ungated IoU term keeps a signal alive while
                # answer accuracy (the gate) is still low. 0.0 => pure gated.
                rewards[row_idx] = gated + self.rl_ungated_weight * iou
            else:  # "iou" — ungated ablation control
                rewards[row_idx] = iou
        return rewards, iou_vals, has_box

    # ------------------------------------------------------------------- RL loss
    def _box_rl_loss(
        self,
        student_logits: torch.Tensor,
        completion_ids: torch.Tensor,
        box_coord_mask: torch.Tensor,
        advantage: torch.Tensor,
    ) -> tuple[torch.Tensor, float, float]:
        """REINFORCE-with-group-baseline PG loss on the box coordinate tokens.

        ``-advantage * log pi(token)`` token-mean over the box-coordinate mask, then
        DDP-renormalized by global box-token count (same path as the KL term so sparse
        masks stay correctly weighted). vLLM is resynced to the policy every step, so
        the sampling policy ~= the gradient policy and a plain on-policy PG (no
        importance ratio) is valid. ``advantage`` is detached (a weight, not a target).

        Pass the FULL (untruncated) student logits — the action is the student's own
        sample, so ``log pi`` must normalize over the student's whole vocab, not the
        teacher-shared slice used for the KL. Only the box-coordinate positions are
        gathered out (and cast to fp32) before the softmax-normalizer, so this stays
        cheap (≈ #box tokens × vocab) regardless of completion length.
        """
        mask = box_coord_mask.to(device=student_logits.device, dtype=torch.bool)
        rows, cols = mask.nonzero(as_tuple=True)
        if rows.numel() == 0:
            # No box tokens this micro-batch. Keep a graph-connected zero so DDP
            # backward does not deadlock on a rank with no boxes.
            local_loss = student_logits.flatten()[0] * 0.0
        else:
            sel_logits = student_logits[rows, cols].float()  # [N_box, V], full vocab
            sel_ids = completion_ids[rows, cols].to(device=sel_logits.device)
            token_logp = sampled_token_logprobs(sel_logits, sel_ids)  # [N_box]
            adv = advantage.detach().to(device=sel_logits.device, dtype=torch.float32)
            per_token = -(adv[rows] * token_logp)  # maximize adv*logp
            local_loss = per_token.sum() / per_token.new_tensor(
                float(per_token.numel())
            ).clamp_min(1.0)
        objective, _, numerator, count = self._distributed_masked_loss_with_stats(
            local_loss, mask
        )
        return objective, numerator, count

    @torch.no_grad()
    def _teacher_gt_student_mask(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        completion_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Position gate: True where the teacher gives the sampled token higher logprob
        than the student (the evidence-dependent tokens). Both logits are already at the
        shared (min) vocab. Grad-free — the result is only used to refine a loss mask."""
        vocab = min(student_logits.shape[-1], teacher_logits.shape[-1])
        ids = completion_ids.clamp(0, vocab - 1).unsqueeze(-1)
        student = student_logits[..., :vocab]
        teacher = teacher_logits[..., :vocab]
        student_lp = student.gather(-1, ids).squeeze(-1) - student.logsumexp(dim=-1)
        teacher_lp = teacher.gather(-1, ids).squeeze(-1) - teacher.logsumexp(dim=-1)
        return teacher_lp > student_lp

    # -------------------------------------------------------------------- loss
    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        student_prompt = self._prompt_inputs(inputs, "student")  # locate-once prompt
        teacher_prompt = self._prompt_inputs(inputs, "teacher")  # hidden-hint prompt

        rollout = self._generate_on_policy(model, student_prompt, inputs)
        self._maybe_log_completion_snapshot(inputs, rollout)

        completion_ids = rollout["completion_ids"]
        completion_attention = rollout["completion_attention_mask"].to(dtype=torch.bool)

        # Box spans from the sampled completion (one pass; no model needed).
        box_full_mask, box_coord_mask, student_boxes, box_present, box_late = (
            self._locate_box_masks(completion_ids, completion_attention)
        )

        # --- Student forward (with gradients) on the locate-once prompt -----------
        student_inputs = self._with_completion(
            student_prompt,
            full_input_ids=rollout["generated_ids"],
            full_attention_mask=rollout["generated_attention_mask"],
        )
        student_inputs["logits_to_keep"] = completion_ids.shape[1] + 1
        student_outputs = model(**student_inputs)
        student_logits = self._completion_logits(
            student_outputs.logits, completion_ids.shape[1]
        )
        del student_outputs

        # --- Teacher forward (frozen, no grad) on the hidden-hint prompt ----------
        teacher_inputs = self._append_completion(
            teacher_prompt,
            completion_ids,
            rollout["completion_attention_mask"],
        )
        teacher_logits = self._batched_teacher_completion_logits(
            self.teacher_model,
            [
                {
                    "name": "locate",
                    "inputs": teacher_inputs,
                    "completion_length": completion_ids.shape[1],
                }
            ],
        )["locate"]
        vocab = min(student_logits.shape[-1], teacher_logits.shape[-1])
        student_kl_logits = student_logits[..., :vocab].float()
        teacher_kl_logits = teacher_logits[..., :vocab].float()
        del teacher_logits

        # --- OPD KL on the completion EXCEPT the box span -------------------------
        opd_mask = completion_attention & (~box_full_mask)
        if self.kl_position_gate:
            opd_mask = opd_mask & self._teacher_gt_student_mask(
                student_kl_logits, teacher_kl_logits, completion_ids
            )
        opd_loss = masked_topk_kl_loss(
            student_kl_logits,
            teacher_kl_logits,
            opd_mask,
            top_k=self.opd_top_k if self.opd_loss_mode == "topk_kl" else None,
            direction=self.opd_kl_direction,
            temperature=self.distill_temperature,
            token_clip=self.token_loss_clip,
        )
        if not bool(torch.isfinite(opd_loss.detach())):
            self._report_opd_nan(
                model,
                student_logits,
                student_kl_logits,
                teacher_kl_logits,
                opd_mask,
                completion_ids,
            )
        opd_loss, _, opd_loss_numerator, opd_loss_count = (
            self._distributed_masked_loss_with_stats(opd_loss, opd_mask)
        )

        # --- Box RL (GRPO) on the coordinate span --------------------------------
        answer_correct = self._rollout_answer_correctness(inputs, rollout)
        rewards, iou_vals, has_box = self._locate_rewards(
            inputs, student_boxes, answer_correct, completion_ids.device
        )
        group_ids = inputs.get("group_ids")
        if group_ids is None:
            # No group structure (e.g. group_size=1) → no baseline → advantage 0.
            group_ids = torch.arange(rewards.shape[0], device=rewards.device)
        advantage = group_normalize_advantage(
            rewards,
            group_ids,
            normalize_std=self.rl_normalize_adv,
            eps=self.rl_adv_eps,
        )
        # Full student logits (NOT the teacher-shared-vocab slice): log pi of the
        # student's own sampled box token must normalize over the student's whole vocab.
        rl_loss, rl_loss_numerator, rl_loss_count = self._box_rl_loss(
            student_logits, completion_ids, box_coord_mask, advantage
        )

        loss = self.lambda_opd * opd_loss + self.lambda_rl * rl_loss

        # --- metrics -------------------------------------------------------------
        _, answer_correct_count, answer_count = self._distributed_rate_stats(
            answer_correct
        )
        _, completion_token_count, completion_token_total = self._distributed_rate_stats(
            completion_attention
        )
        seq_indicator = completion_attention.new_ones(
            (completion_attention.shape[0],), dtype=torch.float32
        )
        _, num_sequences, _ = self._distributed_rate_stats(seq_indicator)
        _, reward_sum, reward_count = self._distributed_rate_stats(rewards)
        _, box_sum, box_count = self._distributed_rate_stats(has_box.to(torch.float32))
        _, adv_abs_sum, adv_abs_count = self._distributed_rate_stats(advantage.abs())
        _, iou_sum, iou_count = self._distributed_rate_stats(iou_vals[has_box])
        boxed_correct = has_box & answer_correct.to(device=has_box.device, dtype=torch.bool)
        _, iou_correct_sum, iou_correct_count = self._distributed_rate_stats(
            iou_vals[boxed_correct]
        )
        # Box-emission health: present (any <box> tag) vs coverage (a fully parsed valid
        # box) splits "didn't emit" from "malformed"; late = box after \boxed{} (no RL).
        _, present_sum, present_count = self._distributed_rate_stats(
            box_present.to(torch.float32)
        )
        _, late_sum, late_count = self._distributed_rate_stats(box_late.to(torch.float32))
        # RL-signal density: fraction of rollouts whose group gave a non-zero advantage
        # (0 for singleton or all-equal-reward groups → no gradient that row).
        _, nonzero_adv_sum, nonzero_adv_count = self._distributed_rate_stats(
            (advantage.abs() > 1e-6).to(torch.float32)
        )
        # Mean area of parsed boxes (collapse monitor: a crash toward a constant
        # center/whole-image box shows up here and in its variance across steps).
        box_areas = torch.tensor(
            [(b[2] - b[0]) * (b[3] - b[1]) for b in student_boxes if b is not None],
            dtype=torch.float32,
            device=completion_ids.device,
        )
        _, area_sum, area_count = self._distributed_rate_stats(box_areas)
        metrics: dict[str, tuple[float, float]] = {
            "loss_opd": (opd_loss_numerator, opd_loss_count),
            "loss_rl": (rl_loss_numerator, rl_loss_count),
            "answer_accuracy": (answer_correct_count, answer_count),
            "completion_length": (completion_token_count, num_sequences),
            "completion_token_ratio": (completion_token_count, completion_token_total),
            "reward_mean": (reward_sum, reward_count),
            # box_present = emitted a <box> tag; box_coverage = parsed a valid box.
            "box_present_rate": (present_sum, present_count),
            "box_coverage": (box_sum, box_count),
            "late_box_rate": (late_sum, late_count),
            "advantage_abs_mean": (adv_abs_sum, adv_abs_count),
            "nonzero_adv_rate": (nonzero_adv_sum, nonzero_adv_count),
            "mean_box_area": (area_sum, area_count),
            # IoU over rollouts that emitted a box; iou_correct over correct+boxed ones
            # (the actual gated reward signal — watch this rise if grounding improves).
            "iou_mean": (iou_sum, iou_count),
            "iou_correct_mean": (iou_correct_sum, iou_correct_count),
        }
        has_hint = inputs.get("has_hint")
        if has_hint is not None:
            _, hint_count, hint_total = self._distributed_rate_stats(
                has_hint.to(device=completion_attention.device, dtype=torch.float32)
            )
            metrics["hint_coverage"] = (hint_count, hint_total)
        metrics.update(
            self._rollout_diagnostic_metrics(
                completion_ids,
                completion_attention,
                student_kl_logits,
                teacher_kl_logits,
                completion_token_count,
                num_sequences,
            )
        )
        self._record_loss_metrics(metrics)

        if return_outputs:
            return loss, {"logits": student_logits.detach()}
        return loss
