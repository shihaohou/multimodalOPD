"""Trainer for Grounding-Hint Distillation (GHD).

``OPDHintTrainer`` is vanilla :class:`~baseline.opd_trainer.OPDTrainer` with a
single change: the frozen teacher is scored on a **privileged prompt** that
carries the GT evidence bounding box as a text coordinate hint
(``teacher_prompt_*``, built by :class:`OPDHintDataCollator`) instead of on the
plain student prompt. Everything else — the on-policy rollout, the student
forward, the reverse-KL token loss, DDP loss normalization, NaN localization and
the rollout diagnostics — is inherited unchanged.

Why this is the whole method:

* **On-policy.** The completion ``y`` is sampled by the *student* from
  ``(image, question)``. The loss lands on the tokens the student would actually
  emit, not on the teacher's trajectory.
* **Privileged teacher.** The teacher forwards on ``(image, question, bbox-hint, y)``
  → a more *grounded* per-token distribution ``p_T`` (it knows where to look).
  The student forwards on ``(image, question, y)`` → ``p_S``.
* **Distill.** ``KL(student‖teacher)`` per token pulls the un-hinted student toward
  the grounded teacher. The teacher's privilege is *spatial direction only* — same
  image, same resolution — so the signal it adds is "look here", which is exactly
  what we want the student to internalize for visual-search benchmarks (V*Bench).

The completion positions align across the two forwards even though the teacher's
prefix is longer: :meth:`_completion_logits` slices the trailing ``completion``
block from the end of each sequence, so the extra hint tokens (all *before* the
completion) never shift it. ``local_hf`` teacher only (reverse KL needs full
teacher logits; the privileged prompt is also built locally).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from baseline.opd_losses import masked_topk_kl_loss
from baseline.opd_trainer import OPDTrainer


class OPDHintTrainer(OPDTrainer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.teacher_source != "local_hf":
            raise ValueError(
                "OPDHintTrainer requires teacher_source='local_hf': the grounding-hint "
                "teacher is scored on a locally-built privileged prompt and the reverse "
                "KL needs its full logits (a vLLM server returns only top-k logprobs)."
            )

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        student_prompt = self._prompt_inputs(inputs, "student")
        # The privileged prompt: same image + question, plus the evidence-box hint
        # text. Built by OPDHintDataCollator; the student never sees it.
        teacher_prompt = self._prompt_inputs(inputs, "teacher")

        rollout = self._generate_on_policy(model, student_prompt, inputs)
        # compute_loss is overridden wholesale, so the rollout-snapshot hook from
        # ViGOSTrainer.compute_loss is not on this path — call it here (grad-free).
        self._maybe_log_completion_snapshot(inputs, rollout)

        completion_ids = rollout["completion_ids"]
        completion_attention = rollout["completion_attention_mask"].to(dtype=torch.bool)

        # --- Student forward (with gradients) on the NON-privileged prompt --------
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

        # --- Teacher forward (frozen, no grad) on the PRIVILEGED prompt -----------
        # Same student completion, scored under (image, question, bbox-hint). The
        # extra hint tokens sit before the completion, so _completion_logits' tail
        # slice still lines up token-for-token with the student logits above.
        teacher_inputs = self._append_completion(
            teacher_prompt,
            completion_ids,
            rollout["completion_attention_mask"],
        )
        teacher_logits = self._batched_teacher_completion_logits(
            self.teacher_model,
            [
                {
                    "name": "ghd",
                    "inputs": teacher_inputs,
                    "completion_length": completion_ids.shape[1],
                }
            ],
        )["ghd"]

        # Truncate both to the shared (min) padded vocab; fp32 for KL safety.
        vocab = min(student_logits.shape[-1], teacher_logits.shape[-1])
        student_kl_logits = student_logits[..., :vocab].float()
        teacher_kl_logits = teacher_logits[..., :vocab].float()
        del teacher_logits
        opd_loss = masked_topk_kl_loss(
            student_kl_logits,
            teacher_kl_logits,
            completion_attention,
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
                completion_attention,
                completion_ids,
            )

        opd_loss, _, opd_loss_numerator, opd_loss_count = (
            self._distributed_masked_loss_with_stats(opd_loss, completion_attention)
        )
        loss = self.lambda_opd * opd_loss

        # --- metrics -------------------------------------------------------------
        rollout_answer_correct = self._rollout_answer_correctness(inputs, rollout)
        _, answer_correct_count, answer_count = self._distributed_rate_stats(
            rollout_answer_correct
        )
        _, completion_token_count, completion_token_total = self._distributed_rate_stats(
            completion_attention
        )
        seq_indicator = completion_attention.new_ones(
            (completion_attention.shape[0],), dtype=torch.float32
        )
        _, num_sequences, _ = self._distributed_rate_stats(seq_indicator)
        metrics: dict[str, tuple[float, float]] = {
            "loss_opd": (opd_loss_numerator, opd_loss_count),
            "answer_accuracy": (answer_correct_count, answer_count),
            "completion_length": (completion_token_count, num_sequences),
            "completion_token_ratio": (completion_token_count, completion_token_total),
        }
        # Fraction of the batch that actually carried a parseable evidence box (and
        # thus a real hint). <1.0 means some rows fell back to vanilla OPD; watch
        # this to confirm the privileged signal is reaching most of the batch.
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
