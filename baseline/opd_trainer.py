"""Trainer for vanilla multimodal On-Policy Distillation (OPD).

OPD = the student samples an on-policy rollout from the (non-privileged) dataset
prompt, and a *separate, frozen, stronger* teacher model scores the same
prompt+completion. The student minimizes per-token reverse KL ``KL(student||teacher)``
over the full completion.

This differs from ViGOS / OPSD, where the "teacher" is the same weights with the
LoRA adapter disabled and a *privileged* prompt that contains the reference
answer. Here the teacher is a genuinely different checkpoint and never sees the
answer.

``OPDTrainer`` subclasses :class:`~vigos.trainer.ViGOSTrainer` purely to reuse its
machinery (on-policy vLLM/HF rollout, batched teacher forward pass, exact
full-vocabulary masked KL, DDP loss normalization, answer-accuracy metrics). Only
``compute_loss`` is replaced.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from vigos.losses import masked_kl_loss
from vigos.trainer import ViGOSTrainer


class OPDTrainer(ViGOSTrainer):
    def __init__(
        self,
        *args: Any,
        teacher_model: nn.Module | None = None,
        lambda_opd: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if teacher_model is None:
            raise ValueError(
                "OPDTrainer requires a teacher_model (a separate frozen VLM)."
            )
        self.lambda_opd = float(lambda_opd)
        # The teacher is inference-only: no grad, eval mode, and it is NOT wrapped
        # by Accelerate/DeepSpeed and NOT synced into vLLM (only self.model is).
        teacher_model.requires_grad_(False)
        teacher_model.eval()
        self.teacher_model = teacher_model.to(self.accelerator.device)

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        student_prompt = self._prompt_inputs(inputs, "student")
        rollout = self._generate_on_policy(model, student_prompt, inputs)

        completion_ids = rollout["completion_ids"]
        completion_attention = rollout["completion_attention_mask"].to(dtype=torch.bool)

        # Student forward (with gradients) over the sampled completion.
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

        # Teacher forward (frozen, no grad) on the SAME non-privileged prompt +
        # completion. _batched_teacher_completion_logits already runs under
        # torch.no_grad()/eval; for a non-PEFT teacher the adapter-disable context
        # degrades to a no-op, so we can pass the separate teacher model directly.
        teacher_inputs = self._append_completion(
            student_prompt,
            completion_ids,
            rollout["completion_attention_mask"],
        )
        teacher_logits = self._batched_teacher_completion_logits(
            self.teacher_model,
            [
                {
                    "name": "opd",
                    "inputs": teacher_inputs,
                    "completion_length": completion_ids.shape[1],
                }
            ],
        )["opd"]

        # Reverse KL: source=student -> KL(student || teacher), masked to the
        # completion tokens (full completion, mode-seeking).
        opd_loss = masked_kl_loss(
            student_logits,
            teacher_logits,
            completion_attention,
            temperature=self.distill_temperature,
            token_clip=self.token_loss_clip,
        )
        opd_loss, _, opd_loss_numerator, opd_loss_count = (
            self._distributed_masked_loss_with_stats(opd_loss, completion_attention)
        )
        del teacher_logits
        loss = self.lambda_opd * opd_loss

        rollout_answer_correct = self._rollout_answer_correctness(inputs, rollout)
        _, answer_correct_count, answer_count = self._distributed_rate_stats(
            rollout_answer_correct
        )
        _, completion_token_count, completion_token_total = self._distributed_rate_stats(
            completion_attention
        )
        self._record_loss_metrics(
            {
                "loss_opd": (opd_loss_numerator, opd_loss_count),
                "answer_accuracy": (answer_correct_count, answer_count),
                "completion_token_ratio": (
                    completion_token_count,
                    completion_token_total,
                ),
            }
        )

        if return_outputs:
            return loss, {"logits": student_logits.detach()}
        return loss
