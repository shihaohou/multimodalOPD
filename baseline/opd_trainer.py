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

from baseline.opd_losses import (
    masked_topk_kl_loss,
    masked_topk_kl_loss_from_teacher_topk,
)
from vigos.losses import masked_kl_loss
from vigos.trainer import ViGOSTrainer


class OPDTrainer(ViGOSTrainer):
    def __init__(
        self,
        *args: Any,
        teacher_model: nn.Module | None = None,
        teacher_source: str = "local_hf",
        teacher_client: Any = None,
        lambda_opd: float = 1.0,
        opd_loss_mode: str = "full_kl",
        opd_kl_direction: str = "reverse",
        opd_top_k: int = 32,
        opd_mask_invalid_vocab: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if teacher_source not in {"local_hf", "vllm_server"}:
            raise ValueError(
                f"Unknown teacher_source {teacher_source!r}; "
                "use 'local_hf' or 'vllm_server'."
            )
        if opd_loss_mode not in {"full_kl", "topk_kl"}:
            raise ValueError(
                f"Unknown opd_loss_mode {opd_loss_mode!r}; use 'full_kl' or 'topk_kl'."
            )
        if opd_kl_direction not in {"reverse", "forward", "jsd"}:
            raise ValueError(
                f"Unknown opd_kl_direction {opd_kl_direction!r}; "
                "use 'reverse', 'forward', or 'jsd'."
            )
        self.lambda_opd = float(lambda_opd)
        self.opd_loss_mode = opd_loss_mode
        self.opd_kl_direction = opd_kl_direction
        self.opd_top_k = int(opd_top_k)
        self.opd_mask_invalid_vocab = bool(opd_mask_invalid_vocab)
        self.teacher_source = teacher_source
        self.teacher_client = teacher_client
        self.teacher_model = None
        # IDs never valid as assistant text + real tokenizer length, used to drop
        # padded/control columns from the completion-KL support (see compute_loss).
        self._invalid_vocab_ids, self._tokenizer_vocab_len = (
            self._collect_invalid_vocab_ids()
        )
        self._valid_vocab_cache: dict[tuple[int, torch.device], torch.Tensor] = {}

        if teacher_source == "local_hf":
            if teacher_model is None:
                raise ValueError("teacher_source='local_hf' requires a teacher_model.")
            # Inference-only: no grad, eval, NOT wrapped by Accelerate/DeepSpeed and
            # NOT synced into vLLM (only self.model is). Replicated per GPU.
            teacher_model.requires_grad_(False)
            teacher_model.eval()
            self.teacher_model = teacher_model.to(self.accelerator.device)
        else:  # vllm_server: no per-GPU replica; teacher returns top-k logprobs.
            if teacher_client is None:
                raise ValueError(
                    "teacher_source='vllm_server' requires a teacher_client."
                )
            if not (self.opd_loss_mode == "topk_kl" and self.opd_kl_direction == "forward"):
                raise ValueError(
                    "vllm_server teacher only supports opd_loss_mode='topk_kl' with "
                    "opd_kl_direction='forward' (the server returns top-k logprobs)."
                )

    def _collect_invalid_vocab_ids(self) -> tuple[set[int], int | None]:
        """Vocab columns that are never valid assistant-text outputs.

        Returns ``(ids, tokenizer_len)``: ``ids`` are the non-text multimodal
        control tokens (vision/image/video placeholders) plus pad (unless it
        doubles as EOS); ``tokenizer_len`` is the real tokenizer length so the
        padded lm_head rows above it can be dropped too. The teacher assigns all
        of these ~0 probability, so any student mass there is a sharp, exploding
        reverse-KL tail term — they must leave the KL *support*, not just the
        supervised positions.
        """
        ids: set[int] = set()
        config = getattr(self.model, "config", None)
        for name in (
            "image_token_id",
            "video_token_id",
            "vision_start_token_id",
            "vision_end_token_id",
            "image_pad_token_id",
            "video_pad_token_id",
            "vision_token_id",
        ):
            tok_id = getattr(config, name, None)
            if isinstance(tok_id, int) and tok_id >= 0:
                ids.add(tok_id)
        pad_id = self._pad_token_id()
        eos_ids = set(self._normalize_eos_token_ids(self._eos_token_id()))
        if isinstance(pad_id, int) and pad_id not in eos_ids:
            ids.add(pad_id)
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        try:
            tok_len: int | None = int(len(tokenizer))
        except TypeError:
            tok_len = None
        return ids, tok_len

    def _valid_vocab_mask(
        self, vocab: int, device: torch.device
    ) -> torch.Tensor | None:
        """Bool ``[vocab]`` mask (True = keep in the KL), cached per (vocab, device).

        Returns ``None`` when ``opd_mask_invalid_vocab`` is off.
        """
        if not self.opd_mask_invalid_vocab:
            return None
        key = (int(vocab), torch.device(device))
        cached = self._valid_vocab_cache.get(key)
        if cached is not None:
            return cached
        valid = torch.ones(int(vocab), dtype=torch.bool, device=device)
        if self._tokenizer_vocab_len is not None and self._tokenizer_vocab_len < vocab:
            valid[self._tokenizer_vocab_len :] = False
        for tok_id in self._invalid_vocab_ids:
            if 0 <= tok_id < vocab:
                valid[tok_id] = False
        if not bool(valid.any()):
            raise ValueError("valid_vocab_mask excluded every column.")
        self._valid_vocab_cache[key] = valid
        return valid

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

        if self.teacher_source == "vllm_server":
            # Server returns the teacher's top-k token ids + logprobs at each
            # completion position (vLLM prompt_logprobs); forward top-k KL only.
            teacher_topk_ids, teacher_topk_logprobs = self.teacher_client.score_topk(
                student_prompt["input_ids"],
                student_prompt["attention_mask"],
                completion_ids,
                rollout["completion_attention_mask"],
                inputs.get("student_images") or [],
            )
            opd_loss = masked_topk_kl_loss_from_teacher_topk(
                student_logits.float(),
                teacher_topk_ids,
                teacher_topk_logprobs,
                completion_attention,
                temperature=self.distill_temperature,
                token_clip=self.token_loss_clip,
            )
        else:
            # Local frozen teacher forward (full logits). full_kl+reverse uses the
            # exact full-vocab path (vigos.losses.masked_kl_loss); everything else
            # (top-k, forward, jsd) goes through masked_topk_kl_loss (top_k=None
            # recovers exact full-vocab KL for the forward/jsd full_kl cases).
            # _batched_teacher_completion_logits runs under no_grad/eval; for a
            # non-PEFT teacher the adapter-disable context is a no-op.
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
            # Same-family checkpoints can have different padded vocab sizes (e.g.
            # Qwen2.5-VL 3B=151936 vs 7B=152064). Truncate both to the shared (min)
            # vocab; fp32 for KL numerical safety (the bf16 p·log p entropy term
            # explodes when a student prob underflows to exactly 0).
            vocab = min(student_logits.shape[-1], teacher_logits.shape[-1])
            student_kl_logits = student_logits[..., :vocab].float()
            teacher_kl_logits = teacher_logits[..., :vocab].float()
            del teacher_logits
            # Drop vocab columns that are never valid assistant-text outputs (padded
            # lm_head rows, pad, vision/image/video control tokens) from the KL
            # *support* before log_softmax. Full-vocab reverse KL otherwise spends a
            # sharp, exploding tail term on those columns (teacher ~0 prob, small
            # student mass) -> grad blow-up / NaN. This is distinct from the position
            # mask (completion_attention), which says which positions are supervised.
            valid = self._valid_vocab_mask(vocab, student_kl_logits.device)
            if valid is not None:
                fill = ~valid.view(1, 1, -1)
                student_kl_logits = student_kl_logits.masked_fill(fill, -1e9)
                teacher_kl_logits = teacher_kl_logits.masked_fill(fill, -1e9)
            # Route through masked_topk_kl_loss so the verl-style log-prob diff clamp
            # (gradient bound) also covers full_kl + reverse; top_k=None = exact
            # full-vocab KL over the surviving support.
            opd_loss = masked_topk_kl_loss(
                student_kl_logits,
                teacher_kl_logits,
                completion_attention,
                top_k=self.opd_top_k if self.opd_loss_mode == "topk_kl" else None,
                direction=self.opd_kl_direction,
                temperature=self.distill_temperature,
                token_clip=self.token_loss_clip,
            )
        opd_loss, _, opd_loss_numerator, opd_loss_count = (
            self._distributed_masked_loss_with_stats(opd_loss, completion_attention)
        )
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
