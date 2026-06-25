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
        opd_loss_mode: str = "topk_kl",
        opd_kl_direction: str = "reverse",
        opd_top_k: int = 100,
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
        self.teacher_source = teacher_source
        self.teacher_client = teacher_client
        self.teacher_model = None

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

    def _report_opd_nan(
        self,
        model: nn.Module,
        student_logits: torch.Tensor,
        student_kl_logits: torch.Tensor,
        teacher_kl_logits: torch.Tensor,
        completion_attention: torch.Tensor,
        completion_ids: torch.Tensor,
    ) -> None:
        """Localize a non-finite OPD loss: is the source the student forward, the
        teacher forward, or the KL math, and at which completion token? The ±20
        diff clamp bounds the KL math, so a NaN here almost always means a forward
        produced inf/NaN logits. Also scans the student parameters to tell a
        corrupted optimizer update (weights already NaN) apart from a forward-only
        overflow on this rollout. Runs only on a NaN step (caller-guarded)."""
        rank = getattr(self.accelerator, "process_index", 0)
        with torch.no_grad():
            s_fwd = bool(torch.isfinite(student_logits).all())
            t_fwd = bool(torch.isfinite(teacher_kl_logits).all())
            lines = [
                f"[OPD-NaN][rank{rank}] student_forward_finite={s_fwd} "
                f"teacher_forward_finite={t_fwd}",
                f"  |student_logit|max={student_logits.abs().amax().item():.4g} "
                f"|teacher_logit|max={teacher_kl_logits.abs().amax().item():.4g} "
                f"active_tokens={int(completion_attention.sum())}",
            ]
            # Are the student's own parameters still finite? ZeRO-2 replicates
            # params, so this sees the whole model. n_bad>0 => a prior optimizer
            # update corrupted the weights; n_bad==0 => forward-only overflow.
            n_bad_params = 0
            first_bad: list[str] = []
            for name, param in model.named_parameters():
                if not bool(torch.isfinite(param).all()):
                    n_bad_params += 1
                    if len(first_bad) < 6:
                        first_bad.append(name)
            lines.append(
                f"  nonfinite_param_tensors={n_bad_params} first={first_bad}"
            )
            temp = max(self.distill_temperature, 1e-6)
            s_lp = torch.log_softmax(student_kl_logits.float() / temp, dim=-1)
            t_lp = torch.log_softmax(teacher_kl_logits.float() / temp, dim=-1)
            diff = (s_lp - t_lp).clamp(-20.0, 20.0)
            per_tok = (s_lp.exp() * diff).sum(dim=-1)  # [B, C]
            active = completion_attention.to(torch.bool)
            bad = ~torch.isfinite(per_tok) & active
            ok = torch.isfinite(per_tok) & active
            kl_max = per_tok[ok].max().item() if bool(ok.any()) else float("nan")
            lines.append(
                f"  per_token_KL: nonfinite={int(bad.sum())} "
                f"active_max_finite={kl_max:.4g}"
            )
            if bool(bad.any()):
                bi, ci = bad.nonzero(as_tuple=True)
                b0, c0 = int(bi[0]), int(ci[0])
                lines.append(
                    f"  first nonfinite KL at b={b0} c={c0} "
                    f"token_id={int(completion_ids[b0, c0])} "
                    f"student_pos_finite="
                    f"{bool(torch.isfinite(student_kl_logits[b0, c0]).all())} "
                    f"teacher_pos_finite="
                    f"{bool(torch.isfinite(teacher_kl_logits[b0, c0]).all())}"
                )
        print("\n".join(lines), flush=True)

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
            # The server returns only top-k teacher logprobs, not full logits, so
            # the teacher-vs-student rollout curves are unavailable here.
            student_diag_logits = student_logits
            teacher_diag_logits = None
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
            # Route through masked_topk_kl_loss so the verl-style log-prob diff clamp
            # (gradient bound) also covers full_kl + reverse; top_k=None = exact
            # full-vocab KL.
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
            student_diag_logits = student_kl_logits
            teacher_diag_logits = teacher_kl_logits
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
        # Mean response length = total active completion tokens / number of rollout
        # sequences. completion_token_ratio (active/total) sits near 1.0 and is not a
        # length; this is the actual generated-tokens-per-response curve.
        seq_indicator = completion_attention.new_ones(
            (completion_attention.shape[0],), dtype=torch.float32
        )
        _, num_sequences, _ = self._distributed_rate_stats(seq_indicator)
        metrics: dict[str, tuple[float, float]] = {
            # loss_opd is the per-token reverse KL KL(student||teacher) — the KL curve.
            "loss_opd": (opd_loss_numerator, opd_loss_count),
            "answer_accuracy": (answer_correct_count, answer_count),
            "completion_length": (completion_token_count, num_sequences),
            "completion_token_ratio": (
                completion_token_count,
                completion_token_total,
            ),
        }
        metrics.update(
            self._rollout_diagnostic_metrics(
                completion_ids,
                completion_attention,
                student_diag_logits,
                teacher_diag_logits,
                completion_token_count,
                num_sequences,
            )
        )
        self._record_loss_metrics(metrics)

        if return_outputs:
            return loss, {"logits": student_logits.detach()}
        return loss

    @torch.no_grad()
    def _rollout_diagnostic_metrics(
        self,
        completion_ids: torch.Tensor,
        completion_attention: torch.Tensor,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor | None,
        completion_token_count: float,
        num_sequences: float,
    ) -> dict[str, tuple[float, float]]:
        """Per-step rollout health curves under the ``rollout/`` W&B namespace.

        Mirrors the rich rollout logging in the Uni-OPD / miles RL stack: instead
        of only the mean loss we surface what the policy is actually *doing* each
        step — how long and how often-truncated its samples are, how confident vs.
        exploratory it is, and how far the frozen teacher disagrees with the tokens
        the student actually drew.

        Cheap and grad-free: one extra ``log_softmax`` over the completion logits
        we already hold (no second forward). Every value is reduced to a
        ``(global_numerator, global_denominator)`` pair through the same DDP path
        as the loss metrics (:meth:`_distributed_rate_stats`), so sparse masks stay
        correctly weighted. The returned dict is ready to merge into the
        ``_record_loss_metrics`` call.

        ``teacher_logits=None`` (the ``vllm_server`` top-k path, which has no full
        teacher logits) records the student-only curves and skips the teacher ones.
        """
        mask = completion_attention.to(dtype=torch.float32)
        metrics: dict[str, tuple[float, float]] = {}

        # --- response length / truncation ------------------------------------
        # "clipped" = the rollout never emitted EOS inside its active span, i.e.
        # generation stopped on max_new_tokens rather than finishing cleanly. A
        # rising clip ratio means the length budget is throttling the policy.
        eos_ids = self._normalize_eos_token_ids(self._eos_token_id())
        if eos_ids and num_sequences > 0:
            is_eos = torch.zeros_like(completion_attention, dtype=torch.bool)
            for token_id in eos_ids:
                is_eos = is_eos | (completion_ids == token_id)
            finished = (is_eos & completion_attention.to(dtype=torch.bool)).any(dim=1)
            clipped = (~finished).to(dtype=torch.float32)
            _, clipped_sum, _ = self._distributed_rate_stats(clipped)
            metrics["rollout/response_clip_ratio"] = (clipped_sum, num_sequences)

        if completion_token_count <= 0:
            return metrics

        student_logits = student_logits.float()
        # clamp() (not in-place) so we never mutate the shared rollout tensor.
        ids = completion_ids.clamp(0, student_logits.shape[-1] - 1).unsqueeze(-1)

        # --- student policy curves (available for every teacher source) -------
        student_logp = torch.log_softmax(student_logits, dim=-1)
        # log p the student assigns to its own sampled tokens — policy confidence.
        student_tok_logp = student_logp.gather(-1, ids).squeeze(-1) * mask
        _, student_logp_sum, _ = self._distributed_rate_stats(student_tok_logp)
        metrics["rollout/student_logprob"] = (student_logp_sum, completion_token_count)
        # Token-level entropy of the policy on its own samples — exploration signal.
        entropy = -(student_logp.exp() * student_logp).sum(dim=-1) * mask
        _, entropy_sum, _ = self._distributed_rate_stats(entropy)
        metrics["rollout/entropy"] = (entropy_sum, completion_token_count)
        del student_logp, entropy

        # --- teacher-vs-student curves (local full-logit teacher only) --------
        if teacher_logits is not None:
            teacher_logits = teacher_logits.float()
            t_ids = completion_ids.clamp(0, teacher_logits.shape[-1] - 1).unsqueeze(-1)
            teacher_logp = torch.log_softmax(teacher_logits, dim=-1)
            teacher_tok_logp = teacher_logp.gather(-1, t_ids).squeeze(-1) * mask
            _, teacher_logp_sum, _ = self._distributed_rate_stats(teacher_tok_logp)
            metrics["rollout/teacher_logprob"] = (
                teacher_logp_sum,
                completion_token_count,
            )
            del teacher_logp
            # teacher - student log p on the student's own samples: the core OPD
            # signal. >0 ⇒ the teacher would rather have drawn what the student drew
            # (it shrinks as the student converges toward the teacher).
            metrics["rollout/teacher_minus_student_logprob"] = (
                teacher_logp_sum - student_logp_sum,
                completion_token_count,
            )
            # Fraction of completion tokens where the argmax token agrees — a
            # coarse distillation-progress curve, robust to vocab-size mismatch.
            vocab = min(student_logits.shape[-1], teacher_logits.shape[-1])
            agree = (
                student_logits[..., :vocab].argmax(dim=-1)
                == teacher_logits[..., :vocab].argmax(dim=-1)
            ).to(dtype=torch.float32) * mask
            _, agree_sum, _ = self._distributed_rate_stats(agree)
            metrics["rollout/teacher_top1_agreement"] = (
                agree_sum,
                completion_token_count,
            )

        return metrics
