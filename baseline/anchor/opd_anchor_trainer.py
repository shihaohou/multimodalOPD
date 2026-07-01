"""Trainer for Evidence Anchor OPD.

This is Grounding-Hint Distillation plus a representation bottleneck loss at an
anchor marker in the prompt:

    loss = lambda_opd * L_OPD + lambda_anchor * (1 - cos(P_s h_s, sg(P_t h_t)))

The student rolls out from the plain anchored prompt. The frozen teacher scores
the same rollout under the hidden-hint anchored prompt. Both forwards also return
their last hidden states; the anchor positions are supplied by
``OPDAnchorDataCollator``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline.opd_losses import masked_topk_kl_loss
from baseline.opd_trainer import OPDTrainer


def _first_present(obj: object, names: tuple[str, ...]):
    for name in names:
        if obj is not None and hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return None


def resolve_text_hidden_size(model: nn.Module) -> int:
    """Resolve the decoder hidden size for Qwen-VL-like checkpoints."""
    cfg = getattr(model, "config", None)
    for candidate in (
        cfg,
        getattr(cfg, "text_config", None),
        getattr(cfg, "llm_config", None),
        getattr(cfg, "language_config", None),
    ):
        value = _first_present(candidate, ("hidden_size", "hidden_dim", "n_embd"))
        if value is not None:
            return int(value)
    embeddings = model.get_input_embeddings()
    if embeddings is not None and getattr(embeddings, "weight", None) is not None:
        return int(embeddings.weight.shape[-1])
    raise ValueError(f"Could not resolve hidden size for {type(model).__name__}.")


class AnchorProjectors(nn.Module):
    """Student/teacher hidden-state projectors into the shared anchor space."""

    def __init__(
        self,
        student_hidden_size: int,
        teacher_hidden_size: int,
        projection_dim: int,
        *,
        bias: bool = False,
        train_teacher_projector: bool = False,
    ) -> None:
        super().__init__()
        self.student = nn.Linear(student_hidden_size, projection_dim, bias=bias)
        self.teacher = nn.Linear(teacher_hidden_size, projection_dim, bias=bias)
        nn.init.xavier_uniform_(self.student.weight)
        nn.init.xavier_uniform_(self.teacher.weight)
        if self.student.bias is not None:
            nn.init.zeros_(self.student.bias)
        if self.teacher.bias is not None:
            nn.init.zeros_(self.teacher.bias)
        if not train_teacher_projector:
            self.teacher.requires_grad_(False)


def ensure_anchor_projectors(
    model: nn.Module,
    teacher_model: nn.Module,
    *,
    projection_dim: int,
    bias: bool = False,
    train_teacher_projector: bool = False,
) -> AnchorProjectors:
    """Attach projectors to the trainable model so HF/DeepSpeed optimizes P_s."""
    def mark_training_auxiliary(module: nn.Module) -> None:
        ignore = list(getattr(module, "_keys_to_ignore_on_save", None) or [])
        pattern = r"opd_anchor_projectors\..*"
        if pattern not in ignore:
            ignore.append(pattern)
        module._keys_to_ignore_on_save = ignore

    existing = getattr(model, "opd_anchor_projectors", None)
    if existing is not None:
        mark_training_auxiliary(model)
        return existing
    s_dim = resolve_text_hidden_size(model)
    t_dim = resolve_text_hidden_size(teacher_model)
    dim = int(projection_dim) if int(projection_dim) > 0 else min(s_dim, t_dim)
    projectors = AnchorProjectors(
        s_dim,
        t_dim,
        dim,
        bias=bias,
        train_teacher_projector=train_teacher_projector,
    )
    # Keep the freshly attached module in the same dtype as the loaded student.
    try:
        dtype = next(model.parameters()).dtype
        projectors.to(dtype=dtype)
    except StopIteration:
        pass
    model.add_module("opd_anchor_projectors", projectors)
    # Projectors are training-only auxiliaries. Keep them in named_parameters()
    # for the optimizer, but keep inference checkpoints/vLLM eval clean.
    mark_training_auxiliary(model)
    return projectors


class OPDAnchorTrainer(OPDTrainer):
    def __init__(
        self,
        *args: Any,
        lambda_anchor: float = 1.0,
        anchor_projection_dim: int = 1024,
        anchor_projector_bias: bool = False,
        anchor_train_teacher_projector: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.teacher_source != "local_hf":
            raise ValueError(
                "OPDAnchorTrainer requires teacher_source='local_hf': the anchor "
                "loss needs the teacher hidden states from a local forward."
            )
        self.lambda_anchor = float(lambda_anchor)
        self.anchor_projection_dim = int(anchor_projection_dim)
        self.anchor_projector_bias = bool(anchor_projector_bias)
        self.anchor_train_teacher_projector = bool(anchor_train_teacher_projector)
        unwrapped = self.accelerator.unwrap_model(self.model)
        projectors = ensure_anchor_projectors(
            unwrapped,
            self.teacher_model,
            projection_dim=self.anchor_projection_dim,
            bias=self.anchor_projector_bias,
            train_teacher_projector=self.anchor_train_teacher_projector,
        )
        if self.accelerator.is_main_process:
            print(
                "[OPD-anchor] projectors: "
                f"student {projectors.student.in_features}->{projectors.student.out_features}, "
                f"teacher {projectors.teacher.in_features}->{projectors.teacher.out_features}, "
                f"lambda_anchor={self.lambda_anchor}",
                flush=True,
            )

    @staticmethod
    def _is_anchor_projector_weight(name: str) -> bool:
        return name.startswith("opd_anchor_projectors.") or (
            ".opd_anchor_projectors." in name
        )

    @staticmethod
    def _map_weight_name_for_vllm(name: str, model: nn.Module) -> str | None:
        if OPDAnchorTrainer._is_anchor_projector_weight(name):
            return None
        return OPDTrainer._map_weight_name_for_vllm(name, model)

    def _load_named_weight_into_vllm(self, name: str, weight: torch.Tensor) -> None:
        if self._is_anchor_projector_weight(name):
            return
        return super()._load_named_weight_into_vllm(name, weight)

    @staticmethod
    def _gather_positions(hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        positions = positions.clamp(min=0, max=max(0, hidden.shape[1] - 1))
        index = positions.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
        return torch.gather(hidden, dim=1, index=index)

    @staticmethod
    def _distributed_sum_count(
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[float, float]:
        mask_f = mask.detach().to(device=values.device, dtype=torch.float32)
        local = torch.stack(
            [
                (values.detach().float() * mask_f).sum(),
                mask_f.sum(),
            ]
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(local, op=torch.distributed.ReduceOp.SUM)
        return float(local[0].detach().cpu()), float(local[1].detach().cpu())

    def _anchor_alignment_loss(
        self,
        model: nn.Module,
        student_hidden_last: torch.Tensor,
        teacher_hidden_last: torch.Tensor,
        inputs: dict[str, Any],
    ) -> tuple[torch.Tensor | None, dict[str, tuple[float, float]]]:
        s_pos = inputs["student_anchor_positions"].to(student_hidden_last.device)
        t_pos = inputs["teacher_anchor_positions"].to(teacher_hidden_last.device)
        s_mask = inputs["student_anchor_attention_mask"].to(
            device=student_hidden_last.device, dtype=torch.bool
        )
        t_mask = inputs["teacher_anchor_attention_mask"].to(
            device=teacher_hidden_last.device, dtype=torch.bool
        )
        max_anchors = min(s_pos.shape[1], t_pos.shape[1])
        s_pos = s_pos[:, :max_anchors]
        t_pos = t_pos[:, :max_anchors]
        mask = s_mask[:, :max_anchors] & t_mask[:, :max_anchors]
        valid_count, total_count = self._distributed_sum_count(
            mask.to(dtype=torch.float32), mask.new_ones(mask.shape)
        )
        if valid_count <= 0:
            return None, {"anchor_coverage": (0.0, total_count)}

        s_anchor = self._gather_positions(student_hidden_last, s_pos)
        t_anchor = self._gather_positions(teacher_hidden_last, t_pos)
        projectors = getattr(
            self.accelerator.unwrap_model(model), "opd_anchor_projectors"
        )
        s_weight = projectors.student.weight
        t_weight = projectors.teacher.weight
        s_z = projectors.student(s_anchor.to(dtype=s_weight.dtype))
        if self.anchor_train_teacher_projector:
            # Ablation only: teacher hidden states are already no-grad, but P_t can
            # learn. Default keeps the paper-style sg(P_t h_T) fixed target.
            t_z = projectors.teacher(t_anchor.to(dtype=t_weight.dtype))
        else:
            with torch.no_grad():
                t_z = projectors.teacher(t_anchor.to(dtype=t_weight.dtype)).detach()

        cos = F.cosine_similarity(s_z.float(), t_z.float(), dim=-1)
        per_anchor_loss = 1.0 - cos
        local_count = mask.to(dtype=torch.float32).sum().clamp_min(1.0)
        local_loss = (
            per_anchor_loss * mask.to(dtype=per_anchor_loss.dtype)
        ).sum() / local_count
        anchor_loss, _, anchor_num, anchor_count = (
            self._distributed_masked_loss_with_stats(local_loss, mask)
        )
        cos_num, cos_count = self._distributed_sum_count(cos, mask)
        stats = {
            "loss_anchor": (anchor_num, anchor_count),
            "anchor_cos": (cos_num, cos_count),
            "anchor_coverage": (valid_count, total_count),
        }
        return anchor_loss, stats

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        student_prompt = self._prompt_inputs(inputs, "student")
        teacher_prompt = self._prompt_inputs(inputs, "teacher")

        rollout = self._generate_on_policy(model, student_prompt, inputs)
        self._maybe_log_completion_snapshot(inputs, rollout)

        completion_ids = rollout["completion_ids"]
        completion_attention = rollout["completion_attention_mask"].to(dtype=torch.bool)
        completion_length = completion_ids.shape[1]

        # Student forward: logits for OPD plus last hidden state for the anchor.
        student_inputs = self._with_completion(
            student_prompt,
            full_input_ids=rollout["generated_ids"],
            full_attention_mask=rollout["generated_attention_mask"],
        )
        student_inputs["logits_to_keep"] = completion_length + 1
        student_outputs = model(**student_inputs, output_hidden_states=True)
        student_hidden_last = student_outputs.hidden_states[-1]
        student_logits = self._completion_logits(student_outputs.logits, completion_length)
        del student_outputs

        # Teacher forward: privileged hidden-hint prompt, same sampled completion.
        teacher_inputs = self._append_completion(
            teacher_prompt, completion_ids, rollout["completion_attention_mask"]
        )
        teacher_inputs["logits_to_keep"] = completion_length + 1
        with (
            torch.no_grad(),
            self._temporary_eval_context(self.teacher_model),
            self._teacher_context(self.teacher_model),
        ):
            teacher_outputs = self.teacher_model(
                **teacher_inputs, output_hidden_states=True, use_cache=False
            )
        teacher_hidden_last = teacher_outputs.hidden_states[-1]
        teacher_logits = self._completion_logits(teacher_outputs.logits, completion_length)
        del teacher_outputs

        # OPD token loss.
        vocab = min(student_logits.shape[-1], teacher_logits.shape[-1])
        student_kl_logits = student_logits[..., :vocab].float()
        teacher_kl_logits = teacher_logits[..., :vocab].float()
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

        anchor_stats: dict[str, tuple[float, float]] = {}
        if self.lambda_anchor > 0:
            anchor_loss, anchor_stats = self._anchor_alignment_loss(
                model, student_hidden_last, teacher_hidden_last, inputs
            )
            if anchor_loss is not None and bool(torch.isfinite(anchor_loss.detach())):
                loss = loss + self.lambda_anchor * anchor_loss

        # Metrics.
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
        metrics.update(anchor_stats)
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
