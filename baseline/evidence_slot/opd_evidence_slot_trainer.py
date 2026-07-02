"""Trainer for Visual-Value Evidence Slot OPD.

The evidence-slot loss is a structured replacement for raw anchor hidden-state
alignment. The student still rolls out from a plain prompt, and the frozen teacher
scores the same rollout under a hidden evidence-box hint. The auxiliary target is
not a raw teacher hidden state:

    Q = hidden(<EVID> slots)
    K,V = hidden(image placeholder tokens)
    Z = softmax(Q K^T / sqrt(d) + bbox_bias) V

Only the value path comes from visual tokens. The hint/bbox can steer the teacher
query or attention bias, but the latent that gets aligned is pooled out of image
token states. On the student side, a detached prompt prepass can also inject the
pooled ``Z_S`` back into the prompt's ``<EVID>`` input embeddings for the main OPD
scoring forward, so completion tokens train while attending to the evidence slot.
"""

from __future__ import annotations

from contextlib import contextmanager
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline.anchor.opd_anchor_trainer import (
    OPDAnchorTrainer,
    ensure_anchor_projectors,
    strip_anchor_auxiliary_weights,
)
from baseline.opd_losses import masked_topk_kl_loss
from baseline.opd_trainer import OPDTrainer
from baseline.tam.tam_engine import resolve_tam_parts


class OPDEvidenceSlotTrainer(OPDTrainer):
    """OPD + question-conditioned visual-value evidence-slot alignment."""

    def __init__(
        self,
        *args: Any,
        lambda_evidence_slot: float = 1.0,
        evidence_slot_projection_dim: int = 1024,
        evidence_slot_projector_bias: bool = False,
        evidence_slot_train_teacher_projector: bool = False,
        evidence_slot_bbox_bias_beta: float = 2.0,
        evidence_slot_normalize_qk: bool = True,
        evidence_slot_inject_student: bool = True,
        evidence_slot_injection_scale: float = 1.0,
        evidence_slot_injection_match_norm: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.teacher_source != "local_hf":
            raise ValueError(
                "OPDEvidenceSlotTrainer requires teacher_source='local_hf': the "
                "evidence-slot loss needs teacher hidden states from a local forward."
            )
        self.lambda_evidence_slot = float(lambda_evidence_slot)
        self.evidence_slot_projection_dim = int(evidence_slot_projection_dim)
        self.evidence_slot_projector_bias = bool(evidence_slot_projector_bias)
        self.evidence_slot_train_teacher_projector = bool(
            evidence_slot_train_teacher_projector
        )
        self.evidence_slot_bbox_bias_beta = float(evidence_slot_bbox_bias_beta)
        self.evidence_slot_normalize_qk = bool(evidence_slot_normalize_qk)
        self.evidence_slot_inject_student = bool(evidence_slot_inject_student)
        self.evidence_slot_injection_scale = float(evidence_slot_injection_scale)
        self.evidence_slot_injection_match_norm = bool(
            evidence_slot_injection_match_norm
        )
        self._student_parts = None
        self._teacher_parts = None
        self._grid_checked = False
        self._warned_injection_hook = False

        unwrapped = self.accelerator.unwrap_model(self.model)
        projectors = ensure_anchor_projectors(
            unwrapped,
            self.teacher_model,
            projection_dim=self.evidence_slot_projection_dim,
            bias=self.evidence_slot_projector_bias,
            train_teacher_projector=self.evidence_slot_train_teacher_projector,
        )
        if self.accelerator.is_main_process:
            print(
                "[OPD-evidence-slot] projectors: "
                f"student {projectors.student.in_features}->{projectors.student.out_features}, "
                f"teacher {projectors.teacher.in_features}->{projectors.teacher.out_features}, "
                f"lambda={self.lambda_evidence_slot}, "
                f"bbox_beta={self.evidence_slot_bbox_bias_beta}, "
                f"normalize_qk={self.evidence_slot_normalize_qk}, "
                f"inject_student={self.evidence_slot_inject_student}, "
                f"injection_scale={self.evidence_slot_injection_scale}, "
                f"match_norm={self.evidence_slot_injection_match_norm}",
                flush=True,
            )

    @staticmethod
    def _is_evidence_slot_aux_weight(name: str) -> bool:
        return OPDAnchorTrainer._is_anchor_projector_weight(name)

    @staticmethod
    def _map_weight_name_for_vllm(name: str, model: nn.Module) -> str | None:
        if OPDEvidenceSlotTrainer._is_evidence_slot_aux_weight(name):
            return None
        return OPDTrainer._map_weight_name_for_vllm(name, model)

    def _load_named_weight_into_vllm(self, name: str, weight: torch.Tensor) -> None:
        if self._is_evidence_slot_aux_weight(name):
            return
        return super()._load_named_weight_into_vllm(name, weight)

    def save_model(
        self,
        output_dir: str | None = None,
        _internal_call: bool = False,
    ) -> None:
        super().save_model(output_dir=output_dir, _internal_call=_internal_call)
        out = output_dir or self.args.output_dir
        if self.accelerator.is_main_process:
            removed = strip_anchor_auxiliary_weights(out)
            if removed:
                print(
                    "[OPD-evidence-slot] stripped training-only checkpoint weights: "
                    + ", ".join(removed),
                    flush=True,
                )
        self.accelerator.wait_for_everyone()

    def _ensure_parts(self, model: nn.Module) -> None:
        if self._student_parts is None:
            self._student_parts = resolve_tam_parts(self.accelerator.unwrap_model(model))
        if self._teacher_parts is None:
            self._teacher_parts = resolve_tam_parts(self.teacher_model)
        if self._grid_checked:
            return
        self._grid_checked = True
        if self._student_parts.image_token_id != self._teacher_parts.image_token_id:
            raise ValueError(
                "Teacher/student image token ids differ "
                f"({self._student_parts.image_token_id} vs "
                f"{self._teacher_parts.image_token_id}); evidence-slot visual "
                "positions would be incomparable."
            )
        if self.accelerator.is_main_process:
            print(
                "[OPD-evidence-slot] visual token config OK: "
                f"image_token_id={self._student_parts.image_token_id}, "
                f"student_merge={self._student_parts.spatial_merge_size}, "
                f"teacher_merge={self._teacher_parts.spatial_merge_size}.",
                flush=True,
            )

    @staticmethod
    def _nested_module(root: nn.Module, path: str) -> nn.Module | None:
        obj: object = root
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                return None
        return obj if isinstance(obj, nn.Module) else None

    def _language_model_module(self, model: nn.Module) -> nn.Module | None:
        root = self.accelerator.unwrap_model(model)
        for path in (
            "model.language_model",
            "language_model",
            "base_model.model.model.language_model",
            "base_model.model.language_model",
        ):
            module = self._nested_module(root, path)
            if module is not None:
                return module
        return None

    @staticmethod
    def _gather_positions(hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        positions = positions.clamp(min=0, max=max(0, hidden.shape[1] - 1))
        index = positions.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
        return torch.gather(hidden, dim=1, index=index)

    @staticmethod
    def _visual_positions(input_ids: torch.Tensor, image_token_id: int) -> torch.Tensor:
        return (input_ids == int(image_token_id)).nonzero(as_tuple=True)[0]

    @staticmethod
    def _bbox_inside_mask(
        *,
        grid: torch.Tensor,
        n_visual_tokens: int,
        spatial_merge_size: int,
        bbox: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor | None:
        if n_visual_tokens <= 0:
            return None
        merge = max(1, int(spatial_merge_size))
        t_dim = int(grid[0].item())
        h_grid = int(grid[1].item()) // merge
        w_grid = int(grid[2].item()) // merge
        if t_dim <= 0 or h_grid <= 0 or w_grid <= 0:
            return None
        if t_dim * h_grid * w_grid != int(n_visual_tokens):
            return None

        x1, y1, x2, y2 = [float(v) for v in bbox.detach().cpu().tolist()]
        xs = (torch.arange(w_grid, device=device, dtype=torch.float32) + 0.5) / float(
            w_grid
        )
        ys = (torch.arange(h_grid, device=device, dtype=torch.float32) + 0.5) / float(
            h_grid
        )
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        inside_hw = (xx >= x1) & (xx <= x2) & (yy >= y1) & (yy <= y2)
        inside = inside_hw.reshape(1, -1).expand(t_dim, -1).reshape(-1)
        return inside[:n_visual_tokens]

    def _pool_visual_values(
        self,
        *,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        query_positions: torch.Tensor,
        query_mask: torch.Tensor,
        image_token_id: int,
        image_grid_thw: torch.Tensor | None,
        spatial_merge_size: int,
        bbox: torch.Tensor | None = None,
        bbox_mask: torch.Tensor | None = None,
        bbox_bias_beta: float = 0.0,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Return ``(Z, slot_mask, bbox_mass)`` for one model side.

        ``Z`` is the visual-value latent for each valid evidence slot. ``bbox_mass``
        is the slot attention mass inside the GT box when a bbox/grid is available.
        """
        valid_slots = query_mask.to(device=hidden.device, dtype=torch.bool)
        if not bool(valid_slots.any()):
            return None, None, None
        visual_positions = self._visual_positions(input_ids, image_token_id)
        if visual_positions.numel() == 0:
            return None, None, None

        q = self._gather_positions(
            hidden.unsqueeze(0), query_positions.to(hidden.device).unsqueeze(0)
        ).squeeze(0)
        q = q[valid_slots]
        visual = hidden.index_select(0, visual_positions.to(hidden.device))
        qf = q.float()
        kf = visual.float()
        if self.evidence_slot_normalize_qk:
            qf = F.normalize(qf, dim=-1)
            kf = F.normalize(kf, dim=-1)
            scale = 1.0
        else:
            scale = 1.0 / math.sqrt(max(1, int(qf.shape[-1])))
        scores = torch.matmul(qf, kf.transpose(0, 1)) * scale

        inside = None
        if (
            bbox is not None
            and bbox_mask is not None
            and bool(bbox_mask.item())
            and image_grid_thw is not None
        ):
            inside = self._bbox_inside_mask(
                grid=image_grid_thw,
                n_visual_tokens=int(visual_positions.numel()),
                spatial_merge_size=spatial_merge_size,
                bbox=bbox,
                device=hidden.device,
            )
            if inside is not None and bool(inside.any()) and bbox_bias_beta != 0.0:
                scores = scores + inside.to(dtype=scores.dtype).unsqueeze(0) * float(
                    bbox_bias_beta
                )

        attention = scores.softmax(dim=-1).to(dtype=visual.dtype)
        z = torch.matmul(attention, visual)
        bbox_mass = None
        if inside is not None:
            bbox_mass = attention.float().matmul(inside.to(dtype=torch.float32))
        return z, valid_slots, bbox_mass

    @torch.no_grad()
    def _student_prompt_carrier_deltas(
        self,
        model: nn.Module,
        student_prompt: dict[str, torch.Tensor],
        inputs: dict[str, Any],
    ) -> torch.Tensor | None:
        """Build detached ``Z_S`` deltas for prompt <EVID> slots.

        This is a no-grad prompt prepass to avoid two train-graph forwards through
        the same DeepSpeed/ZeRO-wrapped student in one backward. The main full
        forward still carries gradients; the evidence-slot side loss trains the
        query/value path, while OPD sees the injected carrier in the scoring pass.
        """
        if (
            not self.evidence_slot_inject_student
            or self.evidence_slot_injection_scale == 0.0
        ):
            return None
        self._ensure_parts(model)
        s_parts = self._student_parts
        assert s_parts is not None

        prompt_inputs = dict(student_prompt)
        prompt_inputs["logits_to_keep"] = 1
        with self._temporary_eval_context(model):
            outputs = model(
                **prompt_inputs,
                output_hidden_states=True,
                use_cache=False,
            )
        hidden_last = outputs.hidden_states[-1]
        del outputs

        s_pos = inputs["student_anchor_positions"].to(hidden_last.device)
        s_mask = inputs["student_anchor_attention_mask"].to(
            device=hidden_last.device, dtype=torch.bool
        )
        image_grid_thw = student_prompt.get("image_grid_thw")
        deltas = hidden_last.new_zeros(hidden_last.shape)
        for b in range(hidden_last.shape[0]):
            slot_mask = s_mask[b]
            if not bool(slot_mask.any()):
                continue
            z, _, _ = self._pool_visual_values(
                hidden=hidden_last[b],
                input_ids=student_prompt["input_ids"][b],
                query_positions=s_pos[b],
                query_mask=slot_mask,
                image_token_id=s_parts.image_token_id,
                image_grid_thw=(
                    image_grid_thw[b] if isinstance(image_grid_thw, torch.Tensor) else None
                ),
                spatial_merge_size=s_parts.spatial_merge_size,
                bbox=None,
                bbox_mask=None,
                bbox_bias_beta=0.0,
            )
            if z is None:
                continue
            positions = s_pos[b][slot_mask].clamp(
                min=0, max=max(0, hidden_last.shape[1] - 1)
            )
            deltas[b, positions] = z.to(dtype=deltas.dtype)
        return deltas.detach()

    @contextmanager
    def _student_evidence_slot_injection(
        self,
        model: nn.Module,
        prompt_deltas: torch.Tensor | None,
    ):
        if prompt_deltas is None:
            yield
            return
        language_model = self._language_model_module(model)
        if language_model is None:
            if not self._warned_injection_hook and self.accelerator.is_main_process:
                print(
                    "[OPD-evidence-slot] warning: could not locate inner language "
                    "model; student carrier injection is disabled for this step.",
                    flush=True,
                )
                self._warned_injection_hook = True
            yield
            return

        def hook(module, args, kwargs):
            embeds = kwargs.get("inputs_embeds")
            if embeds is None:
                if not self._warned_injection_hook and self.accelerator.is_main_process:
                    print(
                        "[OPD-evidence-slot] warning: language-model hook did not "
                        "receive inputs_embeds; carrier injection skipped.",
                        flush=True,
                    )
                    self._warned_injection_hook = True
                return args, kwargs
            prompt_len = int(prompt_deltas.shape[1])
            if (
                embeds.shape[0] != prompt_deltas.shape[0]
                or embeds.shape[1] < prompt_len
                or embeds.shape[-1] != prompt_deltas.shape[-1]
            ):
                return args, kwargs

            delta = prompt_deltas.to(device=embeds.device, dtype=embeds.dtype)
            if self.evidence_slot_injection_match_norm:
                base = embeds[:, :prompt_len, :]
                delta_norm = delta.norm(dim=-1, keepdim=True)
                base_norm = base.detach().norm(dim=-1, keepdim=True)
                delta = torch.where(
                    delta_norm > 0,
                    delta * (base_norm / delta_norm.clamp_min(1e-6)),
                    delta,
                )

            updated = embeds.clone()
            updated[:, :prompt_len, :] = (
                updated[:, :prompt_len, :]
                + float(self.evidence_slot_injection_scale) * delta
            )
            kwargs["inputs_embeds"] = updated
            return args, kwargs

        handle = language_model.register_forward_pre_hook(hook, with_kwargs=True)
        try:
            yield
        finally:
            handle.remove()

    def _distributed_vector_objective(
        self,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, float, float]:
        local_count = values.new_tensor(float(values.numel()), dtype=torch.float32)
        local_numerator = values.float().sum()
        global_count = local_count.detach().clone()
        global_numerator = local_numerator.detach().clone()
        world_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                global_count, op=torch.distributed.ReduceOp.SUM
            )
            torch.distributed.all_reduce(
                global_numerator, op=torch.distributed.ReduceOp.SUM
            )
            world_size = torch.distributed.get_world_size()

        denominator = global_count.clamp_min(1.0)
        gradient_loss = local_numerator * (float(world_size) / denominator)
        logged_loss = global_numerator / denominator
        value_loss = logged_loss.to(device=values.device, dtype=torch.float32)
        objective = gradient_loss + (value_loss - gradient_loss.detach())
        return (
            objective,
            float(global_numerator.detach().cpu()),
            float(global_count.detach().cpu()),
        )

    def _distributed_sum_count(
        self,
        values: torch.Tensor,
    ) -> tuple[float, float]:
        local = torch.stack(
            [
                values.detach().float().sum(),
                values.new_tensor(float(values.numel()), dtype=torch.float32),
            ]
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(local, op=torch.distributed.ReduceOp.SUM)
        return float(local[0].detach().cpu()), float(local[1].detach().cpu())

    def _evidence_slot_loss(
        self,
        model: nn.Module,
        *,
        student_hidden_last: torch.Tensor,
        teacher_hidden_last: torch.Tensor,
        student_full_ids: torch.Tensor,
        teacher_full_ids: torch.Tensor,
        student_prompt: dict[str, torch.Tensor],
        teacher_prompt: dict[str, torch.Tensor],
        inputs: dict[str, Any],
    ) -> tuple[torch.Tensor | None, dict[str, tuple[float, float]]]:
        self._ensure_parts(model)
        s_parts = self._student_parts
        t_parts = self._teacher_parts
        assert s_parts is not None and t_parts is not None

        s_pos = inputs["student_anchor_positions"].to(student_hidden_last.device)
        t_pos = inputs["teacher_anchor_positions"].to(teacher_hidden_last.device)
        s_mask = inputs["student_anchor_attention_mask"].to(
            device=student_hidden_last.device, dtype=torch.bool
        )
        t_mask = inputs["teacher_anchor_attention_mask"].to(
            device=teacher_hidden_last.device, dtype=torch.bool
        )
        max_slots = min(s_pos.shape[1], t_pos.shape[1])
        s_pos = s_pos[:, :max_slots]
        t_pos = t_pos[:, :max_slots]
        s_mask = s_mask[:, :max_slots]
        t_mask = t_mask[:, :max_slots]
        bbox_norm = inputs.get("bbox_norm")
        bbox_attention = inputs.get("bbox_attention_mask")

        projectors = getattr(
            self.accelerator.unwrap_model(model), "opd_anchor_projectors"
        )
        s_weight = projectors.student.weight
        t_weight = projectors.teacher.weight
        loss_terms: list[torch.Tensor] = []
        cos_terms: list[torch.Tensor] = []
        s_bbox_terms: list[torch.Tensor] = []
        t_bbox_terms: list[torch.Tensor] = []

        s_grid = student_prompt.get("image_grid_thw")
        t_grid = teacher_prompt.get("image_grid_thw")
        for b in range(student_hidden_last.shape[0]):
            slot_mask = s_mask[b] & t_mask[b]
            if not bool(slot_mask.any()):
                continue

            bbox_b = (
                bbox_norm[b].to(teacher_hidden_last.device)
                if isinstance(bbox_norm, torch.Tensor)
                else None
            )
            bbox_mask_b = (
                bbox_attention[b].to(teacher_hidden_last.device)
                if isinstance(bbox_attention, torch.Tensor)
                else None
            )

            s_z, _, s_bbox_mass = self._pool_visual_values(
                hidden=student_hidden_last[b],
                input_ids=student_full_ids[b],
                query_positions=s_pos[b],
                query_mask=slot_mask,
                image_token_id=s_parts.image_token_id,
                image_grid_thw=s_grid[b] if isinstance(s_grid, torch.Tensor) else None,
                spatial_merge_size=s_parts.spatial_merge_size,
                bbox=bbox_b.to(student_hidden_last.device) if bbox_b is not None else None,
                bbox_mask=(
                    bbox_mask_b.to(student_hidden_last.device)
                    if bbox_mask_b is not None
                    else None
                ),
                bbox_bias_beta=0.0,
            )
            t_z, _, t_bbox_mass = self._pool_visual_values(
                hidden=teacher_hidden_last[b],
                input_ids=teacher_full_ids[b],
                query_positions=t_pos[b],
                query_mask=slot_mask,
                image_token_id=t_parts.image_token_id,
                image_grid_thw=t_grid[b] if isinstance(t_grid, torch.Tensor) else None,
                spatial_merge_size=t_parts.spatial_merge_size,
                bbox=bbox_b,
                bbox_mask=bbox_mask_b,
                bbox_bias_beta=self.evidence_slot_bbox_bias_beta,
            )
            if s_z is None or t_z is None:
                continue

            s_proj = projectors.student(s_z.to(dtype=s_weight.dtype))
            if self.evidence_slot_train_teacher_projector:
                t_proj = projectors.teacher(t_z.to(dtype=t_weight.dtype))
            else:
                with torch.no_grad():
                    t_proj = projectors.teacher(t_z.to(dtype=t_weight.dtype)).detach()

            cos = F.cosine_similarity(s_proj.float(), t_proj.float(), dim=-1)
            loss_terms.append(1.0 - cos)
            cos_terms.append(cos.detach())
            if s_bbox_mass is not None:
                s_bbox_terms.append(s_bbox_mass.detach())
            if t_bbox_mass is not None:
                t_bbox_terms.append(t_bbox_mass.detach())

        device = student_hidden_last.device
        losses = (
            torch.cat([term.reshape(-1) for term in loss_terms], dim=0)
            if loss_terms
            else torch.empty(0, device=device)
        )
        objective, loss_num, loss_count = self._distributed_vector_objective(losses)
        coverage_total_tensor = s_mask.detach().new_tensor(
            float(s_mask.numel()), dtype=torch.float32
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                coverage_total_tensor, op=torch.distributed.ReduceOp.SUM
            )
        coverage_total = float(coverage_total_tensor.detach().cpu())
        if loss_count <= 0:
            return None, {"evidence_slot_coverage": (0.0, coverage_total)}

        cos_values = (
            torch.cat([term.reshape(-1) for term in cos_terms], dim=0)
            if cos_terms
            else torch.empty(0, device=device)
        )
        cos_num, cos_count = self._distributed_sum_count(cos_values)

        stats: dict[str, tuple[float, float]] = {
            "loss_evidence_slot": (loss_num, loss_count),
            "evidence_slot_cos": (cos_num, cos_count),
            "evidence_slot_coverage": (loss_count, coverage_total),
        }
        vals = (
            torch.cat([term.reshape(-1) for term in s_bbox_terms], dim=0)
            if s_bbox_terms
            else torch.empty(0, device=device)
        )
        num, count = self._distributed_sum_count(vals)
        if count > 0:
            stats["evidence_slot/student_bbox_mass"] = (num, count)
        vals = (
            torch.cat([term.reshape(-1) for term in t_bbox_terms], dim=0)
            if t_bbox_terms
            else torch.empty(0, device=device)
        )
        num, count = self._distributed_sum_count(vals)
        if count > 0:
            stats["evidence_slot/teacher_bbox_mass"] = (num, count)
        return objective, stats

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

        prompt_carrier_deltas = self._student_prompt_carrier_deltas(
            model, student_prompt, inputs
        )
        student_inputs = self._with_completion(
            student_prompt,
            full_input_ids=rollout["generated_ids"],
            full_attention_mask=rollout["generated_attention_mask"],
        )
        student_inputs["logits_to_keep"] = completion_length + 1
        with self._student_evidence_slot_injection(model, prompt_carrier_deltas):
            student_outputs = model(**student_inputs, output_hidden_states=True)
        student_hidden_last = student_outputs.hidden_states[-1]
        student_logits = self._completion_logits(student_outputs.logits, completion_length)
        del student_outputs

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

        evidence_stats: dict[str, tuple[float, float]] = {}
        if self.lambda_evidence_slot > 0:
            evidence_loss, evidence_stats = self._evidence_slot_loss(
                model,
                student_hidden_last=student_hidden_last,
                teacher_hidden_last=teacher_hidden_last,
                student_full_ids=rollout["generated_ids"],
                teacher_full_ids=teacher_inputs["input_ids"],
                student_prompt=student_prompt,
                teacher_prompt=teacher_prompt,
                inputs=inputs,
            )
            if evidence_loss is not None and bool(torch.isfinite(evidence_loss.detach())):
                loss = loss + self.lambda_evidence_slot * evidence_loss

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
        metrics.update(evidence_stats)
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
