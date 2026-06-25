"""OPD trainer with the differentiable TAM visual-evidence alignment loss.

``TAMTrainer`` adds, on top of the vanilla OPD token-distillation loss, a Token
Activation Map alignment term:

    loss = lambda_opd * L_opd  +  lambda_tam * L_tam

``L_opd`` is exactly :class:`baseline.opd_trainer.OPDTrainer`'s reverse-KL token
loss (the *behavior* channel — *what* token to generate). ``L_tam`` pulls the
student's per-token TAM map toward the frozen teacher's, on the visual-dependent
tokens, via the gated divergence in :mod:`baseline.tam.tam_losses` (the *visual
evidence* channel — *where* in the image the token draws support). See
``baseline/tam/README.md``.

**One student forward.** The TAM logit-lens needs the last-layer hidden states
(``output_hidden_states=True``) but — unlike the saliency engine — **no attention
weights**, so the student runs under its normal SDPA/Flash attention with no eager
switch and no hooks. Crucially it is the SAME forward that produces the OPD
logits: a second grad forward through the (DeepSpeed-wrapped) student would make
ZeRO reduce each shared parameter's gradient twice in one backward ("parameter
... has already been reduced"). So the OPD logits and the TAM hidden states both
come from a single ``model(..., output_hidden_states=True)`` call.

The teacher runs one no-grad forward producing both its OPD logits and its TAM
hidden states. ``local_hf`` teacher only (the vLLM-server path returns top-k
logprobs, not hidden states). Steps whose rollouts have no valid visual tokens
fall back to a cheap OPD-only loss.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from baseline.opd_losses import masked_topk_kl_loss
from baseline.opd_trainer import OPDTrainer
from baseline.tam.tam_engine import compute_tam_token_maps, resolve_tam_parts
from baseline.tam.tam_losses import (
    concentration_gate,
    gaussian_blur_maps,
    tam_alignment_loss,
)


class TAMTrainer(OPDTrainer):
    def __init__(
        self,
        *args: Any,
        lambda_tam: float = 1.0,
        tam_align_span: str = "completion",
        tam_use_eci: bool = True,
        tam_detach_lm_head: bool = True,
        tam_divergence: str = "cosine",
        tam_blur: bool = True,
        tam_blur_kernel: int = 3,
        tam_blur_sigma: float = 1.0,
        tam_gate_temp: float = 1.0,
        tam_gate_h0: float = 0.9,
        tam_gate_tau: float = 0.1,
        tam_mass_threshold: float = 0.0,
        tam_max_tokens: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.teacher_source != "local_hf":
            raise ValueError(
                "TAMTrainer requires teacher_source='local_hf' (the TAM term needs "
                "the teacher's last-layer hidden states from a local forward)."
            )
        if tam_align_span not in {"completion", "answer", "reason_answer"}:
            raise ValueError(
                f"Unknown tam_align_span {tam_align_span!r}; use 'completion', "
                "'answer', or 'reason_answer'."
            )
        if tam_divergence not in {"cosine", "js", "l1", "mse"}:
            raise ValueError(
                f"Unknown tam_divergence {tam_divergence!r}; use 'cosine', 'js', 'l1', or 'mse'."
            )
        self.lambda_tam = float(lambda_tam)
        self.tam_align_span = tam_align_span
        self.tam_use_eci = bool(tam_use_eci)
        self.tam_detach_lm_head = bool(tam_detach_lm_head)
        self.tam_divergence = tam_divergence
        self.tam_blur = bool(tam_blur)
        self.tam_blur_kernel = int(tam_blur_kernel)
        self.tam_blur_sigma = float(tam_blur_sigma)
        self.tam_gate_temp = float(tam_gate_temp)
        self.tam_gate_h0 = float(tam_gate_h0)
        self.tam_gate_tau = float(tam_gate_tau)
        self.tam_mass_threshold = float(tam_mass_threshold)
        self.tam_max_tokens = int(tam_max_tokens)
        self._student_parts = None
        self._teacher_parts = None
        self._grid_checked = False
        self._span_parser = None

    def _candidate_completion_indices(
        self, completion_ids: torch.Tensor, completion_attention: torch.Tensor, b: int
    ) -> torch.Tensor:
        """Indices *within the completion* (0-based) of the tokens to align on.

        Default ``completion`` = all non-pad rollout tokens (the concentration gate
        then picks the visual-dependent ones — migration doc §2). ``answer`` /
        ``reason_answer`` restrict to the parsed ``\\boxed{}`` (and ``<reason>``)
        spans via the shared OPD span parser (lazy-imported so the default path
        carries no dependency on the evidence package)."""
        valid = completion_attention[b].nonzero(as_tuple=True)[0]
        if self.tam_align_span == "completion":
            return valid
        if self._span_parser is None:
            from baseline.evidence.span_utils import parse_completion_spans

            self._span_parser = parse_completion_spans
        ids_row = completion_ids[b][completion_attention[b]].tolist()
        spans = self._span_parser(
            getattr(self.processor, "tokenizer", self.processor), ids_row
        )
        # Span indices are into the *compacted* (valid-only) completion; map them
        # back to full-completion positions via `valid` so they line up with
        # `prompt_length + comp_idx` regardless of where the pad tokens sit.
        keep: list[int] = []
        if spans.answer is not None:
            keep.extend(range(spans.answer[0], spans.answer[1] + 1))
        if self.tam_align_span == "reason_answer" and spans.reason is not None:
            keep.extend(range(spans.reason[0], spans.reason[1] + 1))
        keep = sorted({i for i in keep if 0 <= i < valid.numel()})
        if not keep:
            return valid.new_zeros(0)
        return valid.index_select(0, valid.new_tensor(keep))

    def _tam_loss(
        self,
        student_prompt: dict[str, torch.Tensor],
        rollout: dict[str, torch.Tensor],
        student_hidden_last: torch.Tensor,
        teacher_hidden_last: torch.Tensor,
    ) -> tuple[torch.Tensor | None, dict[str, float]]:
        """TAM alignment loss from the already-computed last-layer hidden states.

        Per sample: locate the visual tokens (shared between teacher & student —
        identical token sequence), pick the candidate completion tokens, build the
        ECI text context, compute teacher (no-grad) and student (grad) TAM maps,
        and reduce with the gated divergence."""
        s_parts, t_parts = self._student_parts, self._teacher_parts
        prompt_length = int(student_prompt["input_ids"].shape[1])
        generated_ids = rollout["generated_ids"]
        generated_attention = rollout["generated_attention_mask"].to(dtype=torch.bool)
        completion_ids = rollout["completion_ids"]
        completion_attention = rollout["completion_attention_mask"].to(dtype=torch.bool)
        image_grid_thw = student_prompt.get("image_grid_thw")
        s_weight = s_parts.lm_head.weight
        t_weight = t_parts.lm_head.weight

        loss_terms: list[torch.Tensor] = []
        div_sum = 0.0
        js_sum = 0.0
        gate_sum = 0.0
        n_total = 0
        valid_samples = 0
        for b in range(generated_ids.shape[0]):
            seq_ids = generated_ids[b]
            visual_positions = (seq_ids == s_parts.image_token_id).nonzero(as_tuple=True)[0]
            if visual_positions.numel() == 0 or image_grid_thw is None:
                continue
            grid = image_grid_thw[b]
            merge = s_parts.spatial_merge_size
            t_dim, h_grid, w_grid = int(grid[0]), int(grid[1]) // merge, int(grid[2]) // merge
            if t_dim * h_grid * w_grid != visual_positions.numel():
                continue  # grid/visual-token mismatch for this sample — skip.

            comp_idx = self._candidate_completion_indices(
                completion_ids, completion_attention, b
            )
            if comp_idx.numel() == 0:
                continue
            candidate_positions = prompt_length + comp_idx
            candidate_ids = seq_ids.index_select(0, candidate_positions)

            # ECI context = non-visual, non-pad text positions (prompt + completion).
            context_mask = generated_attention[b] & (seq_ids != s_parts.image_token_id)
            context_positions = context_mask.nonzero(as_tuple=True)[0]
            context_ids = seq_ids.index_select(0, context_positions)

            with torch.no_grad():
                teacher_maps = compute_tam_token_maps(
                    teacher_hidden_last[b],
                    t_weight,
                    visual_positions=visual_positions,
                    token_ids=candidate_ids,
                    token_positions=candidate_positions,
                    context_positions=context_positions,
                    context_ids=context_ids,
                    use_eci=self.tam_use_eci,
                    detach_lm_head=True,
                )

            # Optional hard cap: keep the most concentrated teacher tokens (the
            # doc's hard position gate). Default off — TAM is cheap, the soft gate
            # in the loss handles diffuse tokens.
            if 0 < self.tam_max_tokens < candidate_ids.shape[0]:
                grid_thw = (t_dim, h_grid, w_grid)
                gate_for_select = concentration_gate(
                    gaussian_blur_maps(
                        teacher_maps.float(),
                        grid_thw,
                        kernel_size=self.tam_blur_kernel,
                        sigma=self.tam_blur_sigma,
                    )
                    if self.tam_blur
                    else teacher_maps.float(),
                    temp=self.tam_gate_temp,
                    h0=self.tam_gate_h0,
                    tau=self.tam_gate_tau,
                )
                sel = torch.topk(gate_for_select, k=self.tam_max_tokens).indices
                candidate_ids = candidate_ids.index_select(0, sel)
                candidate_positions = candidate_positions.index_select(0, sel)
                teacher_maps = teacher_maps.index_select(0, sel)

            student_maps = compute_tam_token_maps(
                student_hidden_last[b],
                s_weight,
                visual_positions=visual_positions,
                token_ids=candidate_ids,
                token_positions=candidate_positions,
                context_positions=context_positions,
                context_ids=context_ids,
                use_eci=self.tam_use_eci,
                detach_lm_head=self.tam_detach_lm_head,
            )

            loss_b, stats_b = tam_alignment_loss(
                student_maps,
                teacher_maps,
                grid_thw=(t_dim, h_grid, w_grid),
                divergence=self.tam_divergence,
                blur=self.tam_blur,
                blur_kernel=self.tam_blur_kernel,
                blur_sigma=self.tam_blur_sigma,
                gate_temp=self.tam_gate_temp,
                gate_h0=self.tam_gate_h0,
                gate_tau=self.tam_gate_tau,
                mass_threshold=self.tam_mass_threshold,
            )
            loss_terms.append(loss_b)
            n_b = int(stats_b["tam_n"])
            div_sum += float(stats_b["tam_div"]) * n_b
            js_sum += float(stats_b["tam_js"]) * n_b
            gate_sum += float(stats_b["tam_gate_mean"]) * n_b
            n_total += n_b
            valid_samples += 1

        if not loss_terms:
            return None, {"tam_valid_samples": 0.0}
        loss = torch.stack(loss_terms).mean()
        return loss, {
            "tam_valid_samples": float(valid_samples),
            "tam_n_selected": float(n_total),
            "tam_div_sum": div_sum,
            "tam_js_sum": js_sum,
            "tam_gate_sum": gate_sum,
        }

    def _assert_shared_grid(self) -> None:
        """Teacher & student must share the merge size + image-token id, else the
        per-patch TAM maps are computed on incompatible grids (silent confound)."""
        if self._grid_checked:
            return
        self._grid_checked = True
        sp, tp = self._student_parts, self._teacher_parts
        if (
            sp.spatial_merge_size != tp.spatial_merge_size
            or sp.image_token_id != tp.image_token_id
        ):
            raise ValueError(
                "Teacher/student visual grids differ "
                f"(spatial_merge {sp.spatial_merge_size} vs {tp.spatial_merge_size}, "
                f"image_token_id {sp.image_token_id} vs {tp.image_token_id}); the "
                "per-patch TAM maps would be incomparable. Use a shared-tokenizer, "
                "same-ViT-design line (e.g. Qwen3-VL 8B->2B)."
            )
        if self.accelerator.is_main_process:
            print(
                f"[OPD-TAM] grid check OK: spatial_merge={sp.spatial_merge_size} "
                f"image_token_id={sp.image_token_id} — teacher & student share the "
                "patch grid (same pixel_values fed to both).",
                flush=True,
            )

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
        completion_length = completion_ids.shape[1]

        run_tam = self.lambda_tam > 0
        if run_tam:
            unwrapped = self.accelerator.unwrap_model(model)
            if self._student_parts is None:
                self._student_parts = resolve_tam_parts(unwrapped)
            if self._teacher_parts is None:
                self._teacher_parts = resolve_tam_parts(self.teacher_model)
            self._assert_shared_grid()

        # --- SINGLE student grad forward: OPD logits + TAM hidden states ---------
        student_inputs = self._with_completion(
            student_prompt,
            full_input_ids=rollout["generated_ids"],
            full_attention_mask=rollout["generated_attention_mask"],
        )
        student_inputs["logits_to_keep"] = completion_length + 1
        student_hidden_last = None
        if run_tam:
            student_outputs = model(**student_inputs, output_hidden_states=True)
            # Keep only the last (post-norm) hidden state; the OPD logits are a view
            # of `.logits`. Dropping `student_outputs` lets gradient checkpointing
            # free the other layers' hidden states (output_hidden_states would else
            # pin every layer's [B,S,hidden] for the whole backward).
            student_hidden_last = student_outputs.hidden_states[-1]
            student_logits = self._completion_logits(
                student_outputs.logits, completion_length
            )
            del student_outputs
        else:
            student_outputs = model(**student_inputs)
            student_logits = self._completion_logits(
                student_outputs.logits, completion_length
            )
            del student_outputs

        # --- teacher forward (frozen, no grad): OPD logits + TAM hidden states ----
        teacher_inputs = self._append_completion(
            student_prompt, completion_ids, rollout["completion_attention_mask"]
        )
        teacher_hidden_last = None
        if run_tam:
            teacher_inputs["logits_to_keep"] = completion_length + 1
            with torch.no_grad():
                t_out = self.teacher_model(
                    **teacher_inputs, output_hidden_states=True, use_cache=False
                )
            teacher_logits = self._completion_logits(t_out.logits, completion_length)
            teacher_hidden_last = t_out.hidden_states[-1]
            del t_out
        else:
            teacher_logits = self._batched_teacher_completion_logits(
                self.teacher_model,
                [
                    {
                        "name": "opd",
                        "inputs": teacher_inputs,
                        "completion_length": completion_length,
                    }
                ],
            )["opd"]

        # --- OPD token loss (identical to OPDTrainer) ----------------------------
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

        # --- TAM visual-evidence alignment loss ----------------------------------
        tam_loss_value = 0.0
        tam_stats: dict[str, float] = {}
        if run_tam and student_hidden_last is not None and teacher_hidden_last is not None:
            tam_loss, tam_stats = self._tam_loss(
                student_prompt, rollout, student_hidden_last, teacher_hidden_last
            )
            if tam_loss is not None and bool(torch.isfinite(tam_loss.detach())):
                loss = loss + self.lambda_tam * tam_loss
                tam_loss_value = float(tam_loss.detach())

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
        n_sel = tam_stats.get("tam_n_selected", 0.0)
        if n_sel > 0:
            metrics["loss_tam"] = (tam_loss_value, 1.0)
            metrics["tam_div"] = (tam_stats.get("tam_div_sum", 0.0), n_sel)
            metrics["tam_js"] = (tam_stats.get("tam_js_sum", 0.0), n_sel)
            metrics["tam_gate_mean"] = (tam_stats.get("tam_gate_sum", 0.0), n_sel)
            metrics["tam_n_selected"] = (
                n_sel,
                max(tam_stats.get("tam_valid_samples", 1.0), 1.0),
            )
        self._record_loss_metrics(metrics)

        if return_outputs:
            return loss, {"logits": student_logits.detach()}
        return loss
