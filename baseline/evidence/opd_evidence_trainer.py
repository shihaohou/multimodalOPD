"""OPD trainer with the differentiable evidence-alignment loss (Step 3).

``OPDEvidenceTrainer`` adds, on top of the vanilla OPD token-distillation loss, a
saliency evidence-alignment term:

    loss = lambda_opd * L_opd  +  lambda_evidence * L_evidence

``L_opd`` is exactly :class:`baseline.opd_trainer.OPDTrainer`'s reverse-KL token
loss; ``L_evidence`` pulls the student's per-token saliency map toward the frozen
teacher's, on a small subset of high-KL answer tokens, via the gated
signed-Pearson loss (see :mod:`baseline.evidence.evidence_loss`).

**One forward, not two.** The saliency engine needs in-graph attention weights, so
the student forward runs with ``output_attentions=True`` under the **eager**
attention implementation. Crucially it is the SAME forward that produces the OPD
logits — doing a second grad forward through the (DeepSpeed-wrapped) student would
make ZeRO reduce each shared parameter's gradient twice in one backward
("parameter ... has already been reduced"). So the OPD logits and the evidence
attentions/hidden-states both come from a single forward; ``loss.backward()``
traverses one graph and each parameter is reduced once.

Because that single forward is eager + ``output_attentions`` over the whole
micro-batch, the **evidence run wants a small ``per_device_train_batch_size``**
(1-2) with higher grad-accum — eager attention over thousands of visual tokens is
the dominant memory cost. Steps whose rollouts have no valid ``<reason>``+answer
span (or when gradient checkpointing swallows the attentions) fall back to a
cheap OPD-only forward. ``local_hf`` teacher only.
"""

from __future__ import annotations

import contextlib
from typing import Any

import torch
import torch.nn as nn

from baseline.evidence.evidence_loss import (
    evidence_alignment_loss,
    per_token_kl,
    top_indices_by_score,
)
from baseline.evidence.saliency_engine import (
    compute_token_saliency_maps,
    resolve_model_parts,
)
from baseline.evidence.span_utils import parse_batch_spans
from baseline.opd_losses import masked_topk_kl_loss
from baseline.opd_trainer import OPDTrainer


@contextlib.contextmanager
def force_eager_attention(model: nn.Module):
    """Temporarily set every ``_attn_implementation`` in ``model`` to ``eager``.

    ``output_attentions=True`` only returns attention weights under eager
    attention (SDPA/FlashAttention drop them). Walks both the config tree
    (text/vision sub-configs) and every submodule ``.config`` so the switch
    reaches the decoder attention modules regardless of how the family nests
    its configs. Restored on exit.
    """
    configs: list[Any] = []
    seen: set[int] = set()

    def visit(cfg: Any) -> None:
        if cfg is None or id(cfg) in seen:
            return
        seen.add(id(cfg))
        if hasattr(cfg, "_attn_implementation"):
            configs.append(cfg)
        for sub in ("text_config", "vision_config", "thinker_config"):
            visit(getattr(cfg, sub, None))

    visit(getattr(model, "config", None))
    for module in model.modules():
        cfg = getattr(module, "config", None)
        if cfg is not None and hasattr(cfg, "_attn_implementation") and id(cfg) not in seen:
            seen.add(id(cfg))
            configs.append(cfg)

    saved = [c._attn_implementation for c in configs]
    try:
        for c in configs:
            c._attn_implementation = "eager"
        yield
    finally:
        for c, old in zip(configs, saved):
            c._attn_implementation = old


class OPDEvidenceTrainer(OPDTrainer):
    def __init__(
        self,
        *args: Any,
        lambda_evidence: float = 1.0,
        evidence_max_samples: int = 1,
        evidence_layers: tuple[int, ...] | None = None,
        evidence_top_ratio: float = 0.2,
        evidence_min_tokens: int = 1,
        evidence_max_tokens: int = 8,
        evidence_signed: bool = True,
        evidence_kl_direction: str = "forward",
        evidence_gate_temp: float = 1.0,
        evidence_gate_h0: float = 0.9,
        evidence_gate_tau: float = 0.1,
        evidence_kl_threshold: float = 0.0,
        evidence_mass_threshold: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.teacher_source != "local_hf":
            raise ValueError(
                "OPDEvidenceTrainer requires teacher_source='local_hf' (the evidence "
                "term needs a full local teacher forward for the teacher saliency map)."
            )
        self.lambda_evidence = float(lambda_evidence)
        self.evidence_max_samples = int(evidence_max_samples)
        self.evidence_layers = (
            tuple(int(x) for x in evidence_layers) if evidence_layers else None
        )
        self.evidence_top_ratio = float(evidence_top_ratio)
        self.evidence_min_tokens = int(evidence_min_tokens)
        self.evidence_max_tokens = int(evidence_max_tokens)
        self.evidence_signed = bool(evidence_signed)
        self.evidence_kl_direction = evidence_kl_direction
        self.evidence_gate_temp = float(evidence_gate_temp)
        self.evidence_gate_h0 = float(evidence_gate_h0)
        self.evidence_gate_tau = float(evidence_gate_tau)
        self.evidence_kl_threshold = float(evidence_kl_threshold)
        self.evidence_mass_threshold = float(evidence_mass_threshold)
        self._student_parts = None
        self._teacher_parts = None

    def _evidence_loss(
        self,
        unwrapped_student: nn.Module,
        student_prompt: dict[str, torch.Tensor],
        rollout: dict[str, torch.Tensor],
        completion_ids: torch.Tensor,
        spans_list: list,
        valid_idx: list[int],
        s_attentions: tuple[torch.Tensor, ...],
        s_hidden: tuple[torch.Tensor, ...],
        t_attentions: tuple[torch.Tensor, ...],
        t_hidden: tuple[torch.Tensor, ...],
        student_logits_det: torch.Tensor,
        teacher_logits_det: torch.Tensor,
    ) -> tuple[torch.Tensor | None, dict[str, float]]:
        """Saliency evidence-alignment loss from the ALREADY-computed attentions /
        hidden states (no forward here — both models were forwarded once in
        ``compute_loss``). ``*_logits_det`` are the detached OPD completion logits
        ``[B, C, vocab]`` reused for token selection."""
        if self._student_parts is None:
            self._student_parts = resolve_model_parts(unwrapped_student)
        if self._teacher_parts is None:
            self._teacher_parts = resolve_model_parts(self.teacher_model)
        s_parts, t_parts = self._student_parts, self._teacher_parts

        prompt_length = int(student_prompt["input_ids"].shape[1])
        generated_ids = rollout["generated_ids"]
        image_grid_thw = student_prompt.get("image_grid_thw")
        device = completion_ids.device

        loss_terms: list[torch.Tensor] = []
        corr_sum = 0.0
        gate_sum = 0.0
        n_tokens_total = 0
        for b in valid_idx:
            spans = spans_list[b]
            rs, re_ = spans.reason
            a_start, a_end = spans.answer

            ans_slice = slice(a_start, a_end + 1)
            kl = per_token_kl(
                student_logits_det[b, ans_slice],
                teacher_logits_det[b, ans_slice],
                temperature=self.distill_temperature,
                direction=self.evidence_kl_direction,
            ).detach()
            sel = top_indices_by_score(
                kl,
                self.evidence_top_ratio,
                min_keep=self.evidence_min_tokens,
                max_keep=self.evidence_max_tokens,
            )
            if sel.numel() == 0:
                continue
            sel_completion = torch.arange(a_start, a_end + 1, device=device)[sel]
            answer_q = (prompt_length + sel_completion - 1).clamp_min(0)
            reason_k = prompt_length + torch.arange(rs, re_ + 1, device=device)
            reason_q = (
                prompt_length + torch.arange(rs, re_ + 1, device=device) - 1
            ).clamp_min(0)
            direction_ids = completion_ids[b, sel_completion]

            visual_positions = (generated_ids[b] == s_parts.image_token_id).nonzero(
                as_tuple=True
            )[0]
            grid = image_grid_thw[b]
            grid_hw = (
                int(grid[1]) // s_parts.spatial_merge_size,
                int(grid[2]) // s_parts.spatial_merge_size,
            )

            student_maps = compute_token_saliency_maps(
                unwrapped_student,
                s_attentions,
                s_hidden,
                batch_index=b,
                answer_query_positions=answer_q,
                reason_key_positions=reason_k,
                reason_query_positions=reason_q,
                visual_positions=visual_positions,
                direction_ids=direction_ids,
                grid_hw=grid_hw,
                layers=self.evidence_layers,
                signed=self.evidence_signed,
                parts=s_parts,
            )
            with torch.no_grad():
                teacher_maps = compute_token_saliency_maps(
                    self.teacher_model,
                    t_attentions,
                    t_hidden,
                    batch_index=b,
                    answer_query_positions=answer_q,
                    reason_key_positions=reason_k,
                    reason_query_positions=reason_q,
                    visual_positions=visual_positions,
                    direction_ids=direction_ids,
                    grid_hw=grid_hw,
                    layers=self.evidence_layers,
                    signed=self.evidence_signed,
                    parts=t_parts,
                ).detach()

            loss_b, stats_b = evidence_alignment_loss(
                student_maps,
                teacher_maps,
                gate_temp=self.evidence_gate_temp,
                gate_h0=self.evidence_gate_h0,
                gate_tau=self.evidence_gate_tau,
                kl_scores=kl[sel],
                kl_threshold=self.evidence_kl_threshold,
                mass_threshold=self.evidence_mass_threshold,
            )
            loss_terms.append(loss_b)
            corr_sum += float(stats_b["ev_corr"]) * int(stats_b["ev_n"])
            gate_sum += float(stats_b["ev_gate_mean"]) * int(stats_b["ev_n"])
            n_tokens_total += int(stats_b["ev_n"])

        if not loss_terms:
            return None, {"ev_valid_samples": float(len(valid_idx))}
        loss = torch.stack(loss_terms).mean()
        stats = {
            "ev_valid_samples": float(len(valid_idx)),
            "ev_n_selected": float(n_tokens_total),
            "ev_corr_sum": corr_sum,
            "ev_gate_sum": gate_sum,
        }
        return loss, stats

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

        # Decide whether evidence runs this step (needs valid <reason>+answer spans).
        spans_list = None
        valid_idx: list[int] = []
        if self.lambda_evidence > 0:
            tokenizer = getattr(self.processor, "tokenizer", self.processor)
            spans_list = parse_batch_spans(tokenizer, completion_ids, completion_attention)
            valid_idx = [b for b, s in enumerate(spans_list) if s.valid][
                : self.evidence_max_samples
            ]
        run_evidence = bool(valid_idx)
        unwrapped = self.accelerator.unwrap_model(model)

        # --- SINGLE student forward (one grad graph; see module docstring) ------
        student_inputs = self._with_completion(
            student_prompt,
            full_input_ids=rollout["generated_ids"],
            full_attention_mask=rollout["generated_attention_mask"],
        )
        student_inputs["logits_to_keep"] = completion_length + 1
        if run_evidence:
            with force_eager_attention(unwrapped):
                student_outputs = model(
                    **student_inputs, output_attentions=True, output_hidden_states=True
                )
            if not getattr(student_outputs, "attentions", None):
                print(
                    "[OPD-evidence] student forward returned no attentions (gradient "
                    "checkpointing?); OPD-only this step. Set GRADIENT_CHECKPOINTING=false "
                    "to enable evidence.",
                    flush=True,
                )
                run_evidence = False
        else:
            student_outputs = model(**student_inputs)
        student_logits = self._completion_logits(
            student_outputs.logits, completion_length
        )
        s_attentions = student_outputs.attentions if run_evidence else None
        s_hidden = student_outputs.hidden_states if run_evidence else None

        # --- teacher forward (frozen, no grad) ---------------------------------
        teacher_inputs = self._append_completion(
            student_prompt, completion_ids, rollout["completion_attention_mask"]
        )
        t_attentions = t_hidden = None
        if run_evidence:
            teacher_inputs["logits_to_keep"] = completion_length + 1
            with torch.no_grad(), force_eager_attention(self.teacher_model):
                t_out = self.teacher_model(
                    **teacher_inputs,
                    output_attentions=True,
                    output_hidden_states=True,
                    use_cache=False,
                )
            teacher_logits = self._completion_logits(t_out.logits, completion_length)
            t_attentions, t_hidden = t_out.attentions, t_out.hidden_states
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

        # --- evidence alignment loss (same single forward's attentions) --------
        evidence_loss_value = 0.0
        evidence_stats: dict[str, float] = {}
        if run_evidence:
            ev_loss, evidence_stats = self._evidence_loss(
                unwrapped,
                student_prompt,
                rollout,
                completion_ids,
                spans_list,
                valid_idx,
                s_attentions,
                s_hidden,
                t_attentions,
                t_hidden,
                student_kl_logits.detach(),
                teacher_kl_logits.detach(),
            )
            if ev_loss is not None and bool(torch.isfinite(ev_loss.detach())):
                loss = loss + self.lambda_evidence * ev_loss
                evidence_loss_value = float(ev_loss.detach())

        # --- metrics -----------------------------------------------------------
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
        n_sel = evidence_stats.get("ev_n_selected", 0.0)
        if n_sel > 0:
            metrics["loss_ev"] = (evidence_loss_value, 1.0)
            metrics["ev_corr"] = (evidence_stats.get("ev_corr_sum", 0.0), n_sel)
            metrics["ev_gate_mean"] = (evidence_stats.get("ev_gate_sum", 0.0), n_sel)
            metrics["ev_n_selected"] = (
                n_sel,
                max(evidence_stats.get("ev_valid_samples", 1.0), 1.0),
            )
        self._record_loss_metrics(metrics)

        if return_outputs:
            return loss, {"logits": student_logits.detach()}
        return loss
