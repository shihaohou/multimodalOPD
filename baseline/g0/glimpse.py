"""GLIMPSE (arXiv 2506.18985), tractable one-backward variant — the "using" probe.

GLIMPSE is a faithful, gradient×attention, *response-level* attribution for free-
form LVLM outputs: it explains the whole generated answer (not one token), over
both image and prompt-text, and is markedly more faithful than raw attention or
the TAM/Saliency-R1 maps (which is exactly why we use it to settle "does the
answer actually USE the region it looks at").

This is a faithful re-implementation of GLIMPSE's core gradient-attention fusion,
adapted to stay tractable on Qwen's thousands of visual tokens:

* **Eq. 5** per-head relevance ``G_ℓ^h = ReLU(g_ℓ^h ⊙ A_ℓ^h)`` with
  ``g_ℓ^h = ∂z/∂A_ℓ^h``.
* **Eq. 6** adaptive head weights ``w_ℓ^h = softmax( (1/λ) · ΣG_ℓ^h / Σ ReLU(g_ℓ^h) )``.
* **Eq. 8** fused per-layer relevance ``E_ℓ = Σ_h w_ℓ^h G_ℓ^h``.
* **Eq. 9–11** layer weights ``α_ℓ ∝ ‖Σ_h g_ℓ^h‖₁ · softmax(λ_d(ℓ+1))``.
* **Eq. 17** confidence weight ``p_t = softmax(z_t)`` over generated tokens.

We deviate from GLIMPSE in **one** place, for memory: the paper builds the full
``N×N`` relevance matrix and propagates it across layers (``R ← R + L_ℓ R`` with
``L_ℓ = I + α_ℓ E_ℓ``). At Qwen resolutions ``N`` is thousands, so the ``N×N``
propagation OOMs. Since the holistic aggregation only ever reads the *response
rows* of ``R``, we instead read the response rows of each ``E_ℓ`` directly and
combine them with ``α_ℓ`` (an additive, no-rollout approximation). This keeps the
gradient-attention faithfulness and the layer/head/confidence weighting; it drops
only the multi-hop indirection of the full rollout. Documented so the limitation
is explicit.

A single backward (``torch.autograd.grad`` of ``S = Σ_t z_t`` over the attention
tensors) is all that is needed. The β confidence-weights then re-aggregate that
*one* backward into three answer scopes for free:

  * **full** — the whole response;
  * **answer** — the last-K generated tokens (``answer_k``);
  * **boxed** — the tokens inside ``\\boxed{...}`` (``boxed_span``), the precise
    final answer, with last-K fallback handled by the caller.

Outputs per sample/condition:
  * a **visual saliency map** over the patch grid → IoU / energy-in-bbox /
    pointing-game vs the GT box (``IoU_GL``);
  * **vt_ratio** = visual mass / (visual + textual-prompt mass) — how
    image-driven that answer scope is (low ⇒ the answer is carried by text/prior;
    the "using-failure" signature).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from baseline.g0 import metrics
from baseline.g0.answer_spans import CompletionSpan, span_completion_mask
from baseline.g0.engine import BoxNorm, G0Model, grad_attention_forward, visual_grid


@dataclass
class GlimpseResult:
    iou_gl: float  # thresholded visual map vs GT grid mask (full response)
    bbox_iou: float
    pointing: float
    energy: float  # positive visual mass in GT box
    vt_ratio: float  # V / (V + T_prompt) over the FULL response
    visual_mass: float
    textual_mass: float
    self_mass: float  # mass on the response's own (already-generated) tokens
    pred_box_norm: Optional[BoxNorm]
    # ANSWER-SPAN variants (last-k generated tokens ≈ the answer) — these isolate
    # "does the final answer use the image" from CoT scaffolding dilution.
    vt_ratio_answer: float = float("nan")
    iou_gl_answer: float = float("nan")
    # BOXED-SPAN variants (tokens inside \boxed{...}; last-k fallback by the caller)
    # — the precise final answer, robust to post-answer chatter / long answers.
    vt_ratio_boxed: float = float("nan")
    iou_gl_boxed: float = float("nan")
    visual_map: Optional[np.ndarray] = None  # [H_grid, W_grid] full, for viz (optional)
    visual_map_boxed: Optional[np.ndarray] = None  # [H_grid, W_grid] boxed/answer, for viz


def _layer_response_relevance(
    g: torch.Tensor,
    a: torch.Tensor,
    response_rows: torch.Tensor,
    betas: list[torch.Tensor],
    lam: float,
) -> tuple[list[torch.Tensor], float]:
    """One layer's β-weighted response-row relevance for each β scope, + grad norm.

    ``g``/``a`` are ``[H,S,S]`` (gradient and attention for the layer, fp32).
    Implements Eq.5/6/8 but materializes only the response rows of ``E_ℓ`` (Eq.8)
    instead of the full ``[S,S]`` matrix; every β vector (full / answer / boxed,
    each normalized over the response) reuses the same ``e_rows``. Returns
    ``([rel_β for β in betas], g_ℓ)`` where each ``rel`` is ``[S]``.
    """
    relu_g = torch.relu(g)
    # Eq.6 head weights from full-matrix reductions (no [S,S] kept beyond g·a).
    g_relu_a = torch.relu(g * a)  # [H,S,S]
    num = g_relu_a.sum(dim=(1, 2))  # [H]
    den = relu_g.sum(dim=(1, 2)).clamp_min(1e-12)  # [H]
    w = torch.softmax((num / den) / max(lam, 1e-6), dim=0)  # [H]
    del g_relu_a

    # Eq.8 on response rows only: E_rows[t,:] = Σ_h w_h ReLU(g[h,t,:]·a[h,t,:]).
    g_rows = g.index_select(1, response_rows)  # [H, R, S]
    a_rows = a.index_select(1, response_rows)  # [H, R, S]
    e_rows = (w.view(-1, 1, 1) * torch.relu(g_rows * a_rows)).sum(0)  # [R, S]
    rels = [(beta.view(-1, 1) * e_rows).sum(0) for beta in betas]  # each [S]

    # Eq.9 layer gradient norm ‖Σ_h g_ℓ^h‖₁.
    g_layer_norm = float(g.sum(0).abs().sum().item())
    return rels, g_layer_norm


def _confidence_beta(p_t: torch.Tensor, mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """β over response rows: confidence ``p_t`` restricted to ``mask``, normalized.

    ``mask=None`` → the full response. Returns ``None`` if the masked confidence
    sums to ~0 (empty span), so the caller records NaN for that scope.
    """
    if mask is None:
        s = p_t.sum()
        return (p_t / s).float() if float(s) > 0 else None
    pa = p_t * mask
    s = pa.sum()
    return (pa / s).float() if float(s) > 1e-12 else None


def glimpse_relevance(
    gm: G0Model,
    inputs: dict,
    full_ids: torch.Tensor,
    prompt_len: int,
    completion_ids: torch.Tensor,
    *,
    answer_k: int = 16,
    boxed_span: Optional[CompletionSpan] = None,
    layers: Optional[tuple[int, ...]] = None,
    lam: float = 1.0,
    lambda_depth: float = 0.1,
    out=None,
) -> tuple[dict[str, Optional[np.ndarray]], dict]:
    """GLIMPSE relevance over all sequence positions for the FULL response, the
    ANSWER span (last ``answer_k``) and the BOXED span (``boxed_span``).

    Returns ``({"full": rel[S], "answer": rel[S]|None, "boxed": rel[S]|None}, info)``
    where ``rel[i] ≥ 0`` is how much input position ``i`` drives that scope. A
    scope is ``None`` when its span is empty.

    ``out`` may be a precomputed :func:`grad_attention_forward` output so the LH
    probe can reuse the same forward; if None, the forward is run here. After this
    call the graph is freed (``autograd.grad`` retains nothing), but the attention
    *values* in ``out.attentions`` remain readable for LH.
    """
    device = full_ids.device
    completion_ids = completion_ids.to(device)
    comp_len = int(completion_ids.numel())
    seq_len = int(full_ids.numel())

    # Predictor query rows for the generated tokens: position p predicts p+1, so
    # the row that emits completion token k (at absolute position prompt_len+k) is
    # prompt_len+k-1. Targets are the completion ids themselves.
    response_rows = torch.arange(prompt_len - 1, prompt_len + comp_len - 1, device=device)
    targets = completion_ids  # [R]

    if out is None:
        out = grad_attention_forward(gm, inputs, full_ids)
    logits = out.logits[0]  # [S, vocab]
    sel = logits.index_select(0, response_rows)  # [R, vocab]
    row_idx = torch.arange(comp_len, device=device)
    # Eq.17 confidence weight (detached — it weights aggregation, not the gradient).
    with torch.no_grad():
        probs = torch.softmax(sel.float(), dim=-1)
        p_t = probs[row_idx, targets].clamp_min(1e-12)  # [R]
        # answer span = last min(answer_k, comp_len) generated tokens.
        ak = max(1, min(int(answer_k), comp_len))
        ans_mask = torch.zeros(comp_len, device=device)
        ans_mask[comp_len - ak:] = 1.0
        boxed_mask = (
            span_completion_mask(comp_len, boxed_span, device=device)
            if boxed_span is not None
            else None
        )
        beta_full = _confidence_beta(p_t, None)
        beta_ans = _confidence_beta(p_t, ans_mask)
        beta_boxed = _confidence_beta(p_t, boxed_mask)

    scopes = ["full", "answer", "boxed"]
    betas_all = [beta_full, beta_ans, beta_boxed]
    active = [(name, b) for name, b in zip(scopes, betas_all) if b is not None]
    active_betas = [b for _, b in active]

    # Backward scalar S = Σ_t z_t (raw logit of the generated token). We keep the
    # gradient UNWEIGHTED (paper backprops raw z_t); β reweights at aggregation only
    # (putting β in both the scalar and the aggregation would double-apply it).
    scalar = sel[row_idx, targets].float().sum()
    assert scalar.requires_grad, (
        "GLIMPSE scalar has no grad — load the model with attn_implementation='eager' "
        "and do NOT freeze it (requires_grad must reach the attention tensors)."
    )

    attentions = list(out.attentions)
    layer_ids = tuple(range(len(attentions))) if layers is None else tuple(layers)
    grads = torch.autograd.grad(
        scalar, [attentions[l] for l in layer_ids], retain_graph=False, allow_unused=True
    )

    rel_layers: list[list[torch.Tensor]] = [[] for _ in active_betas]
    g_norms: list[float] = []
    used_layers: list[int] = []
    for gl, l in zip(grads, layer_ids):
        if gl is None:
            continue
        g = gl[0].float()  # [H,S,S]
        a = attentions[l][0].detach().float()  # [H,S,S]
        rels, gnorm = _layer_response_relevance(g, a, response_rows, active_betas, lam)
        for k, rel in enumerate(rels):
            rel_layers[k].append(rel)
        g_norms.append(gnorm)
        used_layers.append(l)
        del g, a, gl
    del grads, out, logits, sel
    if not used_layers:
        raise RuntimeError("GLIMPSE: no layer produced a gradient (all unused?).")

    # Eq.10–11 layer weights: grad-norm × depth prior, normalized.
    g_norm_t = torch.tensor(g_norms, dtype=torch.float64)
    depth = torch.tensor([float(l + 1) for l in used_layers], dtype=torch.float64)
    s_depth = torch.softmax(lambda_depth * depth, dim=0)
    alpha = g_norm_t * s_depth
    alpha = alpha / alpha.sum().clamp_min(1e-12)

    relevances: dict[str, Optional[np.ndarray]] = {name: None for name in scopes}
    for (name, _), per_layer in zip(active, rel_layers):
        relevance = torch.zeros(seq_len, dtype=torch.float64)
        for a_l, rel in zip(alpha.tolist(), per_layer):
            relevance += a_l * rel.double().cpu()
        relevances[name] = torch.relu(relevance).numpy()

    info = {"used_layers": used_layers, "alpha": alpha.tolist(),
            "prompt_len": prompt_len, "seq_len": seq_len, "answer_k": ak,
            "boxed_span": list(boxed_span) if boxed_span is not None else None}
    return relevances, info


def glimpse_probe(
    gm: G0Model,
    inputs: dict,
    full_ids: torch.Tensor,
    prompt_len: int,
    completion_ids: torch.Tensor,
    bbox: BoxNorm,
    *,
    layers: Optional[tuple[int, ...]] = None,
    answer_k: int = 16,
    boxed_span: Optional[CompletionSpan] = None,
    lam: float = 1.0,
    lambda_depth: float = 0.1,
    threshold: str = "mean",
    keep_map: bool = False,
    out=None,
) -> GlimpseResult:
    """Full GLIMPSE probe for one sample/condition: visual IoU + vt_ratio, for the
    full response, the answer span (last-K) AND the boxed span.

    ``out`` may be a precomputed :func:`grad_attention_forward` output to share the
    forward with the LH probe.
    """
    visual_positions, grid_hw = visual_grid(gm, full_ids, inputs["image_grid_thw"])
    h_grid, w_grid = grid_hw
    relevances, _ = glimpse_relevance(
        gm, inputs, full_ids, prompt_len, completion_ids,
        answer_k=answer_k, boxed_span=boxed_span, layers=layers,
        lam=lam, lambda_depth=lambda_depth, out=out,
    )
    rel_full = relevances["full"]

    vis_idx = visual_positions.detach().cpu().numpy()
    n_pos = rel_full.shape[0]
    prompt_mask = np.zeros(n_pos, dtype=bool)
    prompt_mask[:prompt_len] = True
    visual_mask = np.zeros_like(prompt_mask)
    visual_mask[vis_idx] = True
    textual_mask = prompt_mask & ~visual_mask  # prompt text (system+question[+hint])
    self_mask = ~prompt_mask                   # the response's own (autoregressive) tokens

    def _vt(rel):
        if rel is None:
            return float("nan")
        v = float(rel[visual_mask].sum())
        t = float(rel[textual_mask].sum())
        denom = v + t
        return (v / denom) if denom > 0 else float("nan")

    def _iou(rel):
        if rel is None:
            return float("nan")
        vmap = rel[vis_idx].reshape(h_grid, w_grid)
        return float(metrics.iou_map_vs_gt(vmap, bbox, sigma=0.0, threshold=threshold)["mask_iou"])

    # Full-response masses (kept for the teacher/student gap analysis).
    visual_mass = float(rel_full[visual_mask].sum())
    textual_mass = float(rel_full[textual_mask].sum())
    self_mass = float(rel_full[self_mask].sum())
    vt_ratio = _vt(rel_full)
    vt_ratio_answer = _vt(relevances["answer"])
    vt_ratio_boxed = _vt(relevances["boxed"])

    visual_map = rel_full[vis_idx].reshape(h_grid, w_grid)
    res = metrics.iou_map_vs_gt(visual_map, bbox, sigma=0.0, threshold=threshold)
    pred_box = metrics.bbox_from_mask(
        metrics.binarize_mean_relu(visual_map)
        if threshold == "mean"
        else metrics.binarize_top_frac(visual_map, 0.25)
    )
    pred_norm = metrics.grid_box_to_norm(pred_box, h_grid, w_grid) if pred_box else None

    rel_boxed = relevances["boxed"] if relevances["boxed"] is not None else relevances["answer"]
    visual_map_boxed = (
        rel_boxed[vis_idx].reshape(h_grid, w_grid) if (keep_map and rel_boxed is not None) else None
    )

    return GlimpseResult(
        iou_gl=float(res["mask_iou"]),
        bbox_iou=float(res["bbox_iou"]),
        pointing=float(res["pointing"]),
        energy=float(res["energy"]) if np.isfinite(res["energy"]) else 0.0,
        vt_ratio=float(vt_ratio),
        visual_mass=visual_mass,
        textual_mass=textual_mass,
        self_mass=self_mass,
        pred_box_norm=pred_norm,
        vt_ratio_answer=float(vt_ratio_answer),
        iou_gl_answer=_iou(relevances["answer"]),
        vt_ratio_boxed=float(vt_ratio_boxed),
        iou_gl_boxed=_iou(relevances["boxed"]),
        visual_map=visual_map if keep_map else None,
        visual_map_boxed=visual_map_boxed,
    )
