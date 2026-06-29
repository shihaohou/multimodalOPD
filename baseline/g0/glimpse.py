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
tensors) is all that is needed — matching the diagnostic's "needs one backward".

Outputs per sample/condition:
  * a **visual saliency map** over the patch grid → IoU / energy-in-bbox /
    pointing-game vs the GT box (``IoU_GL``);
  * **vt_ratio** = visual mass / (visual + textual-prompt mass) — how
    image-driven the answer is (low ⇒ the answer is carried by text/prior; the
    "using-failure" signature).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from baseline.g0 import metrics
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
    visual_map: Optional[np.ndarray] = None  # [H_grid, W_grid], for viz (optional)


def _layer_response_relevance(
    g: torch.Tensor,
    a: torch.Tensor,
    response_rows: torch.Tensor,
    beta_full: torch.Tensor,
    beta_ans: torch.Tensor,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """One layer's β-weighted response-row relevance for the FULL response and the
    ANSWER span, plus grad-norm ``g_ℓ``. Returns ``(rel_full[S], rel_ans[S], g_ℓ)``.

    ``g``/``a`` are ``[H,S,S]`` (gradient and attention for the layer, fp32).
    Implements Eq.5/6/8 but materializes only the response rows of ``E_ℓ`` (Eq.8)
    instead of the full ``[S,S]`` matrix; the two β vectors (full vs answer-only,
    each normalized) reuse the same ``e_rows``.
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
    rel_full = (beta_full.view(-1, 1) * e_rows).sum(0)  # [S]
    rel_ans = (beta_ans.view(-1, 1) * e_rows).sum(0)  # [S]

    # Eq.9 layer gradient norm ‖Σ_h g_ℓ^h‖₁.
    g_layer_norm = float(g.sum(0).abs().sum().item())
    return rel_full, rel_ans, g_layer_norm


def glimpse_relevance(
    gm: G0Model,
    inputs: dict,
    full_ids: torch.Tensor,
    prompt_len: int,
    completion_ids: torch.Tensor,
    *,
    answer_k: int = 16,
    layers: Optional[tuple[int, ...]] = None,
    lam: float = 1.0,
    lambda_depth: float = 0.1,
    out=None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """GLIMPSE relevance over all sequence positions, for the FULL response and the
    ANSWER span (last ``answer_k`` generated tokens). Returns
    ``(relevance_full[S], relevance_answer[S], info)``; ``relevance[i] ≥ 0`` is how
    much input position ``i`` drives that span.

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
        p_t = probs[row_idx, targets].clamp_min(1e-12)
        beta_full = (p_t / p_t.sum()).float()  # [R] over the whole response
        # answer span = last min(answer_k, comp_len) generated tokens.
        ak = max(1, min(int(answer_k), comp_len))
        ans = torch.zeros(comp_len, device=device)
        ans[comp_len - ak:] = 1.0
        pa = p_t * ans
        beta_ans = (pa / pa.sum().clamp_min(1e-12)).float()  # [R], zero outside the answer span
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

    rel_full_layers: list[torch.Tensor] = []
    rel_ans_layers: list[torch.Tensor] = []
    g_norms: list[float] = []
    used_layers: list[int] = []
    for gl, l in zip(grads, layer_ids):
        if gl is None:
            continue
        g = gl[0].float()  # [H,S,S]
        a = attentions[l][0].detach().float()  # [H,S,S]
        rel_f, rel_a, gnorm = _layer_response_relevance(g, a, response_rows, beta_full, beta_ans, lam)
        rel_full_layers.append(rel_f)
        rel_ans_layers.append(rel_a)
        g_norms.append(gnorm)
        used_layers.append(l)
        del g, a, gl
    del grads, out, logits, sel
    if not rel_full_layers:
        raise RuntimeError("GLIMPSE: no layer produced a gradient (all unused?).")

    # Eq.10–11 layer weights: grad-norm × depth prior, normalized.
    g_norm_t = torch.tensor(g_norms, dtype=torch.float64)
    depth = torch.tensor([float(l + 1) for l in used_layers], dtype=torch.float64)
    s_depth = torch.softmax(lambda_depth * depth, dim=0)
    alpha = g_norm_t * s_depth
    alpha = alpha / alpha.sum().clamp_min(1e-12)

    relevance = torch.zeros(seq_len, dtype=torch.float64)
    relevance_ans = torch.zeros(seq_len, dtype=torch.float64)
    for a_l, rf, ra in zip(alpha.tolist(), rel_full_layers, rel_ans_layers):
        relevance += a_l * rf.double().cpu()
        relevance_ans += a_l * ra.double().cpu()
    relevance = torch.relu(relevance).numpy()
    relevance_ans = torch.relu(relevance_ans).numpy()

    info = {"used_layers": used_layers, "alpha": alpha.tolist(),
            "prompt_len": prompt_len, "seq_len": seq_len, "answer_k": ak}
    return relevance, relevance_ans, info


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
    lam: float = 1.0,
    lambda_depth: float = 0.1,
    threshold: str = "mean",
    keep_map: bool = False,
    out=None,
) -> GlimpseResult:
    """Full GLIMPSE probe for one sample/condition: visual IoU + vt_ratio, for the
    full response AND the answer span.

    ``out`` may be a precomputed :func:`grad_attention_forward` output to share the
    forward with the LH probe.
    """
    visual_positions, grid_hw = visual_grid(gm, full_ids, inputs["image_grid_thw"])
    h_grid, w_grid = grid_hw
    relevance, relevance_ans, _ = glimpse_relevance(
        gm, inputs, full_ids, prompt_len, completion_ids,
        answer_k=answer_k, layers=layers, lam=lam, lambda_depth=lambda_depth, out=out,
    )

    vis_idx = visual_positions.detach().cpu().numpy()
    prompt_mask = np.zeros(relevance.shape[0], dtype=bool)
    prompt_mask[:prompt_len] = True
    visual_mask = np.zeros_like(prompt_mask)
    visual_mask[vis_idx] = True
    textual_mask = prompt_mask & ~visual_mask  # prompt text (system+question[+hint])
    self_mask = ~prompt_mask                   # the response's own (autoregressive) tokens

    def _vt(rel):
        v = float(rel[visual_mask].sum())
        t = float(rel[textual_mask].sum())
        denom = v + t
        return v, t, float(rel[self_mask].sum()), (v / denom if denom > 0 else float("nan"))

    visual_mass, textual_mass, self_mass, vt_ratio = _vt(relevance)
    _, _, _, vt_ratio_answer = _vt(relevance_ans)

    visual_map = relevance[vis_idx].reshape(h_grid, w_grid)
    visual_map_ans = relevance_ans[vis_idx].reshape(h_grid, w_grid)
    res = metrics.iou_map_vs_gt(visual_map, bbox, sigma=0.0, threshold=threshold)
    res_ans = metrics.iou_map_vs_gt(visual_map_ans, bbox, sigma=0.0, threshold=threshold)
    pred_box = metrics.bbox_from_mask(
        metrics.binarize_mean_relu(visual_map)
        if threshold == "mean"
        else metrics.binarize_top_frac(visual_map, 0.25)
    )
    pred_norm = metrics.grid_box_to_norm(pred_box, h_grid, w_grid) if pred_box else None

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
        iou_gl_answer=float(res_ans["mask_iou"]),
        visual_map=visual_map if keep_map else None,
    )
