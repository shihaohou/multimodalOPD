"""Saliency-R1 saliency map (CVPR'26, arXiv 2604.04500) — secondary G0 baseline.

Saliency-R1 (peterant330/Saliency_R1) is an *efficient, training-loop-friendly*
visual attribution: a **value-weighted logit-lens routed through the thinking
bottleneck**. It is NOT perturbation-causal like EAGLE — the paper itself notes it
only models the visual tokens' *direct* contribution — so we use it as a SECONDARY
baseline next to EAGLE (the causal arbiter) and LH (the attention baseline):
"if we swap in the Saliency-R1 map, does the looking-vs-using verdict hold?"

Reimplemented for Qwen3-VL from **one eager teacher-forcing forward** over
``prompt+completion`` (the same forward LH/GLIMPSE already use), so we do NOT need
the paper's patched ``transformers`` / per-generation-step ``generate``. We grab
per-layer **value vectors** with a forward hook on each decoder layer's
``self_attn`` (registered around that one forward).

Algorithm (per layer ℓ, faithful to grpo_trainer.py:1815-1847):
  * ``V_ℓ`` = value states, GQA-repeated to H heads, sliced to the PROMPT VISUAL
    tokens (``input_ids == image_token_id``).                       [H, P, d_head]
  * ``think_attn`` = answer rows' attention onto the THINKING token span. [A, H, T]
  * ``token_attn`` = thinking rows' attention onto the visual tokens.    [H, T, P]
  * ``agg = think_attn @ token_attn``  (visual→thinking→answer flow).    [A, H, P]
  * ``sv = (agg ⊙ V_ℓ)`` → per-layer ``o_proj`` → accumulate over layers.
  * ``norm`` → ``lm_head`` → take the answer token's logit per visual position →
    reshape to the merged grid ``(grid_h//merge, grid_w//merge)`` → **ReLU** →
    sum over the answer span = the holistic map.

We additionally keep the **signed** map (drop the final ReLU): ``pos`` (image
SUPPORTS the answer token), ``neg`` (image OPPOSES it), ``abs`` (|contribution|).
The signed split tests the OPD signed-VD intuition (negative visual evidence is
information, not noise), which the official ReLU-only reward discards.

If the completion has no ``<think>…</think>``, the holistic route is undefined →
we fall back to the **direct-answer** map (answer rows attend straight to the
visual tokens, same value-weighted lens without the thinking hop).

Metrics (per map): ``mass_gt`` = saliency in GT box / total (the paper's reward —
NOTE we fix its ``y2 = shape[1]`` height/width bug to ``shape[0]``), the
area-adjusted ``mass_enrich`` (mass_gt / bbox-area, since the random baseline of
mass_gt is the box area, not 0), ``pointing``, and ``iou_top20``/``iou_top30``
(top-k% positive cells → bbox IoU).

CPU self-test: ``python -m baseline.g0.salr1_probe`` (geometry/mass only).
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from baseline.g0 import metrics
from baseline.g0.answer_spans import CompletionSpan, resolve_answer_spans
from baseline.g0.engine import BoxNorm, G0Model


# --------------------------------------------------------------- value-vector hook
@contextmanager
def _capture_value_states(gm: G0Model):
    """Hook each decoder layer's ``self_attn`` to capture its value projection.

    Yields a dict ``{layer_idx: V}`` where ``V`` is ``[B, n_kv_heads, S, d_head]``
    (pre-GQA-repeat), matching ``past_key_values.layers[ℓ].values``. We read it from
    the attention module's ``v_proj`` output reshaped to heads — robust across the
    Qwen2.5/Qwen3 VL text stacks (both expose ``self_attn.v_proj`` + ``num_key_value_heads``).
    """
    layers = gm.parts.text_model.layers
    store: dict[int, torch.Tensor] = {}
    handles = []

    def mk_hook(idx, attn_mod):
        n_kv = getattr(attn_mod, "num_key_value_heads", None) or getattr(
            attn_mod.config, "num_key_value_heads", None)
        v_proj = attn_mod.v_proj

        def hook(_mod, inp, _out):
            x = inp[0]  # [B, S, hidden]
            v = v_proj(x)  # [B, S, n_kv*d_head]
            b, s, _ = v.shape
            d_head = v.shape[-1] // n_kv
            store[idx] = v.view(b, s, n_kv, d_head).transpose(1, 2).detach()  # [B,n_kv,S,d]
        return hook

    for idx, layer in enumerate(layers):
        attn_mod = layer.self_attn
        handles.append(attn_mod.register_forward_hook(mk_hook(idx, attn_mod)))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


def _repeat_kv(v: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[B, n_kv, S, d] → [B, n_kv*n_rep, S, d] (GQA expand, like the official repeat_v)."""
    b, n_kv, s, d = v.shape
    if n_rep == 1:
        return v
    return v[:, :, None, :, :].expand(b, n_kv, n_rep, s, d).reshape(b, n_kv * n_rep, s, d)


# --------------------------------------------------------------- map metrics
def _topk_mask(map_2d: np.ndarray, frac: float) -> np.ndarray:
    """Keep the top ``frac`` of cells by value, among POSITIVE cells only.

    Guards the flat/all-zero case: a naive ``arr >= partition(...)`` picks ``thr=0``
    when the map is empty/constant and floods the whole grid, turning ``iou_top*``
    into a full-image-box IoU that silently inflates ``corr(correct, salr1_iou)``.
    Here an all-zero (or all-≤0) map yields an EMPTY mask, and the threshold is taken
    over the positive support so a sparse map can't select non-positive cells.
    """
    arr = np.nan_to_num(np.asarray(map_2d, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size == 0 or float(arr.max()) <= 0:
        return np.zeros_like(arr, dtype=bool)
    pos = arr[arr > 0]
    k = min(max(1, int(round(frac * arr.size))), int(pos.size))
    thr = np.partition(pos, -k)[-k]
    return (arr >= thr) & (arr > 0)


def mass_in_box(map_2d: np.ndarray, bbox: BoxNorm) -> float:
    """Saliency mass inside the GT box / total mass — the paper's reward.

    Uses ``gt_box_to_grid_mask`` (floor/ceil overlap, height=axis0/width=axis1) —
    fixing the official code's ``y2 = saliency.shape[1]`` (width) bug.
    """
    arr = np.clip(np.asarray(map_2d, dtype=np.float64), 0.0, None)
    total = float(arr.sum())
    if total <= 0:
        return float("nan")
    h, w = arr.shape
    gt = metrics.gt_box_to_grid_mask(bbox, h, w)
    return float(arr[gt].sum()) / total


def _entropy(map_2d: np.ndarray) -> float:
    p = np.clip(np.asarray(map_2d, dtype=np.float64), 0.0, None)
    s = p.sum()
    if s <= 0:
        return float("nan")
    p = (p / s).ravel()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def map_metrics(map_2d: np.ndarray, bbox: BoxNorm) -> dict:
    """All single-map metrics for a (positive) saliency map vs the GT box.

    ``mass_enrich`` / ``mass_minus_area`` area-adjust ``mass_gt`` (whose random
    baseline is the GT-box AREA fraction, not 0): a diffuse map over a box covering
    30% of the image already scores mass_gt≈0.30, so cross-subset comparison needs
    the area-relative version. ``valid`` flags a non-empty positive map.
    """
    arr = np.clip(np.asarray(map_2d, dtype=np.float64), 0.0, None)
    h, w = arr.shape
    gt = metrics.gt_box_to_grid_mask(bbox, h, w)
    pos_sum = float(arr.sum())
    bbox_area = float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    mass = mass_in_box(arr, bbox)
    out = {
        "mass_gt": mass,
        "mass_enrich": (mass / bbox_area) if (mass == mass and bbox_area > 1e-6) else float("nan"),
        "mass_minus_area": (mass - bbox_area) if mass == mass else float("nan"),
        "pointing": float(metrics.pointing_hit(arr, gt)),
        "entropy": _entropy(arr),
        "pos_sum": pos_sum,
        "valid": bool(pos_sum > 0),
    }
    for frac, tag in ((0.2, "top20"), (0.3, "top30")):
        m = _topk_mask(arr, frac)
        box = metrics.bbox_from_mask(m)
        norm = metrics.grid_box_to_norm(box, h, w) if box else None
        out[f"iou_{tag}"] = float(metrics.mask_iou(m, gt))
        out[f"bbox_iou_{tag}"] = float(metrics.bbox_iou_norm(norm, bbox)) if norm else 0.0
        out[f"area_{tag}"] = float(m.mean())
    return out


# --------------------------------------------------------------- the map itself
@dataclass
class Salr1Result:
    span_mode: str          # "holistic" (had <think>) | "direct" (fallback)
    pos: dict               # map_metrics of the ReLU(+) holistic/direct map (official)
    neg: dict               # map_metrics of ReLU(−) — image OPPOSES the answer token
    abs: dict               # map_metrics of |contribution|
    mass_gt: float          # convenience alias = pos["mass_gt"] (the headline)
    pos_map: Optional[np.ndarray] = None   # [H_grid, W_grid] for viz
    signed_map: Optional[np.ndarray] = None


def _answer_logit_lens(
    gm: G0Model,
    out,
    values: dict[int, torch.Tensor],
    *,
    visual_positions: torch.Tensor,
    answer_rows: torch.Tensor,
    think_rows: Optional[torch.Tensor],
    target_ids: torch.Tensor,
    grid_hw: tuple[int, int],
    layers: tuple[int, ...],
) -> np.ndarray:
    """Signed per-visual-position contribution to the answer tokens' logits.

    Returns the SIGNED holistic map ``[H_grid, W_grid]`` (sum over the answer span);
    the caller derives pos/neg/abs. ``think_rows=None`` → direct-answer route
    (answer rows attend straight to the visual tokens).
    """
    device = next(gm.model.parameters()).device
    text = gm.parts.text_model
    lm_head = gm.model.lm_head if hasattr(gm.model, "lm_head") else gm.model.get_output_embeddings()
    h_grid, w_grid = grid_hw
    P = int(visual_positions.numel())
    A = int(answer_rows.numel())
    vis = visual_positions.to(device)

    # attentions[ℓ]: [B,H,S,S]; we read query rows (answer / think) over key cols.
    attn = out.attentions
    acc = None  # accumulated [A, P, d_model] over layers
    for l in layers:
        a = attn[l][0].to(device)  # [H,S,S]
        H = a.shape[0]
        V = values[l].to(device)  # [B,n_kv,S,d]
        n_kv = V.shape[1]
        V = _repeat_kv(V, H // n_kv)[0]  # [H,S,d]
        Vv = V.index_select(1, vis)  # [H,P,d] value of visual positions

        if think_rows is not None and int(think_rows.numel()) > 0:
            T = int(think_rows.numel())
            think_attn = a.index_select(1, answer_rows.to(device)).index_select(2, think_rows.to(device))  # [H,A,T]
            token_attn = a.index_select(1, think_rows.to(device)).index_select(2, vis)  # [H,T,P]
            agg = think_attn @ token_attn  # [H,A,P]  visual→thinking→answer
        else:
            agg = a.index_select(1, answer_rows.to(device)).index_select(2, vis)  # [H,A,P] direct

        # sv[a,p] = Σ_head agg[h,a,p] * V[h,p,:]  →  [A,P,d_model_per_o_proj]
        sv = (agg.unsqueeze(-1) * Vv.unsqueeze(1)).transpose(0, 1)  # [A,H,P,d]
        sv = sv.permute(0, 2, 1, 3).reshape(A, P, -1)  # [A,P,H*d]
        contrib = text.layers[l].self_attn.o_proj(sv)  # [A,P,d_model]
        acc = contrib if acc is None else acc + contrib

    # norm-scale (official: norm(logits)*||logits|| / ||hidden||), then lm_head.
    acc = text.norm(acc) * acc.norm(dim=-1, keepdim=True)  # [A,P,d_model]
    try:
        hidden = torch.stack([out.hidden_states[-1][0, r] for r in answer_rows.tolist()], 0)  # [A,d]
        acc = acc / hidden.norm(dim=-1, keepdim=True).unsqueeze(1).clamp_min(1e-6)
    except Exception:
        pass  # hidden_states unavailable → skip the per-token normalization
    logits = lm_head(acc)  # [A,P,vocab]
    tgt = target_ids.to(device).view(A, 1, 1).expand(A, P, 1)
    sel = logits.gather(2, tgt).squeeze(-1)  # [A,P] signed contribution of each visual pos
    signed = sel.sum(0)  # [P]  sum over answer span
    return signed.detach().float().cpu().numpy().reshape(h_grid, w_grid)


def salr1_probe(
    gm: G0Model,
    out,
    values: dict[int, torch.Tensor],
    *,
    visual_positions: torch.Tensor,
    grid_hw: tuple[int, int],
    prompt_len: int,
    completion_ids: torch.Tensor,
    bbox: BoxNorm,
    answer_span: CompletionSpan,
    think_span: Optional[CompletionSpan],
    layers: Optional[tuple[int, ...]] = None,
    think_row_mode: str = "state",
    keep_map: bool = False,
) -> Salr1Result:
    """Saliency-R1 holistic (or direct-fallback) map + signed metrics for one sample.

    ``out`` = an eager forward with ``output_attentions=True`` (and ideally
    ``output_hidden_states=True``) over ``prompt+completion``; ``values`` = the
    per-layer value states captured by :func:`_capture_value_states` around it.
    ``answer_span`` / ``think_span`` are completion-token coords.

    ``think_row_mode``: ``"state"`` reads the thinking tokens' own rows (answer
    attends to the existing thinking *states* that carry visual info — matches the
    official code's ``-1`` query slicing intent); ``"predictor"`` shifts them −1 to
    the rows that PREDICTED each thinking token (strict "visual → generating-think"
    causal reading). Kept switchable for the ablation.
    """
    device = next(gm.model.parameters()).device
    comp_len = int(completion_ids.numel())
    n_layers = gm.num_layers
    layers = tuple(range(n_layers)) if layers is None else layers

    a0, a1 = answer_span
    answer_rows = torch.arange(prompt_len + a0 - 1, prompt_len + a1 - 1, device=device).clamp_min(prompt_len - 1)
    target_ids = completion_ids.to(device)[a0:a1]
    if int(answer_rows.numel()) == 0:
        answer_rows = torch.tensor([prompt_len + comp_len - 2], device=device).clamp_min(prompt_len - 1)
        target_ids = completion_ids.to(device)[-1:]

    think_rows = None
    span_mode = "direct"
    if think_span is not None:
        t0, t1 = think_span
        if t1 > t0:
            if think_row_mode == "predictor":
                think_rows = torch.arange(prompt_len + t0 - 1, prompt_len + t1 - 1, device=device).clamp_min(prompt_len - 1)
            else:  # "state": the thinking tokens' own rows
                think_rows = torch.arange(prompt_len + t0, prompt_len + t1, device=device)
            span_mode = "holistic"

    signed = _answer_logit_lens(
        gm, out, values, visual_positions=visual_positions, answer_rows=answer_rows,
        think_rows=think_rows, target_ids=target_ids, grid_hw=grid_hw, layers=layers,
    )
    pos = np.clip(signed, 0.0, None)
    neg = np.clip(-signed, 0.0, None)
    ab = np.abs(signed)
    return Salr1Result(
        span_mode=span_mode,
        pos=map_metrics(pos, bbox),
        neg=map_metrics(neg, bbox),
        abs=map_metrics(ab, bbox),
        mass_gt=mass_in_box(pos, bbox),
        pos_map=pos if keep_map else None,
        signed_map=signed if keep_map else None,
    )


def parse_think_span(text: str, completion_ids: torch.Tensor, tokenizer) -> Optional[CompletionSpan]:
    """Completion-token span of the ``<think>…</think>`` (or ``<reason>…</reason>``)
    content, or None.

    Char→token via incremental decode (same trick as answer_spans), so it lines up
    with the rollout's tokenization. Supports both CoT tag styles since the OPD
    prompt line has used ``<reason>`` as well as ``<think>``.
    """
    import re

    m = None
    for tag in ("think", "reason"):
        m = re.search(rf"<{tag}>\s*(\S.*?)\s*</{tag}>", text, re.DOTALL | re.IGNORECASE)
        if m:
            break
    if not m:
        return None
    ids = [int(x) for x in completion_ids.tolist()]
    n = len(ids)
    if n == 0:
        return None
    c0, c1 = m.start(1), m.end(1)

    def clen(k):
        return len(tokenizer.decode(ids[:k], skip_special_tokens=False, clean_up_tokenization_spaces=False))

    def char_to_tok(cp):
        lo, hi = 1, n
        while lo < hi:
            mid = (lo + hi) // 2
            if clen(mid) > cp:
                hi = mid
            else:
                lo = mid + 1
        return max(0, lo - 1)

    s = char_to_tok(c0)
    e = max(s + 1, min(char_to_tok(max(c0, c1 - 1)) + 1, n))
    return (s, e)


# --------------------------------------------------------------------- self-test
def _selftest() -> None:
    # mass_in_box: a blob fully inside the GT box → mass ≈ 1.
    h, w = 12, 16
    m = np.zeros((h, w)); m[3:6, 4:8] = 1.0
    gt = (4 / w, 3 / h, 8 / w, 6 / h)
    assert abs(mass_in_box(m, gt) - 1.0) < 1e-9, mass_in_box(m, gt)
    # part of the blob outside the box → mass drops below 1 (12 in-box / 18 total).
    m2 = np.zeros((h, w)); m2[3:6, 4:8] = 1.0; m2[3:6, 0:2] = 1.0
    assert abs(mass_in_box(m2, gt) - 12 / 18) < 1e-9, mass_in_box(m2, gt)
    # the y2 height/width fix: a tall box must use the ROW axis. Box spanning rows
    # 0..h, cols 0..w/2; blob in bottom-left → mass should be high (would be wrong
    # if y2 scaled by width).
    tall = np.zeros((h, w)); tall[8:11, 1:3] = 1.0
    gtb = (0.0, 0.0, 0.5, 0.99)
    assert mass_in_box(tall, gtb) > 0.9, mass_in_box(tall, gtb)
    # metrics bundle keys + pointing.
    md = map_metrics(m, gt)
    assert md["pointing"] == 1.0 and set(["mass_gt", "iou_top20", "area_top20", "entropy"]) <= set(md)
    assert _topk_mask(m, 0.2).sum() >= 1
    print("[g0.salr1_probe] self-test passed.")


if __name__ == "__main__":
    _selftest()
