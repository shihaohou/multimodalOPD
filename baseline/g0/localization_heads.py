"""LocalizationHeads, ported from LLaVA to Qwen2.5-VL / Qwen3-VL — the "looking" probe.

Reference: *Your LVLM Only Needs A Few Attention Heads For Visual Grounding*
(arXiv 2503.06287) and the local overhaul at
``/Users/houshihao/project/code/LocalizationHeads-main``. The idea: a handful of
attention heads carry most of an LVLM's spatial grounding, and reading where they
attend recovers the referent's box without any extra training.

What we keep from the reference, and what we change for Qwen:

* **Attention signal** — the first-generation-step query attending back to the
  image tokens, ``[L, H, 1, V]``. We take it from the row of the last prompt
  token (the query that predicts the first response token) in a single eager
  forward. Same quantity as the reference's forward-mode capture.
* **Visual slice → grid** — the reference assumes LLaVA's fixed square 24×24=576
  grid (``P=sqrt(V)``). Qwen has *dynamic-resolution, non-square* grids, so we
  reshape the visual slice to ``(grid_h//merge, grid_w//merge)`` read from
  ``image_grid_thw`` (never ``sqrt``). Image tokens are found by
  ``input_ids == image_token_id`` (Qwen's real ``<|image_pad|>`` run), not the
  LLaVA −200 splice.
* **Head selection** — the reference's *live* selector uses an attention-sum
  elbow + spatial-entropy (unsupervised). Because saliency-r1-8k gives us GT
  boxes, we instead use the paper's **IoU calibration** (its ``IoUStation``):
  rank heads by mean per-head IoU against GT on a calibration split, take the
  top-k. This is what discovers a model's localization heads — and we do it
  **separately for the 8B teacher and the 2B student** (no L14-H24 assumption).
* **Assembly → box** — selected heads' maps are Gaussian-smoothed, summed,
  thresholded at the mean, and the tight envelope of the above-threshold cells is
  the predicted box (reference ``combine_heads`` / ``bbox_from_mask``).

All per-head scoring is *scalar* (IoU, attention-sum), so it averages cleanly
across samples even though every sample has a different grid size.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from baseline.g0 import metrics
from baseline.g0.engine import (
    BoxNorm,
    G0Model,
    build_inputs,
    nograd_attention_forward,
    visual_grid,
)


# ----------------------------------------------------------- attention → maps
def head_visual_maps(
    out_attentions,
    q_row: int,
    visual_positions: torch.Tensor,
    grid_hw: tuple[int, int],
) -> np.ndarray:
    """First-gen-step per-head image maps: ``[L, H, H_grid, W_grid]`` (cpu float).

    ``out_attentions`` is the ``output_attentions`` tuple (each ``[B,H,S,S]``);
    ``q_row`` is the query position whose attention we read (the last prompt
    token). We slice that row to the visual keys and reshape to the patch grid.
    """
    h_grid, w_grid = grid_hw
    layers = []
    for a in out_attentions:
        row = a[0, :, q_row, :]  # [H, S]
        vis = row.index_select(1, visual_positions.to(row.device))  # [H, P]
        layers.append(vis.reshape(vis.shape[0], h_grid, w_grid).float().detach().cpu())
    return torch.stack(layers, 0).numpy()  # [L, H, H_grid, W_grid]


def per_head_scores(
    maps: np.ndarray, bbox: BoxNorm, *, smooth_sigma: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Per-head (IoU vs GT, attention-sum) for one sample. Both ``[L, H]``.

    IoU is the IoUStation rule: binarize the head map at its own mean, IoU vs the
    GT grid mask. ``attn_sum`` is the head's total attention mass on the image.
    """
    n_layers, n_heads, h_grid, w_grid = maps.shape
    gt_mask = metrics.gt_box_to_grid_mask(bbox, h_grid, w_grid)
    iou = np.zeros((n_layers, n_heads), dtype=np.float64)
    attn_sum = np.zeros((n_layers, n_heads), dtype=np.float64)
    for l in range(n_layers):
        for h in range(n_heads):
            m = maps[l, h]
            attn_sum[l, h] = float(m.sum())
            sm = metrics.gaussian_smooth(m, smooth_sigma) if smooth_sigma > 0 else m
            iou[l, h] = metrics.mask_iou(metrics.binarize_mean_relu(sm), gt_mask)
    return iou, attn_sum


# ------------------------------------------------------------------ calibration
@dataclass
class HeadStats:
    """Per-head calibration result for one model under its natural condition."""

    model_name: str
    condition: str
    num_layers: int
    num_heads: int
    n_samples: int
    mean_iou: np.ndarray  # [L, H]
    mean_attn_sum: np.ndarray  # [L, H]
    selection_freq: np.ndarray  # [L, H] — fraction of samples with (l,h) in per-sample top-k IoU
    top_k: int
    min_layer: int
    selected_heads: list[tuple[int, int]] = field(default_factory=list)  # top-k by mean IoU
    selected_by_attn: list[tuple[int, int]] = field(default_factory=list)  # top-k by mean attn-sum

    def to_json(self) -> dict:
        return {
            "model_name": self.model_name,
            "condition": self.condition,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "n_samples": self.n_samples,
            "top_k": self.top_k,
            "min_layer": self.min_layer,
            "selected_heads": [list(h) for h in self.selected_heads],
            "selected_by_attn": [list(h) for h in self.selected_by_attn],
            "best_head_by_mean_iou": list(self.selected_heads[0]) if self.selected_heads else None,
            "best_mean_iou": float(self.mean_iou.max()) if self.mean_iou.size else 0.0,
            # Compact per-head tables (rounded) so the JSON stays readable.
            "mean_iou": np.round(self.mean_iou, 4).tolist(),
            "mean_attn_sum": np.round(self.mean_attn_sum, 5).tolist(),
            "selection_freq": np.round(self.selection_freq, 4).tolist(),
        }


def _topk_heads(score: np.ndarray, k: int, min_layer: int) -> list[tuple[int, int]]:
    """(layer, head) of the top-k entries of a ``[L,H]`` score, layers ≥ min_layer."""
    masked = score.copy()
    if min_layer > 0:
        masked[:min_layer, :] = -np.inf
    flat = np.argsort(masked, axis=None)[::-1][:k]
    return [(int(i // score.shape[1]), int(i % score.shape[1])) for i in flat]


def calibrate_heads(
    gm: G0Model,
    samples,
    *,
    hint: bool = False,
    top_k: int = 3,
    min_layer: int = 2,
    per_sample_topk: int = 3,
    smooth_sigma: float = 0.0,
    progress: bool = True,
) -> HeadStats:
    """Discover a model's localization heads by per-head IoU vs GT over ``samples``.

    Uses the *natural* condition for the model (no hint) unless ``hint=True``. The
    first-gen-step attention is the last prompt row of a causal forward, so no
    generation is needed — we forward the prompt and read row ``prompt_len-1``.
    Aggregates scalar per-head IoU / attention-sum means + a per-sample top-k
    selection frequency, then picks the top-``top_k`` heads by mean IoU.
    """
    n_layers, n_heads = gm.num_layers, gm.num_heads
    sum_iou = np.zeros((n_layers, n_heads), dtype=np.float64)
    sum_attn = np.zeros((n_layers, n_heads), dtype=np.float64)
    freq = np.zeros((n_layers, n_heads), dtype=np.float64)
    n_used = 0
    for idx, s in enumerate(samples):
        if progress and idx % 25 == 0:
            print(f"[lh-calib:{gm.name}] {idx}/{len(samples)}")
        try:
            inputs = build_inputs(gm, s.image, s.problem, hint_bbox=s.bbox_norm if hint else None)
            prompt_len = int(inputs["input_ids"].shape[1])
            full_ids = inputs["input_ids"][0]
            visual_positions, grid_hw = visual_grid(gm, full_ids, inputs["image_grid_thw"])
            out = nograd_attention_forward(gm, inputs, full_ids)
            maps = head_visual_maps(out.attentions, prompt_len - 1, visual_positions, grid_hw)
            iou, attn_sum = per_head_scores(maps, s.bbox_norm, smooth_sigma=smooth_sigma)
        except Exception as exc:  # one bad sample shouldn't sink calibration
            print(f"[lh-calib:{gm.name}] skip sample {getattr(s, 'sample_id', idx)}: {exc}")
            continue
        sum_iou += iou
        sum_attn += attn_sum
        # per-sample top-k by IoU (layers ≥ min_layer) → selection frequency.
        for (l, h) in _topk_heads(iou, per_sample_topk, min_layer):
            freq[l, h] += 1.0
        n_used += 1
        del out

    n = max(1, n_used)
    mean_iou = sum_iou / n
    mean_attn = sum_attn / n
    selection_freq = freq / n
    stats = HeadStats(
        model_name=gm.name,
        condition="hint" if hint else "natural",
        num_layers=n_layers,
        num_heads=n_heads,
        n_samples=n_used,
        mean_iou=mean_iou,
        mean_attn_sum=mean_attn,
        selection_freq=selection_freq,
        top_k=top_k,
        min_layer=min_layer,
        selected_heads=_topk_heads(mean_iou, top_k, min_layer),
        selected_by_attn=_topk_heads(mean_attn, top_k, min_layer),
    )
    print(
        f"[lh-calib:{gm.name}] n={n_used} best_mean_IoU={mean_iou.max():.3f} "
        f"heads(by IoU)={stats.selected_heads}"
    )
    return stats


# --------------------------------------------------------------- apply / measure
@dataclass
class LHResult:
    iou_lh: float  # assembled top-k mask IoU (the headline "looking" number)
    bbox_iou: float  # assembled top-k normalized-box IoU
    pointing: float  # assembled-map argmax in GT box
    energy: float  # assembled positive mass in GT box
    best_single_iou: float  # max per-head mask IoU this sample (any head — upper bound)
    pred_box_norm: Optional[BoxNorm]
    assembled_map: Optional[np.ndarray] = None  # [H_grid, W_grid], for viz (optional)


def localize_from_maps(
    maps: np.ndarray,
    bbox: BoxNorm,
    selected_heads: list[tuple[int, int]],
    *,
    sigma: float = 1.0,
    keep_map: bool = False,
) -> LHResult:
    """Assemble the selected heads into a box and score it against the GT box.

    Also reports ``best_single_iou`` = the best per-head IoU over *all* heads this
    sample (an upper bound on what one head could do, for the head-usability
    analysis), independent of which heads were selected during calibration.
    """
    n_layers, n_heads, h_grid, w_grid = maps.shape
    gt_mask = metrics.gt_box_to_grid_mask(bbox, h_grid, w_grid)

    assembled = np.zeros((h_grid, w_grid), dtype=np.float64)
    for (l, h) in selected_heads:
        if 0 <= l < n_layers and 0 <= h < n_heads:
            assembled += metrics.gaussian_smooth(maps[l, h], sigma)
    res = metrics.iou_map_vs_gt(assembled, bbox, sigma=0.0)  # already smoothed
    pred_mask = metrics.binarize_mean_relu(assembled)
    pred_box = metrics.bbox_from_mask(pred_mask)
    pred_norm = metrics.grid_box_to_norm(pred_box, h_grid, w_grid) if pred_box else None

    best_single = 0.0
    for l in range(n_layers):
        for h in range(n_heads):
            best_single = max(
                best_single,
                metrics.mask_iou(metrics.binarize_mean_relu(maps[l, h]), gt_mask),
            )
    return LHResult(
        iou_lh=float(res["mask_iou"]),
        bbox_iou=float(res["bbox_iou"]),
        pointing=float(res["pointing"]),
        energy=float(res["energy"]) if np.isfinite(res["energy"]) else 0.0,
        best_single_iou=float(best_single),
        pred_box_norm=pred_norm,
        assembled_map=assembled if keep_map else None,
    )
