"""Pure-geometry helpers shared by the LH and GLIMPSE probes.

Everything here is numpy / plain-python (no torch, no model) so it is fast,
unit-testable on CPU, and the single source of truth for "how a heatmap over the
merged patch grid becomes a box and an IoU against the normalized GT box".

Conventions (matching the rest of the repo):
  * GT boxes are normalized ``(x1, y1, x2, y2)`` in [0, 1], top-left origin
    (``baseline.probe.saliency_data.parse_bbox_norm``).
  * A saliency / attention map over the image is a 2-D array ``[H_grid, W_grid]``
    laid out row-major: axis 0 = rows (top→bottom, the ``y`` / height axis),
    axis 1 = cols (left→right, the ``x`` / width axis). This is exactly the order
    Qwen emits the merged image-placeholder tokens, so ``map.reshape(H, W)`` of a
    visual-token vector is correct.
  * Grid boxes are ``(c1, r1, c2, r2)`` with ``c`` = column index, ``r`` = row
    index, both **inclusive** of the end cell (a single cell is ``(c,r,c,r)``).

Run ``python -m baseline.g0.metrics`` for a self-test.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

BoxNorm = tuple[float, float, float, float]
GridBox = tuple[int, int, int, int]  # (c1, r1, c2, r2), inclusive


# --------------------------------------------------------------------- smoothing
def gaussian_smooth(map_2d: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-blur a 2-D map. Uses scipy if present, else a numpy fallback.

    ``sigma <= 0`` is a no-op. The fallback is a separable Gaussian via reflect-
    padded 1-D convolutions, which matches ``scipy.ndimage.gaussian_filter``
    closely enough for thresholding (the only downstream use).
    """
    if sigma is None or sigma <= 0:
        return map_2d.astype(np.float64, copy=False)
    arr = map_2d.astype(np.float64, copy=False)
    try:
        from scipy.ndimage import gaussian_filter  # type: ignore

        return gaussian_filter(arr, sigma=sigma)
    except Exception:
        pass
    # Separable numpy fallback.
    radius = max(1, int(round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x ** 2) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()

    def conv1d(a: np.ndarray, axis: int) -> np.ndarray:
        a = np.apply_along_axis(
            lambda m: np.convolve(np.pad(m, radius, mode="reflect"), kernel, mode="valid"),
            axis,
            a,
        )
        return a

    return conv1d(conv1d(arr, 0), 1)


# ------------------------------------------------------------------ thresholding
def binarize_mean_relu(map_2d: np.ndarray) -> np.ndarray:
    """``map > mean(map)`` boolean mask — the LocalizationHeads ``mean_relu`` rule.

    Robust and parameter-free; this is the default both for assembling the LH box
    and for thresholding the GLIMPSE visual map.
    """
    arr = np.asarray(map_2d, dtype=np.float64)
    if arr.size == 0:
        return np.zeros_like(arr, dtype=bool)
    return arr > float(arr.mean())


def binarize_top_frac(map_2d: np.ndarray, frac: float) -> np.ndarray:
    """Keep the cells holding the top ``frac`` of total (positive) energy.

    A scale-free alternative to ``mean_relu`` that is comparable across maps with
    different sparsity. ``frac`` in (0, 1].
    """
    arr = np.asarray(map_2d, dtype=np.float64)
    pos = np.clip(arr, 0.0, None)
    total = float(pos.sum())
    if total <= 0 or arr.size == 0:
        return np.zeros_like(arr, dtype=bool)
    order = np.argsort(pos, axis=None)[::-1]
    csum = np.cumsum(pos.flatten()[order])
    keep_n = int(np.searchsorted(csum, frac * total)) + 1
    mask_flat = np.zeros(arr.size, dtype=bool)
    mask_flat[order[:keep_n]] = True
    return mask_flat.reshape(arr.shape)


# --------------------------------------------------------------------- box <-> grid
def bbox_from_mask(mask: np.ndarray) -> Optional[GridBox]:
    """Tight envelope (c1, r1, c2, r2) of all True cells, or None if empty.

    Matches LocalizationHeads' ``bbox_from_mask``: global min/max of the
    above-threshold cells (no connected-components / largest-blob selection).
    """
    rows, cols = np.where(np.asarray(mask, dtype=bool))
    if rows.size == 0:
        return None
    return int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())


def grid_box_to_norm(box: GridBox, h_grid: int, w_grid: int) -> BoxNorm:
    """Inclusive grid box (c1,r1,c2,r2) → normalized (x1,y1,x2,y2) in [0,1].

    Cell ``c`` spans ``[c/W, (c+1)/W)`` horizontally, so the inclusive box
    ``[c1, c2]`` spans ``[c1/W, (c2+1)/W]``.
    """
    c1, r1, c2, r2 = box
    x1 = c1 / w_grid
    y1 = r1 / h_grid
    x2 = (c2 + 1) / w_grid
    y2 = (r2 + 1) / h_grid
    return (float(x1), float(y1), float(min(1.0, x2)), float(min(1.0, y2)))


def gt_box_to_grid_mask(bbox: BoxNorm, h_grid: int, w_grid: int) -> np.ndarray:
    """Rasterize a normalized GT box onto the patch grid (bool ``[H_grid,W_grid]``).

    A cell is GT if it OVERLAPS the box: the left/top edge floors, the right/bottom
    edge **ceils** (so a box touching any part of a cell marks it), and the box
    always covers ≥1 cell. ``round`` on the far edge would under-cover small boxes
    on a coarse grid and depress IoU/energy, so we use ceil.
    """
    x1, y1, x2, y2 = bbox
    c1 = max(0, min(w_grid - 1, math.floor(x1 * w_grid)))
    c2 = min(w_grid, max(c1 + 1, math.ceil(x2 * w_grid)))
    r1 = max(0, min(h_grid - 1, math.floor(y1 * h_grid)))
    r2 = min(h_grid, max(r1 + 1, math.ceil(y2 * h_grid)))
    mask = np.zeros((h_grid, w_grid), dtype=bool)
    mask[r1:r2, c1:c2] = True
    return mask


# ------------------------------------------------------------------------- IoUs
def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two boolean grid masks of the same shape."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return inter / union if union else 0.0


def bbox_iou_norm(a: Optional[BoxNorm], b: Optional[BoxNorm]) -> float:
    """IoU of two normalized ``(x1,y1,x2,y2)`` boxes (0 if either is None)."""
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


# ------------------------------------------------------- pointing / energy probes
def pointing_hit(map_2d: np.ndarray, gt_mask: np.ndarray) -> bool:
    """Is the argmax cell of the map inside the GT grid mask? (pointing game)"""
    arr = np.asarray(map_2d, dtype=np.float64)
    if arr.size == 0:
        return False
    flat = int(np.argmax(arr))
    r, c = divmod(flat, arr.shape[1])
    return bool(gt_mask[r, c])


def energy_in_box(map_2d: np.ndarray, gt_mask: np.ndarray) -> Optional[float]:
    """Fraction of positive saliency mass that lands inside the GT mask.

    Returns None if the map has no positive mass (undefined).
    """
    pos = np.clip(np.asarray(map_2d, dtype=np.float64), 0.0, None)
    total = float(pos.sum())
    if total <= 0:
        return None
    return float(pos[np.asarray(gt_mask, dtype=bool)].sum()) / total


# ---------------------------------------------------- map → predicted box + IoU
def map_to_pred_box(
    map_2d: np.ndarray,
    *,
    sigma: float = 1.0,
    threshold: str = "mean",
    top_frac: float = 0.25,
) -> Optional[GridBox]:
    """Smooth → threshold → tight bbox. The LH assembly applied to any 2-D map."""
    smoothed = gaussian_smooth(map_2d, sigma)
    if threshold == "top_frac":
        mask = binarize_top_frac(smoothed, top_frac)
    else:
        mask = binarize_mean_relu(smoothed)
    return bbox_from_mask(mask)


def iou_map_vs_gt(
    map_2d: np.ndarray,
    bbox: BoxNorm,
    *,
    sigma: float = 1.0,
    threshold: str = "mean",
    top_frac: float = 0.25,
) -> dict[str, float]:
    """Bundle of IoU flavors for a saliency/attention map vs a normalized GT box.

    Returns ``mask_iou`` (thresholded grid mask vs GT grid mask — the robust,
    IoUStation-style number), ``bbox_iou`` (envelope box vs GT box, normalized),
    ``pointing`` (argmax-in-box) and ``energy`` (positive mass in box).
    """
    h_grid, w_grid = map_2d.shape
    gt_mask = gt_box_to_grid_mask(bbox, h_grid, w_grid)
    smoothed = gaussian_smooth(map_2d, sigma)
    if threshold == "top_frac":
        pred_mask = binarize_top_frac(smoothed, top_frac)
    else:
        pred_mask = binarize_mean_relu(smoothed)
    pred_box = bbox_from_mask(pred_mask)
    pred_norm = grid_box_to_norm(pred_box, h_grid, w_grid) if pred_box else None
    energy = energy_in_box(map_2d, gt_mask)
    return {
        "mask_iou": mask_iou(pred_mask, gt_mask),
        "bbox_iou": bbox_iou_norm(pred_norm, bbox),
        "pointing": float(pointing_hit(map_2d, gt_mask)),
        "energy": float(energy) if energy is not None else float("nan"),
    }


def _selftest() -> None:
    # A blob in the top-left quadrant; GT box covering the top-left quadrant.
    h, w = 16, 24
    m = np.zeros((h, w), dtype=np.float64)
    m[2:6, 3:8] = 1.0
    gt = (3 / w, 2 / h, 8 / w, 6 / h)  # exactly the blob cells
    res = iou_map_vs_gt(m, gt, sigma=0.0)
    assert res["pointing"] == 1.0, res
    assert res["energy"] == 1.0, res
    assert res["mask_iou"] > 0.4, res  # smoothing off, threshold mean → the blob
    # GT mask round-trips.
    gm = gt_box_to_grid_mask(gt, h, w)
    assert gm[2:6, 3:8].all() and gm.sum() == 4 * 5, gm.sum()
    # bbox round-trip.
    assert bbox_from_mask(gm) == (3, 2, 7, 5)
    assert abs(bbox_iou_norm(gt, gt) - 1.0) < 1e-9
    # disjoint boxes → 0.
    assert bbox_iou_norm((0, 0, 0.1, 0.1), (0.9, 0.9, 1.0, 1.0)) == 0.0
    # top-frac at 1.0 keeps exactly the positive blob (and nothing else).
    tf = binarize_top_frac(m, 1.0)
    assert tf[2:6, 3:8].all() and tf.sum() == 4 * 5, tf.sum()
    print("[g0.metrics] self-test passed.")


if __name__ == "__main__":
    _selftest()
