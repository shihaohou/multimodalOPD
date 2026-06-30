"""Region division + attribution-map builder for the vendored EAGLE core.

* :func:`sub_region_division` — split an image into ``~n_regions`` disjoint masks
  (EAGLE's ``V_set``: a list of ``(H, W, 1)`` 0/1 arrays). Tries, in order,
  OpenCV-contrib SLICO superpixels → scikit-image SLIC → a deterministic
  numpy grid. The grid fallback means region division works on a box with neither
  ``cv2.ximgproc`` nor ``skimage`` installed (the diagnostic still runs; only the
  region shapes change from content-aware superpixels to regular tiles).
* :func:`add_value` — verbatim from EAGLE ``visualization.visualization``: turn the
  greedy region order + sub-modular scores into a normalized ``[H, W, C]``
  attribution heatmap.

EAGLE, arXiv 2509.22496, https://github.com/RuoyuChen10/EAGLE (MIT License).
"""

from __future__ import annotations

import math

import numpy as np


def _labels_to_masks(labels: np.ndarray, n: int) -> list[np.ndarray]:
    out = []
    for i in range(n):
        m = (labels == i)
        if m.any():
            out.append(m[:, :, np.newaxis].astype(np.uint8))
    return out


def _grid_regions(h: int, w: int, n_regions: int) -> list[np.ndarray]:
    """Deterministic regular-tile partition into ~``n_regions`` masks.

    Picks ``cols × rows ≈ n_regions`` with the tile aspect close to the image
    aspect, then assigns each pixel to its tile. Every pixel belongs to exactly
    one mask (the masks tile the image), matching the superpixel contract.
    """
    n_regions = max(1, int(n_regions))
    cols = max(1, int(round(math.sqrt(n_regions * w / max(1, h)))))
    rows = max(1, int(round(n_regions / cols)))
    col_edges = np.linspace(0, w, cols + 1).astype(int)
    row_edges = np.linspace(0, h, rows + 1).astype(int)
    masks = []
    for r in range(rows):
        for c in range(cols):
            m = np.zeros((h, w, 1), dtype=np.uint8)
            m[row_edges[r]:row_edges[r + 1], col_edges[c]:col_edges[c + 1], 0] = 1
            if m.any():
                masks.append(m)
    return masks


def sub_region_division(image: np.ndarray, n_regions: int = 49, mode: str = "auto") -> list[np.ndarray]:
    """Split ``image`` (H×W×3, cv2/BGR uint8) into ~``n_regions`` disjoint masks.

    ``mode``: ``"auto"`` (SLICO→SLIC→grid), ``"slico"``, ``"slic"`` or ``"grid"``.
    Returns a list of ``(H, W, 1)`` uint8 masks (EAGLE's ``V_set``).
    """
    h, w = image.shape[:2]
    region_size = max(8, int((h * w / max(1, n_regions)) ** 0.5))

    if mode in ("auto", "slico"):
        try:
            import cv2  # opencv-contrib (cv2.ximgproc)

            slic = cv2.ximgproc.createSuperpixelSLIC(image, region_size=region_size, ruler=20.0)
            slic.iterate(20)
            labels = slic.getLabels()
            n = slic.getNumberOfSuperpixels()
            masks = _labels_to_masks(labels, n)
            if masks:
                return masks
        except Exception:
            if mode == "slico":
                raise

    if mode in ("auto", "slic"):
        try:
            from skimage.segmentation import slic as sk_slic

            rgb = image[:, :, ::-1] if image.ndim == 3 else image  # BGR→RGB for skimage
            labels = sk_slic(rgb, n_segments=n_regions, compactness=10.0, start_label=0)
            masks = _labels_to_masks(labels, int(labels.max()) + 1)
            if masks:
                return masks
        except Exception:
            if mode == "slic":
                raise

    return _grid_regions(h, w, n_regions)


def add_value(S_set, json_file):
    """Verbatim EAGLE ``visualization.add_value``: region order → attribution map.

    ``S_set`` is the greedy region order (array of ``(H, W, 1)`` masks);
    ``json_file`` carries ``smdl_score`` / ``org_score`` / ``baseline_score``.
    Returns ``(attribution_map[H, W, C] in [0,1], per-region values)``.
    """
    single_mask = np.zeros_like(S_set[0])
    single_mask = single_mask.astype(np.float16)

    value_list_1 = np.array(json_file["smdl_score"])
    value_list_2 = np.array(
        [np.mean(1 - np.array(json_file["org_score"]) + np.array(json_file["baseline_score"]))]
        + json_file["smdl_score"][:-1]
    )
    value_list = value_list_1 - value_list_2

    values = []
    value = 0
    i = 0
    for smdl_single_mask, smdl_value in zip(S_set, value_list):
        value = value - abs(smdl_value)
        single_mask[smdl_single_mask == 1] = value
        values.append(value)
        i += 1
    attribution_map = single_mask - single_mask.min()
    attribution_map = attribution_map / (attribution_map.max() + 1e-8)

    return attribution_map, np.array(values)


def _selftest() -> None:
    # grid regions tile the image exactly once.
    h, w = 30, 40
    masks = _grid_regions(h, w, 12)
    stacked = np.concatenate([m for m in masks], axis=2).sum(axis=2)
    assert stacked.min() == 1 and stacked.max() == 1, (stacked.min(), stacked.max())
    assert 6 <= len(masks) <= 20, len(masks)
    # sub_region_division falls back to grid here (no cv2/skimage) and is non-empty.
    img = np.zeros((h, w, 3), dtype=np.uint8)
    v = sub_region_division(img, n_regions=12)
    assert len(v) >= 4 and v[0].shape == (h, w, 1), (len(v), v[0].shape)
    # add_value: fabricate a 3-region greedy result → normalized map in [0,1].
    S = np.array([m for m in masks[:3]])
    jf = {"smdl_score": [0.9, 0.5, 0.2], "org_score": [0.8], "baseline_score": [0.1]}
    amap, vals = add_value(S, jf)
    assert amap.shape == (h, w, 1) and 0.0 <= float(amap.min()) and float(amap.max()) <= 1.0 + 1e-6
    assert len(vals) == 3
    print("[g0.eagle_src.regions] self-test passed.")


if __name__ == "__main__":
    _selftest()
