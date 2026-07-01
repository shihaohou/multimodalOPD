"""Minimal visualization functions vendored verbatim from EAGLE.

Source: EAGLE ``visualization/visualization.py`` (MIT License).
Only the three functions required for artifact-only heatmap rendering are kept.
"""

from __future__ import annotations

import cv2
import numpy as np

from baseline.g0.eagle_src.regions import add_value


def gen_cam(image_path, mask):
    """Generate EAGLE's VIRIDIS overlay from a normalized attribution mask."""
    w = mask.shape[1]
    h = mask.shape[0]
    image = cv2.resize(cv2.imread(image_path), (w, h))
    mask = cv2.resize(mask, (int(w / 20), int(h / 20)))
    mask = cv2.resize(mask, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(mask), cv2.COLORMAP_VIRIDIS)
    heatmap = np.float32(heatmap)
    cam = 0.5 * heatmap + 0.5 * np.float32(image)
    return cam.astype(np.uint8), heatmap.astype(np.uint8)


def norm_image(image):
    """Normalize an attribution map exactly as EAGLE does."""
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.0
    return np.uint8(image)
