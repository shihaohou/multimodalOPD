"""Pure pixel-space image edits for the evidence-reliance probe.

Everything keys off a normalized ``(x1,y1,x2,y2)`` box in [0, 1]; converting to
pixels is the only image-size-dependent step. No model / tokenizer coupling, so
these are the most robust basis for the go/no-go decision (Stage 0 / 1b). The
patch-token mapping needed for attention (1c) lives elsewhere.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

BoxNorm = tuple[float, float, float, float]
BoxPx = tuple[int, int, int, int]


def norm_to_px(box: BoxNorm, width: int, height: int) -> BoxPx:
    """Normalized (x1,y1,x2,y2) -> pixel box, clamped in-bounds, min 1px."""
    x1, y1, x2, y2 = box
    x1p = int(round(x1 * width))
    y1p = int(round(y1 * height))
    x2p = int(round(x2 * width))
    y2p = int(round(y2 * height))
    x1p = max(0, min(x1p, width - 1))
    y1p = max(0, min(y1p, height - 1))
    x2p = max(x1p + 1, min(x2p, width))
    y2p = max(y1p + 1, min(y2p, height))
    return (x1p, y1p, x2p, y2p)


def _fill_color(image: Image.Image, mode: str) -> tuple[int, int, int]:
    if mode == "black":
        return (0, 0, 0)
    if mode == "mean":
        arr = np.asarray(image.convert("RGB")).reshape(-1, 3).mean(axis=0)
        return tuple(int(round(float(v))) for v in arr)
    return (128, 128, 128)  # "gray" default


def mask_box(image: Image.Image, box_px: BoxPx, *, fill: str = "gray") -> Image.Image:
    """Return a copy with ``box_px`` occluded.

    ``fill`` in {gray, black, mean, blur}. ``blur`` keeps low-frequency layout but
    destroys fine detail (a gentler, less-OOD occlusion than a hard fill).
    """
    out = image.convert("RGB").copy()
    x1, y1, x2, y2 = box_px
    if fill == "blur":
        region = out.crop((x1, y1, x2, y2))
        radius = max(4, int(0.5 * max(x2 - x1, y2 - y1)))
        out.paste(region.filter(ImageFilter.GaussianBlur(radius)), (x1, y1))
        return out
    ImageDraw.Draw(out).rectangle((x1, y1, x2, y2), fill=_fill_color(out, fill))
    return out


def iou(a: BoxPx, b: BoxPx) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter)


def random_box_same_shape(
    evidence_px: BoxPx,
    width: int,
    height: int,
    rng: np.random.Generator,
    *,
    max_iou: float = 0.2,
    tries: int = 100,
) -> BoxPx:
    """A box with the SAME (w, h) as the evidence box at a random in-bounds spot.

    Same shape => area- and aspect-matched, so only *location* differs: it is the
    control that cancels the "N% of the image is occluded" OOD artifact. Rejects
    placements that overlap the evidence box (IoU > ``max_iou``); falls back to the
    lowest-IoU candidate found.
    """
    x1, y1, x2, y2 = evidence_px
    w, h = x2 - x1, y2 - y1
    if w >= width or h >= height:
        return evidence_px  # box ~ whole image; nowhere disjoint to move it
    best: BoxPx | None = None
    best_iou = 2.0
    for _ in range(tries):
        nx = int(rng.integers(0, width - w + 1))
        ny = int(rng.integers(0, height - h + 1))
        cand = (nx, ny, nx + w, ny + h)
        cand_iou = iou(cand, evidence_px)
        if cand_iou < best_iou:
            best, best_iou = cand, cand_iou
        if cand_iou <= max_iou:
            return cand
    return best if best is not None else evidence_px


def crop_box(image: Image.Image, box_norm: BoxNorm, *, pad_frac: float = 0.0) -> Image.Image:
    """Crop to the evidence box, expanded by ``pad_frac`` of the box's own size."""
    width, height = image.size
    x1, y1, x2, y2 = box_norm
    bw, bh = (x2 - x1), (y2 - y1)
    expanded = (x1 - pad_frac * bw, y1 - pad_frac * bh, x2 + pad_frac * bw, y2 + pad_frac * bh)
    return image.convert("RGB").crop(norm_to_px(expanded, width, height))


# --------------------------------------------------------------------- overlays
def _outline(image: Image.Image, box_px: BoxPx, color, width: int = 3) -> Image.Image:
    out = image.convert("RGB").copy()
    ImageDraw.Draw(out).rectangle(box_px, outline=color, width=width)
    return out


def _label(panel: Image.Image, text: str) -> Image.Image:
    """Add a caption strip above a panel."""
    strip_h = 18
    out = Image.new("RGB", (panel.width, panel.height + strip_h), (255, 255, 255))
    out.paste(panel, (0, strip_h))
    ImageDraw.Draw(out).text((2, 3), text, fill=(0, 0, 0))
    return out


def build_sanity_sheet(
    image: Image.Image,
    box_norm: BoxNorm,
    rng: np.random.Generator,
    *,
    fill: str = "gray",
    pads: tuple[float, ...] = (0.0, 0.1),
    panel_h: int = 220,
) -> Image.Image:
    """Horizontal montage to eyeball bbox->image alignment for ONE sample.

    Panels: full+evidence outline, full+random outline, masked-evidence,
    masked-random, crop@each pad. Save these and *look* before trusting numbers.
    """
    image = image.convert("RGB")
    w, h = image.size
    ev_px = norm_to_px(box_norm, w, h)
    rand_px = random_box_same_shape(ev_px, w, h, rng)
    panels = [
        _label(_outline(image, ev_px, (255, 0, 0)), "full +evidence"),
        _label(_outline(image, rand_px, (0, 128, 255)), "full +random"),
        _label(mask_box(image, ev_px, fill=fill), f"mask-evidence ({fill})"),
        _label(mask_box(image, rand_px, fill=fill), f"mask-random ({fill})"),
    ]
    for pad in pads:
        panels.append(_label(crop_box(image, box_norm, pad_frac=pad), f"crop pad={pad}"))

    def _resize(panel: Image.Image) -> Image.Image:
        scale = panel_h / panel.height
        return panel.resize((max(1, int(panel.width * scale)), panel_h))

    panels = [_resize(p) for p in panels]
    gap = 6
    total_w = sum(p.width for p in panels) + gap * (len(panels) - 1)
    sheet = Image.new("RGB", (total_w, panel_h), (255, 255, 255))
    x = 0
    for p in panels:
        sheet.paste(p, (x, 0))
        x += p.width + gap
    return sheet
