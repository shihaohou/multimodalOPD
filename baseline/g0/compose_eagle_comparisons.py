"""Compose original/GT and model-by-prompt EAGLE panels into comparison sheets."""

from __future__ import annotations

import argparse
import glob
import json
import os

from PIL import Image, ImageDraw, ImageFont

from baseline.g0.viz_eagle_g0 import _load_config, load_selected_samples
from baseline.probe.saliency_data import canon_subset


def _font(size: int, bold: bool = False):
    names = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    )
    preferred = names[::2] if bold else names[1::2]
    for path in preferred:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _fit(image: Image.Image, width: int, height: int) -> Image.Image:
    image = image.convert("RGB")
    scale = min(width / max(1, image.width), height / max(1, image.height))
    resized = image.resize(
        (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale)))),
        Image.Resampling.LANCZOS,
    )
    out = Image.new("RGB", (width, height), "white")
    out.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return out


def _bbox_image(image: Image.Image, bbox) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    x0, y0, x1, y1 = [float(v) for v in bbox]
    box = (
        int(round(x0 * out.width)),
        int(round(y0 * out.height)),
        int(round(x1 * out.width)),
        int(round(y1 * out.height)),
    )
    line_width = max(3, int(round(min(out.size) / 120)))
    draw.rectangle(box, outline=(220, 38, 38), width=line_width)
    return out


def _paste_panel(canvas, image, x, y, width, height, label, label_font):
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((x, y, x + width, y + height + 36), fill=(246, 247, 249), outline=(205, 208, 214), width=1)
    draw.text((x + 12, y + 7), label, font=label_font, fill=(24, 27, 32))
    canvas.paste(_fit(image, width, height), (x, y + 36))


def _missing_panel(width: int, height: int, text: str) -> Image.Image:
    image = Image.new("RGB", (width, height), (242, 243, 245))
    draw = ImageDraw.Draw(image)
    draw.text((20, height // 2 - 12), text, font=_font(22), fill=(160, 35, 35))
    return image


def _panel_path(output_root: str, model_name: str, subset: str, sample_id: str, condition: str) -> str | None:
    panel_dir = os.path.join(output_root, model_name, "viz_panels")
    exact = os.path.join(panel_dir, f"{subset}_{sample_id}_{condition}_sentence_span.png")
    if os.path.exists(exact):
        return exact
    matches = sorted(glob.glob(os.path.join(panel_dir, f"*_{sample_id}_{condition}_sentence_span.png")))
    return matches[0] if matches else None


def compose_one(sample, category: str, model_names: list[str], conditions: list[str], output_root: str) -> Image.Image:
    cell_w, cell_h = 520, 390
    gutter, margin, gap = 170, 24, 14
    title_h, label_h = 64, 36
    top_h = cell_h + label_h
    row_h = cell_h + label_h
    width = gutter + margin * 2 + cell_w * len(model_names) + gap * (len(model_names) - 1)
    height = title_h + top_h + gap + len(conditions) * (row_h + gap) + margin
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (margin, 15),
        f"{category} | {sample.subset}/{sample.sample_id}",
        font=_font(30, bold=True),
        fill=(20, 23, 28),
    )

    content_x = gutter + margin
    top_block_w = (cell_w * len(model_names) + gap * (len(model_names) - 1) - gap) // 2
    original = sample.image.convert("RGB")
    _paste_panel(canvas, original, content_x, title_h, top_block_w, cell_h, "Original image", _font(23, bold=True))
    _paste_panel(
        canvas,
        _bbox_image(original, sample.bbox_norm),
        content_x + top_block_w + gap,
        title_h,
        top_block_w,
        cell_h,
        "Ground-truth bounding box",
        _font(23, bold=True),
    )

    start_y = title_h + top_h + gap
    for row, condition in enumerate(conditions):
        y = start_y + row * (row_h + gap)
        label = condition.replace("_", " ")
        draw.text((margin, y + row_h // 2 - 15), label, font=_font(23, bold=True), fill=(40, 43, 49))
        for col, model_name in enumerate(model_names):
            x = content_x + col * (cell_w + gap)
            path = _panel_path(output_root, model_name, sample.subset, str(sample.sample_id), condition)
            panel = Image.open(path).convert("RGB") if path else _missing_panel(cell_w, cell_h, "Missing heatmap")
            _paste_panel(canvas, panel, x, y, cell_w, cell_h, model_name, _font(20, bold=True))
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-manifest", required=True)
    parser.add_argument("--reference-run-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--conditions", default="plain,hint,hidden_hint")
    parser.add_argument("--output-subdir", default="comparisons")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.case_manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)
    categories = manifest.get("categories", {})
    if not categories:
        raise SystemExit(f"[eagle.compose] no categories in {args.case_manifest}")

    keys = {
        (str(row["subset"]), str(row["sample_id"]))
        for rows in categories.values()
        for row in rows
    }
    samples = load_selected_samples(_load_config(args.reference_run_dir), keys)
    model_names = [os.path.basename(os.path.normpath(path)) for path in args.run_dirs]
    conditions = [value.strip() for value in args.conditions.split(",") if value.strip()]
    comparison_root = os.path.join(args.output_root, args.output_subdir)
    os.makedirs(comparison_root, exist_ok=True)

    index_lines = ["# EAGLE comparison sheets", ""]
    wrote = 0
    for category, rows in categories.items():
        category_dir = os.path.join(comparison_root, category)
        os.makedirs(category_dir, exist_ok=True)
        index_lines.extend([f"## {category}", ""])
        for row in rows:
            key = (canon_subset(row["subset"]), str(row["sample_id"]))
            sample = samples.get(key)
            if sample is None:
                print(f"[eagle.compose] missing dataset sample {key[0]}/{key[1]}")
                continue
            filename = f"{sample.subset}_{sample.sample_id}.png"
            path = os.path.join(category_dir, filename)
            compose_one(sample, category, model_names, conditions, args.output_root).save(path, quality=95)
            relpath = os.path.relpath(path, comparison_root)
            index_lines.append(f"- [{sample.subset}/{sample.sample_id}]({relpath})")
            wrote += 1
        index_lines.append("")

    with open(os.path.join(comparison_root, "README.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(index_lines) + "\n")
    print(f"[eagle.compose] wrote {wrote} comparison sheet(s) -> {comparison_root}")


if __name__ == "__main__":
    main()
