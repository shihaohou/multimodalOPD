"""CPU-only sanity check for saliency-r1-8k BEFORE burning any GPU.

Run this first. It (1) prints the schema / per-subset counts / bbox-area stats and
a few example rows, and (2) dumps overlay montages so you can *eyeball* that the
normalized bbox maps onto the evidence region. If the red "evidence" box does not
sit on the answer region, stop and fix the mapping -- every downstream number
depends on it.

  uv run python baseline/probe/inspect_saliency.py \
      --dataset peterant330/saliency-r1-8k --num-sheets 16 --output-dir probe_outputs/inspect
"""

from __future__ import annotations

import argparse
import os
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from baseline.probe.image_ops import build_sanity_sheet
from baseline.probe.saliency_data import load_saliency_samples


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sanity-inspect saliency-r1-8k (CPU only).")
    p.add_argument("--dataset", default="peterant330/saliency-r1-8k")
    p.add_argument("--split", default="train")
    p.add_argument("--limit", type=int, default=200, help="Per-subset cap.")
    p.add_argument("--subsets", default=None, help="Comma list e.g. CUB,DocVQA.")
    p.add_argument("--num-sheets", type=int, default=16, help="Overlay montages to dump.")
    p.add_argument("--mask-fill", default="gray", choices=["gray", "black", "mean", "blur"])
    p.add_argument("--crop-pads", default="0,0.1,0.2")
    p.add_argument("--output-dir", default="probe_outputs/inspect")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    subsets = [s for s in (args.subsets or "").split(",") if s.strip()] or None
    pads = tuple(float(x) for x in args.crop_pads.split(",") if x.strip() != "")

    samples = load_saliency_samples(
        args.dataset, args.split, limit=args.limit, subsets=subsets
    )
    if not samples:
        raise SystemExit("No samples loaded -- check --dataset / --subsets.")

    # ---- schema / distribution stats -------------------------------------
    from collections import defaultdict

    areas: dict[str, list[float]] = defaultdict(list)
    sizes: list[tuple[int, int]] = []
    for s in samples:
        x1, y1, x2, y2 = s.bbox_norm
        areas[s.subset].append((x2 - x1) * (y2 - y1))
        sizes.append(s.image.size)
    print("\n=== saliency-r1-8k probe sample ===")
    print(f"total usable samples: {len(samples)}")
    for subset, vals in sorted(areas.items()):
        arr = np.asarray(vals)
        print(
            f"  [{subset}] n={len(arr):4d}  bbox area frac: "
            f"min={arr.min():.3f} median={np.median(arr):.3f} max={arr.max():.3f}"
        )
    ws = np.asarray([w for w, _ in sizes])
    hs = np.asarray([h for _, h in sizes])
    print(f"image size px: W [{ws.min()}..{ws.max()}] median {int(np.median(ws))}; "
          f"H [{hs.min()}..{hs.max()}] median {int(np.median(hs))}")

    print("\n--- example rows (first per subset) ---")
    seen: set[str] = set()
    for s in samples:
        if s.subset in seen:
            continue
        seen.add(s.subset)
        print(f"  [{s.subset}] id={s.sample_id}")
        print(f"      Q: {s.problem[:140]}")
        print(f"      A: {s.solution[:80]}")
        print(f"      bbox_norm: {tuple(round(v, 3) for v in s.bbox_norm)}")

    # ---- overlay montages -------------------------------------------------
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    # Spread the dumped sheets across subsets, not just the first rows.
    by_subset: dict[str, list] = defaultdict(list)
    for s in samples:
        by_subset[s.subset].append(s)
    picks = []
    per_subset = max(1, args.num_sheets // max(1, len(by_subset)))
    for subset, group in sorted(by_subset.items()):
        picks.extend(group[:per_subset])
    picks = picks[: args.num_sheets]
    for s in picks:
        sheet = build_sanity_sheet(s.image, s.bbox_norm, rng, fill=args.mask_fill, pads=pads)
        path = os.path.join(out_dir, f"{s.subset}_{s.sample_id}.png")
        sheet.save(path)
    print(f"\nWrote {len(picks)} overlay montages to {out_dir}/")
    print("=> OPEN a few and confirm the RED box sits on the answer evidence before running Stage 0.")


if __name__ == "__main__":
    main()
