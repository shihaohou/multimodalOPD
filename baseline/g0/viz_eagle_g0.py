"""Post-hoc EAGLE-G0 visualization for selected records.

Reads an existing EAGLE-G0 run directory, selects a small stratified set of cases
from records*.jsonl, then re-runs only those cases with ``want_viz=True`` to write
overlay PNGs. This avoids redoing judge or the full evaluation.

Example:

    python -m baseline.g0.viz_eagle_g0 \
      --run-dir eval_outputs/eagle_g0/qwen3vl-2b-opd \
      --select wrong --rank-condition plain --conditions plain,hint \
      --per-subset 3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from argparse import Namespace
from collections import defaultdict

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from baseline.g0.analyze_g0 import apply_judge, load_records
from baseline.g0.engine import load_g0_model
from baseline.g0.run_eagle_g0 import run_condition, save_viz
from baseline.probe.saliency_data import (
    SaliencySample,
    _avoid_eager_image_decode,
    _load_hf_split,
    _to_pil,
    bbox_area,
    canon_subset,
    parse_bbox_norm,
)


def _finite(value, default: float = float("inf")) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(v) else v


SELECTORS = {
    "wrong": (lambda r: not bool(r.get("correct", False)), lambda r: _finite(r.get("iou_eagle"))),
    "correct": (lambda r: bool(r.get("correct", False)), lambda r: -_finite(r.get("visual_log_lift"), 0.0)),
    "low_iou_eagle": (lambda r: True, lambda r: _finite(r.get("iou_eagle"))),
    "high_iou_eagle": (lambda r: True, lambda r: -_finite(r.get("iou_eagle"), 0.0)),
    "low_visual_log_lift": (lambda r: True, lambda r: _finite(r.get("visual_log_lift"))),
    "high_visual_log_lift": (lambda r: True, lambda r: -_finite(r.get("visual_log_lift"), 0.0)),
    "low_visual_fraction": (lambda r: True, lambda r: _finite(r.get("visual_fraction"))),
    "high_visual_fraction": (lambda r: True, lambda r: -_finite(r.get("visual_fraction"), 0.0)),
    "random": (lambda r: True, None),
}


def _load_config(run_dir: str) -> dict:
    path = os.path.join(run_dir, "config.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _available_conditions(records: list[dict]) -> list[str]:
    return sorted({str(r.get("condition", "")) for r in records if r.get("condition")})


def _dedupe_records(records: list[dict]) -> list[dict]:
    """Keep one record per model/condition/subset/sample_id.

    Old exploratory runs can leave stale shard files in the same directory. For
    visualization that only needs case selection, duplicate logical records are
    unhelpful; later records in the sorted glob/load order win.
    """
    out: dict[tuple, dict] = {}
    for r in records:
        key = (r.get("model", ""), r.get("condition", ""), r.get("subset", ""), str(r.get("sample_id", "")))
        out[key] = r
    return list(out.values())


def select_cases(
    records: list[dict],
    *,
    select: str,
    rank_condition: str,
    subsets: set[str] | None,
    per_subset: int,
    max_cases: int | None,
    seed: int,
) -> list[dict]:
    filt, key_fn = SELECTORS[select]
    candidates = [
        r for r in records
        if r.get("condition") == rank_condition
        and (subsets is None or canon_subset(r.get("subset", "")) in subsets)
        and filt(r)
    ]
    by_subset: dict[str, list[dict]] = defaultdict(list)
    for r in candidates:
        by_subset[str(r.get("subset", "unknown"))].append(r)

    chosen: list[dict] = []
    for subset, rows in sorted(by_subset.items()):
        if key_fn is None:
            rng = random.Random(f"{seed}:{subset}")
            rng.shuffle(rows)
            picked = rows[:per_subset]
        else:
            picked = sorted(rows, key=key_fn)[:per_subset]
        chosen.extend(picked)

    if max_cases is not None and max_cases >= 0:
        chosen = chosen[:max_cases]
    return chosen


def load_selected_samples(cfg: dict, keys: set[tuple[str, str]]) -> dict[tuple[str, str], SaliencySample]:
    """Load only the images needed for selected ``(subset, sample_id)`` keys."""
    data = _avoid_eager_image_decode(_load_hf_split(cfg["dataset"], cfg.get("split", "train")))
    wanted = {(canon_subset(subset), str(sample_id)) for subset, sample_id in keys}
    out: dict[tuple[str, str], SaliencySample] = {}
    max_bbox_area = cfg.get("max_bbox_area")
    min_bbox_area = cfg.get("min_bbox_area")

    for index in range(len(data)):
        record = data[index]
        subset = str(record.get("dataset", "")).strip() or "unknown"
        sample_id = str(record.get("question_id", index))
        lookup = (canon_subset(subset), sample_id)
        if lookup not in wanted:
            continue

        bbox = parse_bbox_norm(record.get("bbox"))
        if bbox is None:
            continue
        area = bbox_area(bbox)
        if (max_bbox_area is not None and area > max_bbox_area) or (
            min_bbox_area is not None and area < min_bbox_area
        ):
            continue
        problem = str(record.get("problem", "")).strip()
        solution = str(record.get("solution", "")).strip()
        if not problem or not solution:
            continue
        sample = SaliencySample(sample_id, subset, problem, solution, _to_pil(record.get("image")), bbox)
        out[(canon_subset(subset), sample_id)] = sample
        if len(out) == len(wanted):
            break
    return out


def _viz_args_from_config(cfg: dict, cli: argparse.Namespace) -> Namespace:
    ns = Namespace(**cfg)
    # Keep post-hoc visualization cheap and compatible with runs made under sdpa.
    ns.grad_probes = False
    ns.salr1 = bool(cli.with_salr1)
    ns.debug_mem = bool(cli.debug_mem)
    if cli.eagle_batch_size is not None:
        ns.eagle_batch_size = cli.eagle_batch_size
    if cli.eagle_image_size is not None:
        ns.eagle_image_size = cli.eagle_image_size
    if cli.max_new_tokens is not None:
        ns.max_new_tokens = cli.max_new_tokens
    if cli.attn is not None:
        ns.attn = cli.attn
    if not hasattr(ns, "sample"):
        ns.sample = False
    if not hasattr(ns, "temperature"):
        ns.temperature = 1.0
    if not hasattr(ns, "top_p"):
        ns.top_p = 1.0
    if not hasattr(ns, "hint_mode"):
        ns.hint_mode = "generate"
    return ns


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Render selected post-hoc EAGLE-G0 visualizations.")
    ap.add_argument("--run-dir", required=True, help="One model directory under eval_outputs/eagle_g0.")
    ap.add_argument("--select", default="wrong", choices=sorted(SELECTORS))
    ap.add_argument("--rank-condition", default="plain")
    ap.add_argument("--conditions", default="plain,hint", help="Conditions to render for each selected sample.")
    ap.add_argument("--subsets", default="", help="Optional comma list; default uses all subsets present in records.")
    ap.add_argument("--per-subset", type=int, default=3)
    ap.add_argument("--max-cases", type=int, default=None, help="Optional global cap after per-subset selection.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-subdir", default=None, help="Default: viz_<select>.")
    ap.add_argument("--no-use-judge", dest="use_judge", action="store_false",
                    help="Do not overlay judgments.jsonl before selecting cases.")
    ap.set_defaults(use_judge=True)
    ap.add_argument("--with-salr1", action="store_true", help="Also render Saliency-R1; slower.")
    ap.add_argument("--attn", default=None, help="Override config attn implementation, e.g. eager or sdpa.")
    ap.add_argument("--eagle-batch-size", type=int, default=None)
    ap.add_argument("--eagle-image-size", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--debug-mem", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_config(args.run_dir)
    records = _dedupe_records(load_records(args.run_dir))
    if args.use_judge:
        apply_judge(args.run_dir, records)

    available = _available_conditions(records)
    if not available:
        raise SystemExit(f"[eagle.viz] no records found in {args.run_dir}")
    rank_condition = args.rank_condition if args.rank_condition in available else available[0]
    render_conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    render_conditions = [c for c in render_conditions if c in available]
    if not render_conditions:
        render_conditions = [rank_condition]
    if cfg.get("hint_mode") == "score_plain_y" and "hint" in render_conditions and "plain" not in render_conditions:
        render_conditions = ["plain"] + render_conditions

    subset_filter = {canon_subset(s) for s in args.subsets.split(",") if s.strip()} or None
    chosen = select_cases(
        records, select=args.select, rank_condition=rank_condition, subsets=subset_filter,
        per_subset=args.per_subset, max_cases=args.max_cases, seed=args.seed,
    )
    if not chosen:
        raise SystemExit(f"[eagle.viz] no cases selected for {args.select}/{rank_condition}")

    keys = {(str(r["subset"]), str(r["sample_id"])) for r in chosen}
    samples = load_selected_samples(cfg, keys)
    missing = sorted({(canon_subset(subset), sample_id) for subset, sample_id in keys} - set(samples))
    if missing:
        print(f"[eagle.viz] WARNING: {len(missing)} selected sample(s) were not found after dataset filters.")

    ns = _viz_args_from_config(cfg, args)
    gm = load_g0_model(
        cfg["model"], cfg.get("model_name") or os.path.basename(str(cfg["model"]).rstrip("/")),
        attn=ns.attn, dtype=ns.dtype, device=ns.device,
        min_pixels=getattr(ns, "min_pixels", None), max_pixels=getattr(ns, "max_pixels", None),
    )

    subdir = args.output_subdir or f"viz_{args.select}"
    wrote = 0
    for rec in chosen:
        sample = samples.get((canon_subset(rec["subset"]), str(rec["sample_id"])))
        if sample is None:
            continue
        plain_completion = None
        for cond in render_conditions:
            hint = cond == "hint"
            reuse = plain_completion if (hint and ns.hint_mode == "score_plain_y") else None
            try:
                record, (eg, gl_res, lh_first, salr1_res), comp = run_condition(
                    gm, sample, hint=hint, selected_heads=None, args=ns,
                    want_viz=True, reuse_completion=reuse,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[eagle.viz] skip {sample.subset}/{sample.sample_id} {cond}: {exc}")
                continue
            if cond == "plain":
                plain_completion = comp
            save_viz(
                args.run_dir, sample, eg, gl_res, lh_first,
                tag=f"{sample.subset}_{sample.sample_id}_{cond}",
                salr1_res=salr1_res, subdir=subdir,
            )
            print(f"[eagle.viz] wrote {sample.subset}/{sample.sample_id} {cond} "
                  f"(correct={record['correct']}, IoU={record['iou_eagle']:.3f})")
            wrote += 1

    out_dir = os.path.join(args.run_dir, subdir)
    print(f"[eagle.viz] wrote {wrote} PNG(s) -> {out_dir}/")


if __name__ == "__main__":
    main()
