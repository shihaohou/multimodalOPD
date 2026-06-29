"""Post-hoc, criterion-based, stratified visualization for a finished G0 run.

The inline viz in ``run_g0`` is the first-N-per-subset sanity set. This tool
instead lets you *see the cases the numbers point to*: it reads a run's
``records.jsonl`` (which already has every sample's IoU_LH / IoU_GL / vt_ratio /
correct), selects the most informative cases **per subset** by a criterion, and
re-runs just those forwards to render heatmap overlays. Cheap — only a handful of
samples are re-forwarded — and it reuses the run's saved config + calibrated heads.

Criteria (``--select``), ranked on ``--rank-condition`` (default c3 = student):
  * ``low_iou_lh``      — worst "looking" (attention misses the GT box).
  * ``low_iou_gl``      — worst GLIMPSE visual map.
  * ``low_vt``          — most text/prior-driven answers (using-failure flavor).
  * ``using_failure``   — WRONG but high IoU_LH (looked right, answered wrong) ⚠.
  * ``looking_failure`` — WRONG and low IoU_LH (looked wrong).
  * ``wrong``           — any wrong, worst IoU first.
  * ``high_iou_lh``     — best "looking" (positive control / sanity).
  * ``random``          — stratified random (seeded).

  uv run python -m baseline.g0.viz_g0 --run-dir eval_outputs/g0/run1 \
      --select using_failure --per-subset 4 --conditions c1,c2,c3
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
from argparse import Namespace
from collections import defaultdict

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _is_nan(x) -> bool:
    return isinstance(x, float) and math.isnan(x)


# (filter, sort-key) per criterion. Lower key = picked first; for "random" key is None.
SELECTORS = {
    "low_iou_lh": (lambda r: True, lambda r: r["iou_lh"]),
    "high_iou_lh": (lambda r: True, lambda r: -r["iou_lh"]),
    "low_iou_gl": (lambda r: True, lambda r: r["iou_gl"]),
    "low_vt": (lambda r: not _is_nan(r["vt_ratio"]), lambda r: r["vt_ratio"]),
    "using_failure": (lambda r: not r["correct"], lambda r: -r["iou_lh"]),
    "looking_failure": (lambda r: not r["correct"], lambda r: r["iou_lh"]),
    "wrong": (lambda r: not r["correct"], lambda r: r["iou_lh"]),
    "random": (lambda r: True, None),
}


def load_all_records(run_dir: str) -> list[dict]:
    records = []
    for path in sorted(glob.glob(os.path.join(run_dir, "records*.jsonl"))):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # tolerate a half-written trailing line during a live run
    return records


def select_cases(records, *, rank_condition, select, per_subset, seed) -> list[dict]:
    """Pick ``per_subset`` cases per subset from the ``rank_condition`` records."""
    filt, key = SELECTORS[select]
    recs = [r for r in records if r["condition"] == rank_condition and filt(r)]
    by_subset = defaultdict(list)
    for r in recs:
        by_subset[r["subset"]].append(r)
    chosen = []
    for subset, rs in sorted(by_subset.items()):
        if key is None:  # random
            rng = random.Random(f"{seed}:{subset}")
            rng.shuffle(rs)
            picked = rs[:per_subset]
        else:
            picked = sorted(rs, key=key)[:per_subset]
        chosen.extend(picked)
    return chosen


def load_head_list(run_dir: str, tag: str):
    path = os.path.join(run_dir, f"head_stats_{tag}.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [tuple(h) for h in json.load(f).get("selected_heads", [])]


def main() -> None:
    ap = argparse.ArgumentParser(description="Criterion-based stratified G0 viz.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--select", default="low_iou_lh", choices=sorted(SELECTORS))
    ap.add_argument("--per-subset", type=int, default=4)
    ap.add_argument("--conditions", default="c1,c2,c3", help="Which conditions to render per case.")
    ap.add_argument("--rank-condition", default="c3", help="Condition whose records the criterion ranks.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(os.path.join(args.run_dir, "config.json")) as f:
        cfg = json.load(f)
    ns = Namespace(**cfg)

    records = load_all_records(args.run_dir)
    available = {r["condition"] for r in records}
    conditions = [c for c in args.conditions.split(",") if c.strip() in available]
    rank_cond = args.rank_condition if args.rank_condition in available else (
        "c3" if "c3" in available else sorted(available)[0]
    )
    chosen = select_cases(records, rank_condition=rank_cond, select=args.select,
                          per_subset=args.per_subset, seed=args.seed)
    print(f"[g0.viz] select={args.select} rank={rank_cond} → {len(chosen)} cases; render conditions={conditions}")
    if not chosen:
        print("[g0.viz] nothing selected (criterion filtered everything?).")
        return

    # Heavy imports / model load only now (after we know there's work).
    from baseline.g0.engine import load_g0_model
    from baseline.g0.run_g0 import run_condition, save_viz
    from baseline.probe.saliency_data import load_saliency_samples

    glimpse_layers = (
        tuple(int(x) for x in cfg.get("glimpse_layers").split(",") if x.strip())
        if cfg.get("glimpse_layers") else None
    )
    pix = dict(min_pixels=cfg.get("min_pixels"), max_pixels=cfg.get("max_pixels"))
    need_teacher = bool({"c1", "c2"} & set(conditions)) and cfg.get("teacher_model")
    need_student = "c3" in conditions

    student = teacher = None
    cond_specs = {}
    if need_student:
        student = load_g0_model(cfg["student_model"], "student", attn=cfg["attn"],
                                dtype=cfg["dtype"], device=cfg["device"], **pix)
        cond_specs["c3"] = (student, False, load_head_list(args.run_dir, "student"))
    if need_teacher:
        teacher = load_g0_model(cfg["teacher_model"], "teacher", attn=cfg["attn"],
                                dtype=cfg["dtype"], device=cfg["device"], **pix)
        t_heads = load_head_list(args.run_dir, "teacher")
        cond_specs["c1"] = (teacher, False, t_heads)
        cond_specs["c2"] = (teacher, True, t_heads)
    conditions = [c for c in conditions if c in cond_specs]

    # Same deterministic load as the run → identical samples; index by (subset, id).
    subsets = [s for s in cfg.get("subsets", "").split(",") if s.strip()] or None
    limit = cfg.get("limit")
    samples = load_saliency_samples(
        cfg["dataset"], cfg.get("split", "train"),
        limit=(None if (limit in (None, 0) or limit < 0) else limit),
        subsets=subsets, max_bbox_area=cfg.get("max_bbox_area"), min_bbox_area=cfg.get("min_bbox_area"),
    )
    by_key = {(s.subset, str(s.sample_id)): s for s in samples}

    subdir = f"viz_{args.select}"
    n = 0
    for rec in chosen:
        key = (rec["subset"], str(rec["sample_id"]))
        sample = by_key.get(key)
        if sample is None:
            print(f"[g0.viz] sample {key} not found in dataset load; skipping.")
            continue
        for cond in conditions:
            gm, hint, heads = cond_specs[cond]
            try:
                record, (lh_res, gl_res) = run_condition(
                    gm, sample, hint=hint, selected_heads=heads, args=ns,
                    glimpse_layers=glimpse_layers, want_viz=True,
                )
            except Exception as exc:
                print(f"[g0.viz] skip {key} {cond}: {exc}")
                continue
            save_viz(
                args.run_dir, sample.image, sample.bbox_norm, lh_res, gl_res,
                tag=f"{rec['subset']}_{rec['sample_id']}_{cond}",
                subdir=subdir,
                suptitle=f"[{args.select}] {rec['subset']} {rec['sample_id']} {cond} | "
                         f"correct={record['correct']} IoU_LH={record['iou_lh']:.2f} "
                         f"IoU_GL={record['iou_gl']:.2f} vt={record['vt_ratio']:.2f}",
            )
            n += 1
    print(f"[g0.viz] wrote {n} overlays → {os.path.join(args.run_dir, subdir)}/")


if __name__ == "__main__":
    main()
