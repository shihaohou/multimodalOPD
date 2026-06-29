"""Compare several G0 runs side by side (e.g. two teachers: Qwen3-VL-8B vs CapCurriculum-8B).

G0 is one-teacher-per-run, so each teacher lives in its own dir. This tool ingests
N run dirs and tabulates the TEACHER-side analyses (head usability + the C1-vs-C2
hint mechanism + C1 accuracy/IoU/vt) across them, plus the STUDENT (C3) block —
which is the same 2B model in every run, so it doubles as a consistency check
(numbers should be close; differences = different sample counts / shards).

Reuses ``analyze_g0`` so the metrics/verdicts are identical to the per-run report.

  uv run python -m baseline.g0.compare_g0 \
      --run-dirs eval_outputs/g0/Qwen3-VL-8B-Instruct eval_outputs/g0/CapCurriculum-8B
"""

from __future__ import annotations

import argparse
import json
import os
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from baseline.g0.analyze_g0 import (
    _fmt,
    _mean,
    analysis_1_head_usability,
    analysis_2_looking_vs_using,
    analysis_3_hint_mechanism,
    by_condition,
    load_records,
)


def summarize_run(run_dir: str) -> dict:
    records = load_records(run_dir)
    conds = by_condition(records)
    c1, c2, c3 = conds.get("c1", []), conds.get("c2", []), conds.get("c3", [])
    return {
        "name": os.path.basename(run_dir.rstrip("/")),
        "run_dir": run_dir,
        "n": {k: len(v) for k, v in conds.items()},
        "head": analysis_1_head_usability(run_dir, conds),
        "hint": analysis_3_hint_mechanism(c1, c2),
        "student": analysis_2_looking_vs_using(c3) if c3 else {},
        # teacher C1 raw means (for the table)
        "c1_acc": (sum(r["correct"] for r in c1) / len(c1)) if c1 else float("nan"),
        "c1_iou_lh": _mean(c1, "iou_lh"),
        "c1_vt": _mean(c1, "vt_ratio"),
        "c2_acc": (sum(r["correct"] for r in c2) / len(c2)) if c2 else float("nan"),
    }


def _row(cells, widths):
    return "| " + " | ".join(str(c).ljust(w) for c, w in zip(cells, widths)) + " |"


def render(runs: list[dict]) -> str:
    L = ["# G0 cross-run comparison", ""]

    # ---- teacher table (one row per run) ----
    L.append("## Teacher (C1 natural / C2 hidden-hint)")
    L.append("")
    hdr = ["run", "n(C1)", "acc_C1", "head_best_IoU", "C1_IoU_LH", "C1_pointing",
           "head", "Δacc(hint)", "ΔIoU_LH(hint)", "hint_verdict"]
    w = [22, 7, 7, 14, 10, 12, 7, 11, 14, 14]
    L.append(_row(hdr, w))
    L.append(_row(["-" * x for x in w], w))
    for r in runs:
        t = r["head"].get("teacher", {})
        h = r["hint"]
        hv = (h.get("verdict", "—").split(":")[0]) if h else "—"
        L.append(_row([
            r["name"], r["n"].get("c1", 0), _fmt(r["c1_acc"]),
            _fmt(t.get("best_head_mean_iou")), _fmt(t.get("assembled_iou_lh_mean")),
            _fmt(t.get("assembled_pointing_mean")), t.get("verdict", "—"),
            _fmt(h.get("delta_accuracy")) if h else "—",
            _fmt(h.get("delta_iou_lh")) if h else "—", hv,
        ], w))
    L.append("")
    L.append("_Δacc(hint)=acc(C2 hint)−acc(C1); ≈0 ⇒ where-to-look isn't the lever. "
             "C1_pointing = teacher's first-gen-step argmax-in-GT rate._")
    L.append("")

    # ---- student (C3) consistency (same 2B model in every run) ----
    L.append("## Student (C3) — same model across runs (consistency check)")
    L.append("")
    hdr2 = ["run", "n(C3)", "acc", "IoU_LH", "vt", "corr(c,IoU)", "corr(c,vt)", "verdict"]
    w2 = [22, 7, 7, 8, 8, 12, 11, 30]
    L.append(_row(hdr2, w2))
    L.append(_row(["-" * x for x in w2], w2))
    for r in runs:
        s = r["student"]
        if not s:
            L.append(_row([r["name"], r["n"].get("c3", 0), "—", "—", "—", "—", "—", "no C3"], w2))
            continue
        L.append(_row([
            r["name"], s["n"], _fmt(s["accuracy"]), _fmt(s["mean_iou_lh_right"]),
            _fmt(s["mean_vt_ratio"]), _fmt(s["corr_correct_iou_lh"]),
            _fmt(s["corr_correct_vt"]), s["verdict"].split(":")[0],
        ], w2))
    L.append("")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare several G0 run dirs side by side.")
    ap.add_argument("--run-dirs", nargs="+", required=True)
    ap.add_argument("--output", default=None, help="Write the markdown table here (else stdout only).")
    args = ap.parse_args()

    runs = [summarize_run(d) for d in args.run_dirs]
    md = render(runs)
    print(md)
    combined = {r["name"]: {k: r[k] for k in ("n", "head", "hint", "student", "c1_acc", "c2_acc")} for r in runs}
    if args.output:
        with open(args.output, "w") as f:
            f.write(md)
        with open(os.path.splitext(args.output)[0] + ".json", "w") as f:
            json.dump(combined, f, indent=2)
        print(f"[g0.compare] wrote {args.output}")


if __name__ == "__main__":
    main()
