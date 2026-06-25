"""Merge the OPD benchmark-suite groups into one ordered table + suite_summary.json.

``scripts/eval_suite.sh`` runs up to three evaluators and this script stitches
their summaries into a single table:

  * judged group   (``run_opd_eval.py``, LLM judge) — MathVista, MathVerse,
                     MathVision, MMMU, MMMU-Pro (x2 sub-scores), MMStar,
                     HallusionBench. Greedy score = ``pass_at_k`` (== Acc@1).
  * deterministic  (``run_vqa_eval.py``, official metric, no judge) — POPE
                     (per category), ChartQA, VQAv2.
  * sampled (opt)  (judged datasets again, PASS_K=N, temperature>0) — used only to
                     report pass@k / avg@k for k in --ks from the SAME N samples
                     (unbiased estimator, baseline.eval.passk). Not run for the
                     deterministic group: pass@k on yes/no / short-answer isn't a
                     standard metric there (the official greedy metric is the point).

POPE is expanded into its 3 categories (random / popular / adversarial) **+ their
average**, and MMMU-Pro into its 2 sub-scores **+ their average**. Every score is a
0-1 fraction; the printed table also shows it as a percentage. Missing pieces
degrade gracefully to ``-`` instead of erroring.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )

from baseline.eval.passk import multi_k_from_file


def _load(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _basename(name: Any) -> str:
    return os.path.basename(str(name or "").rstrip("/")).lower()


def _mean(values: list[Any]) -> float | None:
    nums = [v for v in values if isinstance(v, (int, float))]
    return sum(nums) / len(nums) if nums else None


def _metrics(summary: dict[str, Any], benchmark: str) -> dict[str, Any]:
    return ((summary.get("benchmarks") or {}).get(benchmark) or {}).get("metrics") or {}


def _sampled_by_tail(sampled: dict[str, Any], ks: list[int]) -> dict[str, dict[str, Any]]:
    """tail(dataset id) -> multi_k dict, read from each dataset's judgment_file."""
    out: dict[str, dict[str, Any]] = {}
    for entry in sampled.get("datasets") or []:
        judgment_file = entry.get("judgment_file")
        if judgment_file:
            out[_basename(entry.get("dataset"))] = multi_k_from_file(judgment_file, ks)
    return out


def aggregate(
    judged: dict[str, Any],
    vqa: dict[str, Any],
    sampled: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    sampled = sampled or {}
    by_tail: dict[str, Any] = {}
    for entry in judged.get("datasets") or []:
        by_tail[_basename(entry.get("dataset"))] = entry.get("pass_at_k")

    rows: list[dict[str, Any]] = []

    def add(group: str, name: str, metric: str, score: Any, tail: str | None = None) -> None:
        row: dict[str, Any] = {"group": group, "name": name, "metric": metric, "score": score}
        if tail and tail in sampled:
            mk = sampled[tail]
            row["n_samples"] = mk.get("n_samples")
            row["avg"] = mk.get("avg")
            row["pass_at_k"] = mk.get("pass_at_k") or {}
        rows.append(row)

    # --- judged group: greedy score = Acc@1; sampled pass@k/avg@k attached by tail ---
    add("math", "MathVista", "Acc@1", by_tail.get("mathvista"), "mathvista")
    add("math", "MathVerse", "Acc@1", by_tail.get("mathverse"), "mathverse")
    add("math", "MathVision", "Acc@1", by_tail.get("mathvision"), "mathvision")
    add("mmmu", "MMMU", "Acc@1", by_tail.get("mmmu"), "mmmu")
    pro_opt = by_tail.get("mmmu_pro_10options")
    pro_vis = by_tail.get("mmmu-pro-vision")
    add("mmmu", "MMMU-Pro (10options)", "Acc@1", pro_opt, "mmmu_pro_10options")
    add("mmmu", "MMMU-Pro (vision)", "Acc@1", pro_vis, "mmmu-pro-vision")
    add("mmmu", "MMMU-Pro (avg)", "Acc@1", _mean([pro_opt, pro_vis]))
    add("other", "MMStar", "Acc@1", by_tail.get("mmstar"), "mmstar")
    add("other", "HallusionBench", "aAcc@1", by_tail.get("hallusionbench"), "hallusionbench")

    # --- POPE: 3 categories + average (F1); deterministic, greedy only ---
    pope = _metrics(vqa, "pope")
    categories = pope.get("by_category") or {}
    category_f1: list[Any] = []
    for category in ("random", "popular", "adversarial"):
        f1 = (categories.get(category) or {}).get("f1")
        category_f1.append(f1)
        add("pope", f"POPE ({category})", "F1", f1)
    pope_avg = _mean(category_f1)
    if pope_avg is None:  # categories absent (single-category run) -> overall F1
        pope_avg = pope.get("f1")
    add("pope", "POPE (avg)", "F1", pope_avg)

    # --- ChartQA / VQAv2 (deterministic, greedy only) ---
    add("chartqa", "ChartQA", "relaxed_acc", _metrics(vqa, "chartqa").get("relaxed_accuracy"))
    add("vqav2", "VQAv2", "vqa_soft_acc", _metrics(vqa, "vqav2").get("vqa_accuracy"))
    return rows


def _pct(value: Any) -> str:
    return f"{value * 100:6.2f}%" if isinstance(value, (int, float)) else f"{'-':>7}"


def print_table(rows: list[dict[str, Any]], model_name: str, ks: list[int]) -> None:
    has_multi = any("pass_at_k" in row for row in rows)
    pass_ks = [k for k in ks if k != 1]  # pass@1 == avg, so only show k>1 columns
    name_width = max((len(row["name"]) for row in rows), default=16) + 2

    print(f"\n=== OPD benchmark suite: {model_name} ===")
    header = f"{'benchmark':<{name_width}} {'metric':<13} {'greedy':>8}"
    if has_multi:
        header += f" {'avg@N':>8}" + "".join(f" {'pass@' + str(k):>8}" for k in pass_ks)
    print(header)
    print("-" * len(header))
    for row in rows:
        line = f"{row['name']:<{name_width}} {row['metric']:<13} {_pct(row['score'])}"
        if has_multi:
            if "pass_at_k" in row:
                line += f" {_pct(row.get('avg'))}"
                line += "".join(f" {_pct(row['pass_at_k'].get(k))}" for k in pass_ks)
            else:
                line += f" {'-':>8}" + "".join(f" {'-':>8}" for _ in pass_ks)
        print(line)
    if has_multi:
        n = next((row.get("n_samples") for row in rows if row.get("n_samples")), None)
        print(f"\n(avg@N = mean per-sample accuracy over N={n} samples; k-independent. "
              f"pass@k from the same N samples, unbiased estimator. greedy = Acc@1.)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate the OPD benchmark suite.")
    parser.add_argument("--judged-summary", required=True)
    parser.add_argument("--vqa-summary", required=True)
    parser.add_argument("--sampled-summary", default=None,
                        help="Optional judged-group run with PASS_K=N (temp>0) for pass@k/avg@k.")
    parser.add_argument("--ks", default="1,8,16", help="Comma-separated k list for pass@k.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", default=None)
    args = parser.parse_args()

    ks = [int(part) for part in str(args.ks).split(",") if part.strip()]
    judged = _load(args.judged_summary)
    vqa = _load(args.vqa_summary)
    sampled_summary = _load(args.sampled_summary)
    sampled = _sampled_by_tail(sampled_summary, ks) if sampled_summary else {}

    rows = aggregate(judged, vqa, sampled)
    model_name = (
        args.model_name
        or judged.get("model_name")
        or vqa.get("model_name")
        or "model"
    )

    print_table(rows, model_name, ks)
    output = {
        "model_name": model_name,
        "ks": ks,
        "rows": rows,
        "table": {row["name"]: row["score"] for row in rows},
        "judged_summary": args.judged_summary,
        "vqa_summary": args.vqa_summary,
        "sampled_summary": args.sampled_summary,
    }
    Path(args.output).write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
