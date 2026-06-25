"""Merge the OPD benchmark-suite groups into one ordered table + suite_summary.json.

``scripts/eval_suite.sh`` runs two evaluators and this script stitches their
``summary.json`` files into a single comparison table:

  * judged group   (``run_opd_eval.py``, LLM judge) — MathVista, MathVerse,
                     MathVision, MMMU, MMMU-Pro (x2 sub-scores), MMStar,
                     HallusionBench. Score = ``pass_at_k`` (== Acc@1 when greedy).
  * deterministic  (``run_vqa_eval.py``, official metric, no judge) — POPE
                     (per category), ChartQA, VQAv2.

POPE is expanded into its 3 categories (random / popular / adversarial) **+ their
average**, and MMMU-Pro into its 2 sub-scores **+ their average**, per request.
Every score is a 0-1 fraction; the printed table also shows it as a percentage.
Missing benchmarks degrade gracefully to ``-`` instead of erroring.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _load(path: str) -> dict[str, Any]:
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


def aggregate(judged: dict[str, Any], vqa: dict[str, Any]) -> list[dict[str, Any]]:
    # Judged datasets keyed by the last path segment of their HF id.
    by_tail: dict[str, Any] = {}
    for entry in judged.get("datasets") or []:
        by_tail[_basename(entry.get("dataset"))] = entry.get("pass_at_k")

    rows: list[dict[str, Any]] = []

    def add(group: str, name: str, metric: str, score: Any) -> None:
        rows.append({"group": group, "name": name, "metric": metric, "score": score})

    # --- judged group (Acc@1 when greedy) ---
    add("math", "MathVista", "Acc@1", by_tail.get("mathvista"))
    add("math", "MathVerse", "Acc@1", by_tail.get("mathverse"))
    add("math", "MathVision", "Acc@1", by_tail.get("mathvision"))
    add("mmmu", "MMMU", "Acc@1", by_tail.get("mmmu"))
    pro_opt = by_tail.get("mmmu_pro_10options")
    pro_vis = by_tail.get("mmmu-pro-vision")
    add("mmmu", "MMMU-Pro (10options)", "Acc@1", pro_opt)
    add("mmmu", "MMMU-Pro (vision)", "Acc@1", pro_vis)
    add("mmmu", "MMMU-Pro (avg)", "Acc@1", _mean([pro_opt, pro_vis]))
    add("other", "MMStar", "Acc@1", by_tail.get("mmstar"))
    add("other", "HallusionBench", "aAcc@1", by_tail.get("hallusionbench"))

    # --- POPE: 3 categories + average (F1) ---
    pope = _metrics(vqa, "pope")
    categories = pope.get("by_category") or {}
    category_f1: list[Any] = []
    for category in ("random", "popular", "adversarial"):
        f1 = (categories.get(category) or {}).get("f1")
        category_f1.append(f1)
        add("pope", f"POPE ({category})", "F1", f1)
    pope_avg = _mean(category_f1)
    if pope_avg is None:  # categories absent (e.g. a single-category run) -> overall F1
        pope_avg = pope.get("f1")
    add("pope", "POPE (avg)", "F1", pope_avg)

    # --- ChartQA / VQAv2 ---
    add("chartqa", "ChartQA", "relaxed_acc", _metrics(vqa, "chartqa").get("relaxed_accuracy"))
    add("vqav2", "VQAv2", "vqa_soft_acc", _metrics(vqa, "vqav2").get("vqa_accuracy"))
    return rows


def print_table(rows: list[dict[str, Any]], model_name: str) -> None:
    name_width = max((len(row["name"]) for row in rows), default=16) + 2
    print(f"\n=== OPD benchmark suite: {model_name} ===")
    print(f"{'benchmark':<{name_width}} {'metric':<14} {'score':>9}")
    print("-" * (name_width + 25))
    for row in rows:
        score = row["score"]
        cell = f"{score * 100:>8.2f}%" if isinstance(score, (int, float)) else f"{'-':>9}"
        print(f"{row['name']:<{name_width}} {row['metric']:<14} {cell}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate the OPD benchmark suite.")
    parser.add_argument("--judged-summary", required=True)
    parser.add_argument("--vqa-summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", default=None)
    args = parser.parse_args()

    judged = _load(args.judged_summary)
    vqa = _load(args.vqa_summary)
    rows = aggregate(judged, vqa)
    model_name = (
        args.model_name
        or judged.get("model_name")
        or vqa.get("model_name")
        or "model"
    )

    print_table(rows, model_name)
    output = {
        "model_name": model_name,
        "rows": rows,
        "table": {row["name"]: row["score"] for row in rows},
        "judged_summary": args.judged_summary,
        "vqa_summary": args.vqa_summary,
    }
    Path(args.output).write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
