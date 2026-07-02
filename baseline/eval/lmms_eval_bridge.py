"""Bridge helpers for running lmms-eval from the OPD eval scripts.

This module keeps two concerns out of shell:

* mapping user-facing benchmark names (``MathVista``, ``V* Bench``) to
  lmms-eval task names (``mathvista_testmini``, ``vstar_bench``);
* converting lmms-eval's aggregated ``*_results.json`` into the small
  ``summary.json`` shape consumed by ``baseline/eval/make_report.py``.

The conversion preserves the full lmms-eval metric dictionary for each benchmark
and exposes a single headline ``score`` for the existing methods x benchmarks
matrix. Values returned as percentages by lmms-eval tasks are normalized to
0..1 ratios so the existing report's percent formatting remains correct.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    tasks: tuple[str, ...]
    label: str
    metrics: tuple[str, ...]
    children: tuple[str, ...] = ()


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


STANDARD_BENCHMARKS = (
    "mathvista",
    "mathverse",
    "mathvision",
    "MMMU",
    "MMMU-Pro",
    "MMStar",
    "HallusionBench",
    "POPE",
    "ChartQA",
    "V* Bench",
    "HRBench4K",
    "HRBench8K",
    "MME-RealWorld-Lite",
)


_SPECS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec(
        "mathvista",
        ("mathvista_testmini",),
        "MathVista",
        ("gpt_eval_score", "llm_as_judge_eval", "accuracy", "acc"),
        ("mathvista_testmini_cot", "mathvista_testmini_solution", "mathvista_testmini_format"),
    ),
    BenchmarkSpec("mathverse", ("mathverse_testmini",), "MathVerse", ("gpt_eval_score", "accuracy", "acc")),
    BenchmarkSpec("mathvision", ("mathvision_testmini",), "MathVision", ("mathvision_standard_eval", "llm_as_judge_eval", "accuracy", "acc")),
    BenchmarkSpec("mmmu", ("mmmu_val",), "MMMU", ("mmmu_acc", "accuracy", "acc")),
    BenchmarkSpec(
        "mmmu-pro",
        ("mmmu_pro",),
        "MMMU-Pro",
        ("mmmu_acc", "accuracy", "acc"),
        ("mmmu_pro_standard", "mmmu_pro_vision"),
    ),
    BenchmarkSpec("mmstar", ("mmstar",), "MMStar", ("average", "acc", "accuracy")),
    BenchmarkSpec("hallusionbench", ("hallusion_bench_image",), "HallusionBench", ("aAcc", "qAcc", "fAcc")),
    BenchmarkSpec("pope", ("pope",), "POPE", ("pope_f1_score", "pope_accuracy", "f1")),
    BenchmarkSpec("chartqa", ("chartqa",), "ChartQA", ("relaxed_overall", "relaxed_accuracy", "accuracy")),
    BenchmarkSpec("vstar", ("vstar_bench",), "V* Bench", ("vstar_overall_acc", "accuracy", "acc")),
    BenchmarkSpec("hrbench4k", ("hrbench4k",), "HRBench4K", ("average", "single", "cross")),
    BenchmarkSpec("hrbench8k", ("hrbench8k",), "HRBench8K", ("average", "single", "cross")),
    BenchmarkSpec("mme-realworld-lite", ("mmerealworld_lite",), "MME-RealWorld-Lite", ("mme_realworld_score", "accuracy", "acc")),
)


ALIASES: dict[str, BenchmarkSpec] = {}
for spec in _SPECS:
    for alias in {spec.name, spec.label, *spec.tasks, *spec.children}:
        ALIASES[_norm(alias)] = spec

# Extra spellings users are likely to type.
for alias in ("mmmu_pro", "mmmupro", "mmmu pro"):
    ALIASES[_norm(alias)] = ALIASES[_norm("mmmu-pro")]
for alias in ("vstarbench", "vstar_bench", "v-star", "v*bench", "v* bench"):
    ALIASES[_norm(alias)] = ALIASES[_norm("vstar")]
for alias in ("mme_realworld_lite", "mmerealworldlite", "mme realworld lite", "mme-realworld-lite"):
    ALIASES[_norm(alias)] = ALIASES[_norm("mme-realworld-lite")]
for alias in ("hallusion_bench", "hallusion bench"):
    ALIASES[_norm(alias)] = ALIASES[_norm("hallusionbench")]


def _split_space_list(text: str) -> list[str]:
    tokens = text.split()
    if not tokens:
        return []
    valid = set(ALIASES)
    out: list[str] = []
    i = 0
    while i < len(tokens):
        # Greedily preserve known benchmark aliases that contain spaces, e.g. V* Bench.
        matched = None
        for n in range(min(5, len(tokens) - i), 0, -1):
            candidate = " ".join(tokens[i : i + n])
            if _norm(candidate) in valid:
                matched = candidate
                i += n
                break
        if matched is None:
            matched = tokens[i]
            i += 1
        out.append(matched)
    return out


def split_benchmarks(text: str | None) -> list[str]:
    if not text or text.strip().lower() in {"all", "standard", "default"}:
        return list(STANDARD_BENCHMARKS)
    stripped = text.strip()
    if "," in stripped:
        return [item.strip() for item in stripped.split(",") if item.strip()]
    return _split_space_list(stripped)


def resolve_benchmarks(names: list[str]) -> list[BenchmarkSpec]:
    specs: list[BenchmarkSpec] = []
    seen: set[str] = set()
    for name in names:
        if name.startswith("task:"):
            task = name.split(":", 1)[1]
            spec = BenchmarkSpec(task, (task,), task, ())
        else:
            spec = ALIASES.get(_norm(name))
            if spec is None:
                # Treat unknown names as raw lmms-eval task names. This keeps the
                # bridge useful when lmms-eval adds new tasks before this map is updated.
                spec = BenchmarkSpec(name, (name,), name, ())
        key = spec.name
        if key not in seen:
            specs.append(spec)
            seen.add(key)
    return specs


def cmd_tasks(args: argparse.Namespace) -> None:
    specs = resolve_benchmarks(split_benchmarks(args.benchmarks))
    tasks: list[str] = []
    seen: set[str] = set()
    for spec in specs:
        for task in spec.tasks:
            if task not in seen:
                tasks.append(task)
                seen.add(task)
    payload = {
        "requested": split_benchmarks(args.benchmarks),
        "tasks": tasks,
        "benchmarks": [asdict(spec) for spec in specs],
    }
    if args.out_map:
        Path(args.out_map).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_map).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(",".join(tasks))


def cmd_benchmarks(args: argparse.Namespace) -> None:
    specs = resolve_benchmarks(split_benchmarks(args.benchmarks))
    for spec in specs:
        print(spec.name)


def _latest_results_file(output_dir: Path) -> Path:
    files = [
        p
        for p in output_dir.rglob("*_results.json")
        if p.name != "summary.json" and not p.name.startswith("lmms_")
    ]
    if not files:
        raise SystemExit(f"No lmms-eval *_results.json found under {output_dir}")
    return max(files, key=lambda p: p.stat().st_mtime)


def _metric_name(key: str) -> str:
    return key.split(",", 1)[0]


def _as_ratio(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    val = float(value)
    # Several lmms-eval tasks return 0..100 percentages while most return 0..1.
    if abs(val) > 1.000001 and abs(val) <= 100.000001:
        return val / 100.0
    return val


def _numeric_metrics(entry: dict[str, Any] | None) -> dict[str, float]:
    if not isinstance(entry, dict):
        return {}
    out: dict[str, float] = {}
    for raw_key, raw_val in entry.items():
        key = _metric_name(str(raw_key))
        if key == "alias" or "stderr" in key.lower():
            continue
        val = _as_ratio(raw_val)
        if val is not None:
            out[key] = val
    return out


def _pick_metric(metrics: dict[str, float], preferred: tuple[str, ...]) -> tuple[str | None, float | None]:
    for key in preferred:
        if key in metrics:
            return key, metrics[key]
    if not metrics:
        return None, None
    key = sorted(metrics)[0]
    return key, metrics[key]


def _collect_entry(
    spec: BenchmarkSpec,
    results: dict[str, Any],
    groups: dict[str, Any],
) -> dict[str, Any]:
    checked: list[str] = []

    # Prefer group aggregates when the requested task is a group.
    for task in (*spec.tasks, spec.name):
        checked.append(task)
        metrics = _numeric_metrics(groups.get(task))
        if metrics:
            metric, score = _pick_metric(metrics, spec.metrics)
            return {
                "label": spec.label,
                "tasks": list(spec.tasks),
                "primary_metric": metric,
                "score": score,
                "metrics": metrics,
                "source": "groups",
            }

    # Then exact task results.
    for task in spec.tasks:
        checked.append(task)
        metrics = _numeric_metrics(results.get(task))
        if metrics:
            metric, score = _pick_metric(metrics, spec.metrics)
            return {
                "label": spec.label,
                "tasks": list(spec.tasks),
                "primary_metric": metric,
                "score": score,
                "metrics": metrics,
                "source": "results",
            }

    # Finally average known children, useful for group aliases when lmms-eval only
    # emits per-subtask metrics in the aggregated file.
    child_scores: dict[str, float] = {}
    child_metrics: dict[str, dict[str, float]] = {}
    for child in spec.children:
        checked.append(child)
        metrics = _numeric_metrics(results.get(child))
        metric, score = _pick_metric(metrics, spec.metrics)
        if score is not None:
            child_scores[child] = score
            child_metrics[child] = metrics
    if child_scores:
        avg = sum(child_scores.values()) / len(child_scores)
        return {
            "label": spec.label,
            "tasks": list(spec.tasks),
            "primary_metric": "mean_child_score",
            "score": avg,
            "metrics": {"mean_child_score": avg, **child_scores},
            "children": child_metrics,
            "source": "children",
        }

    return {
        "label": spec.label,
        "tasks": list(spec.tasks),
        "primary_metric": None,
        "score": None,
        "metrics": {},
        "source": "missing",
        "checked": checked,
    }


def cmd_convert(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    task_map = json.loads(Path(args.task_map).read_text(encoding="utf-8"))
    results_file = Path(args.results_json) if args.results_json else _latest_results_file(output_dir)
    lmms = json.loads(results_file.read_text(encoding="utf-8"))
    results = lmms.get("results") or {}
    groups = lmms.get("groups") or {}

    specs = [
        BenchmarkSpec(
            name=item["name"],
            tasks=tuple(item["tasks"]),
            label=item["label"],
            metrics=tuple(item["metrics"]),
            children=tuple(item.get("children") or ()),
        )
        for item in task_map.get("benchmarks", [])
    ]
    benchmarks = {spec.name: _collect_entry(spec, results, groups) for spec in specs}
    summary = {
        "backend": "lmms-eval",
        "lmms_eval_dir": args.lmms_eval_dir,
        "lmms_results_file": str(results_file),
        "model_path": args.model_path,
        "model_name": args.model_name or os.path.basename(args.model_path.rstrip("/")),
        "output_dir": str(output_dir),
        "prompt_mode": args.prompt_mode,
        "task_map": task_map,
        "benchmarks": benchmarks,
    }
    out = Path(args.summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_tasks = sub.add_parser("tasks", help="Resolve benchmark aliases to lmms-eval task names.")
    p_tasks.add_argument("--benchmarks", default="")
    p_tasks.add_argument("--out-map", default="")
    p_tasks.set_defaults(func=cmd_tasks)

    p_benchmarks = sub.add_parser("benchmarks", help="Resolve benchmark aliases to canonical names.")
    p_benchmarks.add_argument("--benchmarks", default="")
    p_benchmarks.set_defaults(func=cmd_benchmarks)

    p_convert = sub.add_parser("convert", help="Convert lmms-eval results into OPD summary.json.")
    p_convert.add_argument("--output-dir", required=True)
    p_convert.add_argument("--task-map", required=True)
    p_convert.add_argument("--summary", required=True)
    p_convert.add_argument("--model-name", default="")
    p_convert.add_argument("--model-path", required=True)
    p_convert.add_argument("--prompt-mode", default="lmms")
    p_convert.add_argument("--lmms-eval-dir", default="")
    p_convert.add_argument("--results-json", default="")
    p_convert.set_defaults(func=cmd_convert)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
