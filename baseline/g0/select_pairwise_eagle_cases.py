"""Select shared EAGLE visualization cases by teacher/student correctness."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict

from baseline.g0.analyze_g0 import apply_judge, load_records
from baseline.probe.saliency_data import canon_subset


CATEGORIES = (
    "stu_wrong_tea_correct",
    "stu_correct_tea_correct",
    "stu_wrong_tea_wrong",
    "stu_correct_tea_wrong",
)


def _dedupe(records: list[dict]) -> list[dict]:
    out: dict[tuple[str, str, str], dict] = {}
    for record in records:
        key = (
            str(record.get("condition", "")),
            canon_subset(record.get("subset", "")),
            str(record.get("sample_id", "")),
        )
        out[key] = record
    return list(out.values())


def _index(records: list[dict], condition: str, subsets: set[str] | None) -> dict[tuple[str, str], dict]:
    out = {}
    for record in records:
        if str(record.get("condition", "")) != condition:
            continue
        subset = canon_subset(record.get("subset", ""))
        if subsets is not None and subset not in subsets:
            continue
        out[(subset, str(record.get("sample_id", "")))] = record
    return out


def _complete_artifact_keys(run_dir: str, conditions: list[str]) -> set[tuple[str, str]]:
    records = _dedupe(load_records(run_dir))
    index = {
        (
            str(record.get("condition", "")),
            canon_subset(record.get("subset", "")),
            str(record.get("sample_id", "")),
        ): record
        for record in records
    }
    candidate_keys = {
        (subset, sample_id)
        for condition, subset, sample_id in index
        if condition == conditions[0]
    }
    complete = set()
    for subset, sample_id in candidate_keys:
        ok = True
        for condition in conditions:
            record = index.get((condition, subset, sample_id))
            if record is None:
                ok = False
                break
            tag = (
                f"{record.get('subset', '')}_{record.get('sample_id', '')}_{condition}_"
                f"{record.get('eagle_target_span_mode', '')}_{record.get('eagle_token_mode', '')}"
            )
            artifact_dir = os.path.join(run_dir, "eagle_artifacts")
            if not (
                os.path.exists(os.path.join(artifact_dir, f"{tag}.json"))
                and os.path.exists(os.path.join(artifact_dir, f"{tag}.npz"))
            ):
                ok = False
                break
        if ok:
            complete.add((subset, sample_id))
    return complete


def _category(student_correct: bool, teacher_correct: bool) -> str:
    stu = "correct" if student_correct else "wrong"
    tea = "correct" if teacher_correct else "wrong"
    return f"stu_{stu}_tea_{tea}"


def _stratified_pick(rows: list[dict], count: int, seed: int, category: str) -> list[dict]:
    by_subset: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_subset[row["subset"]].append(row)
    for subset, subset_rows in by_subset.items():
        subset_rows.sort(key=lambda r: r["sample_id"])
        random.Random(f"{seed}:{category}:{subset}").shuffle(subset_rows)

    if count <= 0:
        count = sum(len(subset_rows) for subset_rows in by_subset.values())
    picked = []
    subsets = sorted(by_subset)
    while len(picked) < count:
        added = False
        for subset in subsets:
            if by_subset[subset]:
                picked.append(by_subset[subset].pop())
                added = True
                if len(picked) == count:
                    break
        if not added:
            break
    return picked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-run-dir", required=True)
    parser.add_argument("--student-run-dir", required=True)
    parser.add_argument("--output", required=True, help="Shared JSON manifest path.")
    parser.add_argument("--condition", default="plain")
    parser.add_argument("--per-category", type=int, default=5, help="Maximum per category; 0 keeps all.")
    parser.add_argument("--subsets", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-judge", action="store_true")
    parser.add_argument("--allow-fewer", action="store_true")
    parser.add_argument(
        "--required-run-dirs",
        nargs="+",
        default=[],
        help="Keep only samples with complete artifacts in every listed run.",
    )
    parser.add_argument("--required-conditions", default="plain,hint,hidden_hint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    teacher_records = _dedupe(load_records(args.teacher_run_dir))
    student_records = _dedupe(load_records(args.student_run_dir))
    if args.use_judge:
        teacher_judged = apply_judge(args.teacher_run_dir, teacher_records)
        student_judged = apply_judge(args.student_run_dir, student_records)
        if teacher_judged != len(teacher_records) or student_judged != len(student_records):
            raise SystemExit(
                "[eagle.pairwise] incomplete LLM judge coverage: "
                f"teacher={teacher_judged}/{len(teacher_records)}, "
                f"student={student_judged}/{len(student_records)}"
            )

    subsets = {canon_subset(x) for x in args.subsets.split(",") if x.strip()} or None
    teacher = _index(teacher_records, args.condition, subsets)
    student = _index(student_records, args.condition, subsets)
    shared_keys = sorted(set(teacher) & set(student))
    required_conditions = [value.strip() for value in args.required_conditions.split(",") if value.strip()]
    if args.required_run_dirs:
        if not required_conditions:
            raise SystemExit("[eagle.pairwise] --required-conditions is empty")
        complete_keys = None
        for run_dir in args.required_run_dirs:
            run_keys = _complete_artifact_keys(run_dir, required_conditions)
            print(f"[eagle.pairwise] complete artifacts {run_dir}: {len(run_keys)} samples")
            complete_keys = run_keys if complete_keys is None else complete_keys & run_keys
        shared_keys = sorted(set(shared_keys) & (complete_keys or set()))
    if not shared_keys:
        raise SystemExit("[eagle.pairwise] no shared teacher/student records found")

    candidates: dict[str, list[dict]] = {category: [] for category in CATEGORIES}
    for subset, sample_id in shared_keys:
        tea_correct = bool(teacher[(subset, sample_id)].get("correct", False))
        stu_correct = bool(student[(subset, sample_id)].get("correct", False))
        category = _category(stu_correct, tea_correct)
        candidates[category].append(
            {
                "subset": subset,
                "sample_id": sample_id,
                "student_correct": stu_correct,
                "teacher_correct": tea_correct,
            }
        )

    selected = {}
    shortages = []
    for category in CATEGORIES:
        rows = _stratified_pick(candidates[category], args.per_category, args.seed, category)
        selected[category] = rows
        if args.per_category > 0 and len(rows) < args.per_category:
            shortages.append(f"{category}={len(rows)}/{args.per_category}")

    if shortages and not args.allow_fewer:
        available = ", ".join(f"{category}={len(candidates[category])}" for category in CATEGORIES)
        raise SystemExit(
            "[eagle.pairwise] not enough cases: "
            + ", ".join(shortages)
            + f". Available shared cases: {available}. Pass --allow-fewer to keep all available."
        )

    manifest = {
        "format": "eagle_pairwise_cases_v1",
        "teacher_run_dir": args.teacher_run_dir,
        "student_run_dir": args.student_run_dir,
        "condition": args.condition,
        "correctness_source": "llm_judge" if args.use_judge else "rule",
        "per_category": args.per_category,
        "shared_record_count": len(shared_keys),
        "required_run_dirs": args.required_run_dirs,
        "required_conditions": required_conditions if args.required_run_dirs else [],
        "candidate_counts": {category: len(candidates[category]) for category in CATEGORIES},
        "categories": selected,
    }
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    counts = ", ".join(f"{category}={len(selected[category])}" for category in CATEGORIES)
    print(f"[eagle.pairwise] wrote {args.output}: {counts}")


if __name__ == "__main__":
    main()
