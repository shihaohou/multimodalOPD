"""V*Bench (V*) visual-search multiple-choice evaluation for the OPD project.

V*Bench (Wu & Xie, *V*: Guided Visual Search as a Core Mechanism in Multimodal
LLMs*, CVPR 2024): 191 **high-resolution** images -> 191 multiple-choice
questions in two categories -- ``direct_attributes`` (115: the fine attribute of
a small target object) and ``relative_position`` (76: the spatial relation
between two small objects). Because the discriminative detail occupies a tiny
fraction of a very large image, a model that does not actually *look* (search)
cannot answer it from a global glance -- so V*Bench is a clean probe of
fine-grained visual perception, the very property OPD's ViT-unfreezing is meant
to improve (cf. MMVP).

Dedicated and **deterministic**, with **no LLM judge**: answers are option
letters, graded by exact MCQ matching (boxed letter / "(a)" / option text all
handled), reusing the MMVP matcher. Reuses the OPD unified system prompt and the
generic vLLM generation path from :mod:`baseline.eval.run_opd_eval`, so the model
is evaluated under its training-time prompt. ViGOS code is reused as a library,
untouched.

Pipeline:
  load V*Bench -> vLLM greedy (default) -> \\boxed letter extract ->
  MCQ match -> overall acc + per-category acc -> summary.json

The summary is written in the deterministic-group shape
(``{"benchmarks": {"vstar": {"metrics": {...}}}}``) so
:mod:`baseline.eval.make_report` / ``scripts/eval_opd_multi.sh`` fold it straight
into the methods x benchmarks matrix (column ``vstar (acc)``).

Example:
  MODEL_PATH=runs/opd_qwen25_3b_<run> bash scripts/eval_vstar.sh
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )

from vigos.eval_benchmarks import (
    attempt_stats,
    is_correct_judgment,
    optional_ratio,
    ratio,
)
from vigos.eval_utils import EvalSample, extract_images

from baseline.eval.opd_eval_prompt import GENERAL_PROMPT_DESCRIPTION
from baseline.eval.run_mmvp_eval import match_option_letter, parse_mmvp_options
from baseline.eval.run_opd_eval import (
    generate_records,
    make_engine,
    make_sampling_params,
    write_jsonl,
)

# V*Bench's questions already embed the options "(A) ... (B) ..." inline and end
# with its native "Answer with the option's letter ... directly." instruction. We
# strip that trailing instruction (so it doesn't fight the CoT system prompt) and
# append our own MCQ suffix instead -- identical convention to the MMVP eval, so
# the two visual-perception MCQ benchmarks stay on one prompt format.
VSTAR_PROMPT_SUFFIX = "\nAnswer with the option's letter from the given choices."
VSTAR_CATEGORIES = ("direct_attributes", "relative_position")
_ANSWER_INSTRUCTION_RE = re.compile(
    r"\s*answer with the option'?s? letter.*\Z", re.IGNORECASE | re.DOTALL
)


# --------------------------------------------------------------------------- CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V*Bench visual-search MCQ eval.")
    p.add_argument("--model-path", required=True)
    p.add_argument("--model-name", default=None)
    p.add_argument("--output-dir", required=True)
    # V*Bench source
    p.add_argument(
        "--vstar-repo",
        default="craigwu/vstar_bench",
        help="HuggingFace dataset repo id, OR a local snapshot dir (for offline boxes).",
    )
    p.add_argument(
        "--questions-file",
        default="test_questions.jsonl",
        help="JSONL of questions under the repo root (auto-discovered if absent).",
    )
    p.add_argument(
        "--categories",
        default="",
        help="Comma-separated category filter (default: all). V*Bench categories are "
        "direct_attributes, relative_position.",
    )
    p.add_argument("--limit", type=int, default=None, help="Max questions (smoke test).")
    p.add_argument("--prompt-suffix", default=VSTAR_PROMPT_SUFFIX)
    p.add_argument(
        "--system-prompt",
        default="think",
        help="System-prompt style: think (default) | freecot (no tags) | reason | none, "
        "or a raw string. Must match how the checkpoint was trained.",
    )
    # generation (greedy MCQ by default; bump temperature if you set --pass-k>1)
    p.add_argument("--pass-k", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=0,
                   help="0 = feed all prompts to vLLM in one call (recommended). "
                   ">0 = chunk size (bound host memory).")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=None)
    # V*Bench images are very high-resolution and one per question; allow only 1.
    p.add_argument("--limit-images", type=int, default=1)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--tokenizer-mode", default="auto",
                   help="vLLM tokenizer mode: auto = fast (faster preprocessing); slow = fallback.")
    return p.parse_args()


# ------------------------------------------------------------------ V*Bench loader
def clean_vstar_text(text: Any) -> str:
    """Drop V*Bench's trailing 'Answer with the option's letter ... directly.'."""
    return _ANSWER_INSTRUCTION_RE.sub("", str(text or "")).strip()


def _split_csv(raw: str | None) -> list[str]:
    return [part.strip() for part in str(raw or "").split(",") if part.strip()]


def _read_questions_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_vstar_samples(
    repo_id: str,
    questions_file: str,
    categories: list[str],
    limit: int | None,
) -> list[EvalSample]:
    # Accept either a HuggingFace dataset id OR a local snapshot dir, so the eval
    # runs on offline boxes (HF_HUB_OFFLINE=1): pre-fetch once with
    # `hf download craigwu/vstar_bench --repo-type dataset --local-dir <dir>` and
    # pass --vstar-repo <dir>. V*Bench stores images as loose files referenced by a
    # relative path string in the JSONL (NOT an embedded image feature), so we read
    # the JSONL and resolve each path under the snapshot root -- exactly like MMVP.
    source = Path(repo_id).expanduser()
    if source.is_dir():
        root = source
    else:
        from huggingface_hub import snapshot_download

        root = Path(snapshot_download(repo_id, repo_type="dataset"))

    questions_path = root / questions_file
    if not questions_path.exists():
        candidates = sorted(root.rglob("*.jsonl"))
        if not candidates:
            raise FileNotFoundError(
                f"No questions JSONL ({questions_file}) found under {root}."
            )
        questions_path = next(
            (p for p in candidates if "question" in p.name.casefold()), candidates[0]
        )
    rows = _read_questions_jsonl(questions_path)

    want = {c.casefold() for c in categories} or None
    samples: list[EvalSample] = []
    missing_images = 0
    for index, row in enumerate(rows):
        category = str(row.get("category") or "").strip()
        if want is not None and category.casefold() not in want:
            continue
        problem = clean_vstar_text(row.get("text"))
        correct_letter = str(row.get("label") or "").strip().casefold()
        relative_image = str(row.get("image") or "").strip()
        if not problem or not correct_letter or not relative_image:
            continue
        image_path = root / relative_image
        if not image_path.exists():
            missing_images += 1
            continue
        options = parse_mmvp_options(problem)
        sample_id = str(row.get("question_id", index))
        ground_truth = f"({correct_letter}) {options.get(correct_letter, '')}".strip()
        meta = {
            "benchmark": "vstar",
            "question_id": sample_id,
            "category": category or None,
            "correct_letter": correct_letter,
            "options": options,
        }
        samples.append(
            EvalSample(
                dataset="benchmark/vstar",
                sample_id=sample_id,
                problem=problem,
                ground_truth=ground_truth,
                images=extract_images({"image": str(image_path)}),
                image_metadata=[image_path.name],
                raw={"benchmark_meta": meta},
            )
        )

    if not samples:
        hint = (
            f" ({missing_images} questions had a missing image file under {root} -- "
            "the image folders may not have been downloaded; fetch the FULL snapshot: "
            "hf download craigwu/vstar_bench --repo-type dataset --local-dir <dir>)"
            if missing_images
            else ""
        )
        raise RuntimeError(f"[vstar] loaded 0 usable samples from {questions_path}.{hint}")
    if missing_images:
        print(
            f"[vstar] WARNING: skipped {missing_images} questions with a missing image "
            f"file under {root}."
        )
    if limit is not None:
        samples = samples[:limit]
    return samples


# ------------------------------------------------------------------- MCQ grading
def _attempt_letter(attempt: dict[str, Any], options: dict[str, str]) -> str:
    """Map one attempt to an option letter (boxed answer first, then raw text)."""
    letter = match_option_letter(attempt.get("extracted_answer", ""), options)
    if letter:
        return letter
    return match_option_letter(attempt.get("response", ""), options)


def grade_records_vstar(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministic per-attempt MCQ grading; judgment schema matches the harness."""
    judgments: list[dict[str, Any]] = []
    for record in records:
        meta = record.get("benchmark_meta") or {}
        options = {str(k).casefold(): v for k, v in (meta.get("options") or {}).items()}
        correct_letter = str(meta.get("correct_letter") or "").casefold()
        attempts = record.get("attempts") or [
            {
                "response": record.get("response", ""),
                "extracted_answer": record.get("extracted_answer", ""),
            }
        ]
        predicted = [_attempt_letter(attempt, options) for attempt in attempts]
        verdicts = [
            "correct" if correct_letter and letter == correct_letter else "incorrect"
            for letter in predicted
        ]
        correct = sum(verdict == "correct" for verdict in verdicts)
        judgments.append(
            {
                "dataset": record["dataset"],
                "sample_id": record["sample_id"],
                "judge_verdict": "correct" if correct > 0 else "incorrect",
                "judge_attempt_verdicts": verdicts,
                "judge_attempt_count": len(verdicts),
                "judge_attempt_correct_count": correct,
                "greedy_correct": bool(verdicts) and verdicts[0] == "correct",
                "avg_at_k": (correct / len(verdicts)) if verdicts else None,
                "judge_extracted_answers": predicted,
                "judge_reasoning": "rule-based V*Bench MCQ match (no LLM judge)",
                "judge_error": None,
                "benchmark_meta": meta,
            }
        )
    return judgments


# ----------------------------------------------------------------------- scoring
def _mean(values: list[Any]) -> float | None:
    nums = [v for v in values if isinstance(v, (int, float))]
    return sum(nums) / len(nums) if nums else None


def score_vstar(judgments: list[dict[str, Any]]) -> dict[str, Any]:
    """Overall accuracy (greedy Acc@1) + per-category accuracy (+ pass@k / avg@k)."""
    by_category: dict[str, dict[str, int]] = defaultdict(
        lambda: {"greedy": 0, "passk": 0, "attempt_correct": 0, "attempt_total": 0, "total": 0}
    )
    n = greedy = passk = attempt_correct = attempt_total = 0

    for judgment in judgments:
        meta = judgment.get("benchmark_meta") or {}
        category = str(meta.get("category") or "unknown")
        is_greedy = bool(judgment.get("greedy_correct"))
        is_passk = is_correct_judgment(judgment)
        a_correct, a_total = attempt_stats(judgment)

        n += 1
        greedy += int(is_greedy)
        passk += int(is_passk)
        attempt_correct += a_correct
        attempt_total += a_total

        bucket = by_category[category]
        bucket["total"] += 1
        bucket["greedy"] += int(is_greedy)
        bucket["passk"] += int(is_passk)
        bucket["attempt_correct"] += a_correct
        bucket["attempt_total"] += a_total

    accuracy_by_category = {
        key: ratio(value["greedy"], value["total"])
        for key, value in sorted(by_category.items())
    }
    pass_at_k_by_category = {
        key: ratio(value["passk"], value["total"])
        for key, value in sorted(by_category.items())
    }
    return {
        "benchmark": "vstar",
        "questions": n,
        "metrics": {
            # Headline = overall greedy Acc@1 over all questions (the standard V*
            # number); make_report reads this key for the matrix column.
            "accuracy": ratio(greedy, n),
            "pass_at_k": ratio(passk, n),
            "avg_at_k": optional_ratio(attempt_correct, attempt_total),
            # The macro mean of the two categories (also commonly reported).
            "category_mean_accuracy": _mean(list(accuracy_by_category.values())),
            "accuracy_by_category": accuracy_by_category,
            "pass_at_k_by_category": pass_at_k_by_category,
        },
    }


# ------------------------------------------------------------------------- main
def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    for sub in ("responses", "judgments", "benchmark_scores"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=True, use_fast=False
    )
    engine = make_engine(args)
    sampling_params = make_sampling_params(args)

    samples = load_vstar_samples(
        args.vstar_repo, args.questions_file, _split_csv(args.categories), args.limit
    )
    print(f"[vstar] loaded {len(samples)} questions from {args.vstar_repo}")

    records = generate_records(engine, processor, sampling_params, samples, args)
    response_file = output_dir / "responses" / "vstar.jsonl"
    write_jsonl(response_file, records)

    judgments = grade_records_vstar(records)
    judgment_file = output_dir / "judgments" / "vstar.jsonl"
    write_jsonl(judgment_file, judgments)

    score = score_vstar(judgments)
    score["prompt_suffix"] = args.prompt_suffix
    score["response_file"] = str(response_file)
    score["judgment_file"] = str(judgment_file)
    score_file = output_dir / "benchmark_scores" / "vstar.json"
    score_file.write_text(
        json.dumps(score, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    score["score_file"] = str(score_file)

    metrics = score["metrics"]
    # Deterministic-group summary shape (benchmarks: {name: {metrics: {...}}}) so
    # make_report.py / eval_opd_multi.sh pick "vstar" up into the matrix.
    summary = {
        "model_path": args.model_path,
        "model_name": args.model_name or os.path.basename(args.model_path.rstrip("/")),
        "output_dir": str(output_dir),
        "grader": "rule-mcq (no judge)",
        "prompt": f"{GENERAL_PROMPT_DESCRIPTION} | MCQ letter, rule-graded",
        "prompt_suffix": args.prompt_suffix,
        "pass_k": max(1, args.pass_k),
        "generation": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
        },
        "benchmarks": {"vstar": score},
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    by_cat = " | ".join(
        f"{key}={value:.4f}" for key, value in metrics["accuracy_by_category"].items()
    )
    print(
        f"[vstar] accuracy={metrics['accuracy']:.4f} "
        f"(category_mean={metrics['category_mean_accuracy']:.4f}) over "
        f"{score['questions']} questions  [{by_cat}]"
    )
    print(f"Wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
