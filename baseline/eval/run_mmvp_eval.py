"""MMVP multiple-choice evaluation for the OPD project.

MMVP (Tong et al., *Eyes Wide Shut?*, CVPR 2024): 150 CLIP-blind image **pairs**
-> 300 binary multiple-choice questions across 9 visual patterns. The headline
metric is **pair accuracy**: a pair scores 1 only if BOTH of its questions are
answered correctly. Because the two images in a pair differ by a single fine
visual attribute (and the questions are otherwise identical), a language prior /
global-context shortcut cannot get a pair right -- you have to actually *see* the
difference. That makes MMVP a clean probe for whether unfreezing the ViT during
OPD genuinely improved general visual perception or catastrophically degraded it.

Dedicated and deterministic, with **no LLM judge**: answers are option letters,
graded by exact MCQ matching (boxed letter / "(a)" / option text all handled).
Reuses the OPD eval prompt (the unified system prompt) and the generic vLLM
generation path from :mod:`baseline.eval.run_opd_eval`, so the model is evaluated
under its training-time prompt. ViGOS code is reused as a library, untouched.

Pipeline:
  load MMVP -> vLLM greedy (default) -> \\boxed letter extract ->
  MCQ match -> single-question acc + PAIR acc (+ per-category) -> summary.json

Example:
  MODEL_PATH=runs/opd_qwen25_3b_<run> bash scripts/eval_mmvp.sh
"""

from __future__ import annotations

import argparse
import csv
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
from baseline.eval.run_opd_eval import (
    generate_records,
    make_engine,
    make_sampling_params,
    write_jsonl,
)

# MMVP's original answer-format instruction (appended after the system prompt's
# "final answer in \\boxed{}" rule, so the model boxes the chosen option letter).
MMVP_PROMPT_SUFFIX = "\nAnswer with the option's letter from the given choices."

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# --------------------------------------------------------------------------- CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MMVP pair-metric multiple-choice eval.")
    p.add_argument("--model-path", required=True)
    p.add_argument("--model-name", default=None)
    p.add_argument("--output-dir", required=True)
    # MMVP source
    p.add_argument(
        "--mmvp-repo",
        default="MMVP/MMVP",
        help="HuggingFace dataset repo id, OR a local snapshot dir (for offline boxes).",
    )
    p.add_argument(
        "--image-dir",
        default=None,
        help="Optional sub-dir under the repo snapshot holding the numbered images "
        "(auto-discovered by default).",
    )
    p.add_argument(
        "--pair-size",
        type=int,
        default=2,
        help="Questions per CLIP-blind pair (MMVP = 2: pair_id = (index-1)//2).",
    )
    p.add_argument("--limit", type=int, default=None, help="Max questions (smoke test).")
    p.add_argument("--prompt-suffix", default=MMVP_PROMPT_SUFFIX)
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
    p.add_argument("--limit-images", type=int, default=2)
    p.add_argument("--dtype", default="auto")
    return p.parse_args()


# ------------------------------------------------------------------- MMVP loader
def _to_int(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read the Questions CSV with lower-cased, stripped header keys."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, str]] = []
        for record in reader:
            rows.append({(k or "").strip().casefold(): v for k, v in record.items()})
    return rows


def parse_mmvp_options(text: Any) -> dict[str, str]:
    """'(a) Open (b) Closed' -> {'a': 'Open', 'b': 'Closed'} (lower-cased keys)."""
    options: dict[str, str] = {}
    for letter, body in re.findall(r"\(([A-Za-z])\)\s*([^()]*)", str(text or "")):
        options[letter.casefold()] = body.strip()
    return options


def parse_correct_letter(text: Any) -> str:
    """'(a)' / 'A' / 'a)' -> 'a'."""
    match = re.search(r"[A-Za-z]", str(text or ""))
    return match.group(0).casefold() if match else ""


def format_mmvp_problem(question: str, options: dict[str, str]) -> str:
    if options:
        option_text = " ".join(f"({letter}) {body}" for letter, body in options.items())
        return f"{question}\n{option_text}"
    return question


def _discover_mmvp_files(
    root: Path, image_dir: str | None
) -> tuple[Path, dict[int, Path]]:
    csv_candidates = sorted(root.rglob("*.csv"))
    if not csv_candidates:
        raise FileNotFoundError(f"No CSV (expected Questions.csv) found under {root}.")
    csv_path = next(
        (p for p in csv_candidates if "question" in p.name.casefold()), csv_candidates[0]
    )

    search_root = (root / image_dir) if image_dir else root
    if not search_root.exists():
        raise FileNotFoundError(f"--image-dir {search_root} does not exist.")
    image_map: dict[int, Path] = {}
    for path in search_root.rglob("*"):
        if path.suffix.casefold() in _IMAGE_EXTS and path.stem.isdigit():
            image_map[int(path.stem)] = path
    if not image_map:
        raise FileNotFoundError(
            f"No integer-named images (1.jpg ...) found under {search_root}."
        )
    return csv_path, image_map


def load_mmvp_samples(
    repo_id: str,
    pair_size: int,
    image_dir: str | None,
    limit: int | None,
) -> list[EvalSample]:
    # Accept either a HuggingFace dataset id OR a local snapshot dir, so the eval
    # runs on offline boxes (HF_HUB_OFFLINE=1): pre-fetch once with
    # `hf download MMVP/MMVP --repo-type dataset --local-dir <dir>` and pass --mmvp-repo <dir>.
    source = Path(repo_id).expanduser()
    if source.is_dir():
        root = source
    else:
        from huggingface_hub import snapshot_download

        root = Path(snapshot_download(repo_id, repo_type="dataset"))
    csv_path, image_map = _discover_mmvp_files(root, image_dir)
    rows = _read_csv_rows(csv_path)

    pair_size = max(1, pair_size)
    samples: list[EvalSample] = []
    for row in rows:
        index = _to_int(row.get("index"))
        question = str(row.get("question") or "").strip()
        if index is None or not question:
            continue
        image_path = image_map.get(index)
        if image_path is None:
            continue
        options = parse_mmvp_options(row.get("options"))
        correct_letter = parse_correct_letter(row.get("correct answer") or row.get("answer"))
        category = str(row.get("category") or row.get("type") or "").strip() or None
        pair_id = (index - 1) // pair_size
        ground_truth = f"({correct_letter}) {options.get(correct_letter, '')}".strip()
        meta = {
            "benchmark": "mmvp",
            "question_index": index,
            "pair_id": pair_id,
            "correct_letter": correct_letter,
            "options": options,
            "category": category,
        }
        samples.append(
            EvalSample(
                dataset="benchmark/mmvp",
                sample_id=str(index),
                problem=format_mmvp_problem(question, options),
                ground_truth=ground_truth,
                images=extract_images({"image": str(image_path)}),
                image_metadata=[image_path.name],
                raw={"benchmark_meta": meta},
            )
        )

    samples.sort(key=lambda sample: int(sample.sample_id))
    if limit is not None:
        samples = samples[:limit]
    return samples


# ------------------------------------------------------------------- MCQ grading
def _strip_answer_prefix(text: str) -> str:
    return re.sub(
        r"^\s*(the\s+)?(final\s+)?answer\s*(is|:|：)?\s*",
        "",
        str(text or "").strip(),
        flags=re.IGNORECASE,
    )


def _clean(text: Any) -> str:
    cleaned = str(text or "").strip().strip("`'\"().").strip()
    return " ".join(cleaned.casefold().split())


def match_option_letter(prediction: Any, options: dict[str, str]) -> str:
    """Map a free-form prediction to an option letter, or '' if undecidable.

    Handles, in order: a bare/parenthesised letter ('a', '(a)', 'A.'), an
    explicit '(x)' anywhere, an 'x)'/'x.' option marker, an exact option-text
    match, then a *unique* short option-text substring.
    """
    raw = _strip_answer_prefix(prediction)
    if not raw:
        return ""
    low = raw.casefold()
    valid = set(options)

    # 1) pure letter, optionally parenthesised / punctuated.
    pure = re.fullmatch(r"\(?\s*([a-z])\s*[\).:]?\s*", low)
    if pure and pure.group(1) in valid:
        return pure.group(1)

    # 2) explicit "(x)" anywhere -> last valid such letter.
    parenthesised = [p for p in re.findall(r"\(([a-z])\)", low) if p in valid]
    if parenthesised:
        return parenthesised[-1]

    # 3) leading "x)" / "x." / "x:" option marker.
    marker = re.match(r"\s*([a-z])\s*[\).:]\s+", low)
    if marker and marker.group(1) in valid:
        return marker.group(1)

    # 4) exact option-text match.
    cleaned = _clean(prediction)
    for letter, body in options.items():
        if cleaned and cleaned == _clean(body):
            return letter

    # 5) unique option-text substring (only for short predictions, to avoid
    # spuriously matching an option word buried in a long sentence).
    if len(cleaned) <= 64:
        hits = {
            letter
            for letter, body in options.items()
            if _clean(body) and _clean(body) in cleaned
        }
        if len(hits) == 1:
            return next(iter(hits))
    return ""


def _attempt_letter(attempt: dict[str, Any], options: dict[str, str]) -> str:
    letter = match_option_letter(attempt.get("extracted_answer", ""), options)
    if letter:
        return letter
    # Fall back to scanning the raw completion (catches a trailing "(x)").
    return match_option_letter(attempt.get("response", ""), options)


def grade_records_mmvp(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministic per-attempt MCQ grading; judgment schema matches the harness."""
    judgments: list[dict[str, Any]] = []
    for record in records:
        meta = record.get("benchmark_meta") or {}
        options = {
            str(k).casefold(): v for k, v in (meta.get("options") or {}).items()
        }
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
                "judge_reasoning": "rule-based MMVP MCQ match (no LLM judge)",
                "judge_error": None,
                "benchmark_meta": meta,
            }
        )
    return judgments


# ------------------------------------------------------------------- pair scoring
def score_mmvp(judgments: list[dict[str, Any]]) -> dict[str, Any]:
    """Single-question accuracy + the headline pair accuracy (both-correct)."""
    by_pair: dict[Any, dict[Any, dict[str, bool]]] = defaultdict(dict)
    pair_category: dict[Any, str] = {}
    question_category: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})

    n_questions = 0
    q_greedy = 0
    q_passk = 0
    attempt_correct = 0
    attempt_total = 0

    for judgment in judgments:
        meta = judgment.get("benchmark_meta") or {}
        pair_id = meta.get("pair_id")
        question_index = meta.get("question_index")
        greedy = bool(judgment.get("greedy_correct"))
        passk = is_correct_judgment(judgment)
        a_correct, a_total = attempt_stats(judgment)

        n_questions += 1
        q_greedy += int(greedy)
        q_passk += int(passk)
        attempt_correct += a_correct
        attempt_total += a_total
        by_pair[pair_id][question_index] = {"greedy": greedy, "passk": passk}

        category = meta.get("category")
        if category:
            pair_category.setdefault(pair_id, str(category))
            question_category[str(category)]["total"] += 1
            question_category[str(category)]["correct"] += int(greedy)

    n_pairs = 0
    pair_greedy = 0
    pair_passk = 0
    incomplete_pairs = 0
    pair_cat_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})

    for pair_id, questions in by_pair.items():
        n_pairs += 1
        complete = len(questions) >= 2
        if not complete:
            incomplete_pairs += 1
        greedy = complete and all(q["greedy"] for q in questions.values())
        passk = complete and all(q["passk"] for q in questions.values())
        pair_greedy += int(greedy)
        pair_passk += int(passk)
        category = pair_category.get(pair_id)
        if category:
            pair_cat_counts[category]["total"] += 1
            pair_cat_counts[category]["correct"] += int(greedy)

    def by_category(counts: dict[str, dict[str, int]]) -> dict[str, float] | None:
        if not counts:
            return None
        return {
            key: ratio(value["correct"], value["total"])
            for key, value in sorted(counts.items())
        }

    return {
        "benchmark": "mmvp",
        "questions": n_questions,
        "pairs": n_pairs,
        "incomplete_pairs": incomplete_pairs,
        "metrics": {
            "pair_accuracy": ratio(pair_greedy, n_pairs),
            "pair_pass_at_k": ratio(pair_passk, n_pairs),
            "question_accuracy": ratio(q_greedy, n_questions),
            "question_pass_at_k": ratio(q_passk, n_questions),
            "question_avg_at_k": optional_ratio(attempt_correct, attempt_total),
            "pair_accuracy_by_category": by_category(pair_cat_counts),
            "question_accuracy_by_category": by_category(question_category),
        },
    }


# ------------------------------------------------------------------------- main
def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    (output_dir / "responses").mkdir(parents=True, exist_ok=True)
    (output_dir / "judgments").mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark_scores").mkdir(parents=True, exist_ok=True)

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=True, use_fast=False
    )
    engine = make_engine(args)
    sampling_params = make_sampling_params(args)

    samples = load_mmvp_samples(
        args.mmvp_repo, args.pair_size, args.image_dir, args.limit
    )
    print(f"[mmvp] loaded {len(samples)} questions from {args.mmvp_repo}")

    records = generate_records(engine, processor, sampling_params, samples, args)
    response_file = output_dir / "responses" / "mmvp.jsonl"
    write_jsonl(response_file, records)

    judgments = grade_records_mmvp(records)
    judgment_file = output_dir / "judgments" / "mmvp.jsonl"
    write_jsonl(judgment_file, judgments)

    score = score_mmvp(judgments)
    score_file = output_dir / "benchmark_scores" / "mmvp.json"
    score_file.write_text(
        json.dumps(score, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    metrics = score["metrics"]
    summary = {
        "model_path": args.model_path,
        "model_name": args.model_name or os.path.basename(args.model_path.rstrip("/")),
        "output_dir": str(output_dir),
        "benchmark": "mmvp",
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
        "questions": score["questions"],
        "pairs": score["pairs"],
        "incomplete_pairs": score["incomplete_pairs"],
        "metrics": metrics,
        "response_file": str(response_file),
        "judgment_file": str(judgment_file),
        "score_file": str(score_file),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(
        f"[mmvp] pair_accuracy={metrics['pair_accuracy']:.4f} "
        f"(pass@k {metrics['pair_pass_at_k']:.4f}) | "
        f"question_accuracy={metrics['question_accuracy']:.4f} "
        f"over {score['questions']} questions / {score['pairs']} pairs "
        f"({score['incomplete_pairs']} incomplete)"
    )
    print(f"Wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
