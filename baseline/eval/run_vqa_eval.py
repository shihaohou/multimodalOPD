"""Short-answer VLM benchmark eval (POPE / ChartQA / VQAv2) for the OPD project.

Three classic single-image short-answer benchmarks, each with a **canonical,
deterministic** official metric — so, like the MMVP eval, there is **no LLM
judge** (no API key needed) and results are exactly reproducible:

  * POPE    — object-hallucination yes/no probing; headline **F1** (+ accuracy /
              precision / recall / yes-ratio), per category (random/popular/
              adversarial). Li et al., EMNLP 2023.
  * ChartQA — chart question answering; **relaxed accuracy** (numeric answers
              within 5 % relative tolerance, else exact), split by human /
              augmented. Methani et al., WACV 2020.
  * VQAv2   — open-ended VQA; official **VQA soft accuracy** (min(1, agreement/3)
              over the 10 human answers, with official normalization), split by
              answer type. Goyal et al., CVPR 2017.

All three reuse the OPD unified system prompt and the generic vLLM generation
path from :mod:`baseline.eval.run_opd_eval` (so the model is evaluated under its
training-time prompt), and the metric primitives in
:mod:`baseline.eval.vqa_metrics`. ViGOS code is reused as a library, untouched.
One ``--benchmarks`` run loads the vLLM engine once and evaluates each in turn,
writing ``responses/``, ``judgments/``, ``benchmark_scores/`` and a combined
``summary.json``.

Example:
  MODEL_PATH=runs/opd_qwen25_3b_<run> bash scripts/eval_vqa.sh
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )

from vigos.eval_benchmarks import optional_ratio, ratio
from vigos.eval_utils import EvalSample, extract_images

from baseline.eval.opd_eval_prompt import GENERAL_PROMPT_DESCRIPTION
from baseline.eval.run_opd_eval import (
    generate_records,
    make_engine,
    make_sampling_params,
    open_eval_dataset,
    write_jsonl,
)
from baseline.eval.vqa_metrics import (
    pope_label,
    pope_scores,
    relaxed_correctness,
    vqa_accuracy,
)

# Answer-format suffixes appended after the unified system prompt's "\boxed{}"
# rule, so the model boxes a short answer (matches the lmms-eval conventions).
POPE_SUFFIX = "\nAnswer the question using a single word: yes or no."
SHORT_SUFFIX = "\nAnswer the question using a single word or phrase."

ALL_BENCHMARKS = ("pope", "chartqa", "vqav2")


# --------------------------------------------------------------------------- CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="POPE / ChartQA / VQAv2 short-answer eval.")
    p.add_argument("--model-path", required=True)
    p.add_argument("--model-name", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--benchmarks",
        default="pope,chartqa,vqav2",
        help="Comma-separated subset of {pope,chartqa,vqav2}, or 'all'.",
    )
    # per-benchmark sources (HF dataset id OR a local snapshot dir)
    p.add_argument("--pope-repo", default="lmms-lab/POPE")
    p.add_argument("--pope-split", default="test")
    p.add_argument(
        "--pope-category",
        default="",
        help="Optional POPE category filter: random / popular / adversarial.",
    )
    p.add_argument("--chartqa-repo", default="lmms-lab/ChartQA")
    p.add_argument("--chartqa-split", default="test")
    p.add_argument("--vqav2-repo", default="lmms-lab/VQAv2")
    p.add_argument("--vqav2-split", default="validation")
    p.add_argument("--limit", type=int, default=None, help="Max samples per benchmark.")
    p.add_argument(
        "--vqav2-limit",
        type=int,
        default=None,
        help="Max VQAv2 samples (overrides --limit for VQAv2 only; its val set is "
        "~214k, vs a few-k for POPE/ChartQA).",
    )
    p.add_argument(
        "--prompt-suffix",
        default=None,
        help="Override the per-benchmark answer-format suffix (default: per-benchmark).",
    )
    # generation (greedy single-sample by default, the canonical setting)
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
    p.add_argument("--limit-images", type=int, default=4)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--tokenizer-mode", default="auto",
                   help="vLLM tokenizer mode: auto = fast (faster preprocessing); slow = fallback.")
    return p.parse_args()


def parse_benchmark_list(raw: str) -> list[str]:
    value = str(raw or "").strip().lower()
    if value in {"", "all"}:
        return list(ALL_BENCHMARKS)
    names = []
    for part in value.split(","):
        name = part.strip()
        if not name:
            continue
        if name not in ALL_BENCHMARKS:
            raise ValueError(
                f"Unknown benchmark {name!r}; expected a subset of {ALL_BENCHMARKS} or 'all'."
            )
        if name not in names:
            names.append(name)
    return names


# ----------------------------------------------------------------- dataset access
def _open_dataset(source: str, split: str):
    """Load a HuggingFace dataset by id, OR a local snapshot dir (offline boxes).

    Delegates to :func:`baseline.eval.run_opd_eval.open_eval_dataset`, which is
    robust to either local layout (``save_to_disk`` arrow or a hub snapshot of
    parquet) and falls back to the only/first split when ``split`` is absent —
    e.g. ``--pope-repo /data/POPE --pope-split test`` on an offline box.
    """
    return open_eval_dataset(source, split)


def _first(row: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def _require_samples(samples: list[EvalSample], name: str, columns: list[str]) -> None:
    if not samples:
        raise RuntimeError(
            f"[{name}] loaded 0 usable samples. Check the source / field names. "
            f"Dataset columns were: {columns}"
        )


# --------------------------------------------------------------------- POPE loader
def load_pope_samples(args: argparse.Namespace) -> list[EvalSample]:
    data = _open_dataset(args.pope_repo, args.pope_split)
    columns = list(getattr(data, "column_names", []) or [])
    want = str(args.pope_category or "").strip().lower()
    samples: list[EvalSample] = []
    for index in range(len(data)):
        row = dict(data[index])
        category = str(_first(row, ("category", "subset", "type"), "") or "").strip()
        if want and category and category.lower() != want:
            continue
        question = str(_first(row, ("question", "text", "query"), "") or "").strip()
        answer = str(_first(row, ("answer", "label", "gt_answer"), "") or "").strip()
        if not question or not answer:
            continue
        label = pope_label(answer)  # gold yes/no
        meta = {"benchmark": "pope", "category": category or "all", "label": label}
        samples.append(
            EvalSample(
                dataset="benchmark/pope",
                sample_id=str(_first(row, ("question_id", "id"), index)),
                problem=question,
                ground_truth=label,
                images=extract_images(row),
                image_metadata=[],
                raw={"benchmark_meta": meta},
            )
        )
        if args.limit is not None and len(samples) >= args.limit:
            break
    _require_samples(samples, "pope", columns)
    return samples


# ------------------------------------------------------------------ ChartQA loader
def _chartqa_type(value: Any) -> str:
    token = str(value if value is not None else "").strip().lower()
    if token in {"0", "human"}:
        return "human"
    if token in {"1", "augmented", "machine", "aug"}:
        return "augmented"
    return token or "all"


def load_chartqa_samples(args: argparse.Namespace) -> list[EvalSample]:
    data = _open_dataset(args.chartqa_repo, args.chartqa_split)
    columns = list(getattr(data, "column_names", []) or [])
    samples: list[EvalSample] = []
    for index in range(len(data)):
        row = dict(data[index])
        question = str(_first(row, ("question", "query", "problem"), "") or "").strip()
        answer = _first(row, ("answer", "label", "answers"), "")
        if isinstance(answer, (list, tuple)):
            answer = answer[0] if answer else ""
        answer = str(answer).strip()
        if not question or not answer:
            continue
        qtype = _chartqa_type(_first(row, ("type", "human_or_machine", "question_type"), ""))
        meta = {"benchmark": "chartqa", "type": qtype}
        samples.append(
            EvalSample(
                dataset="benchmark/chartqa",
                sample_id=str(_first(row, ("question_id", "id"), index)),
                problem=question,
                ground_truth=answer,
                images=extract_images(row),
                image_metadata=[],
                raw={"benchmark_meta": meta},
            )
        )
        if args.limit is not None and len(samples) >= args.limit:
            break
    _require_samples(samples, "chartqa", columns)
    return samples


# ------------------------------------------------------------------- VQAv2 loader
def _extract_vqa_answers(raw: Any) -> list[str]:
    """Flatten VQAv2 ``answers`` (list of {answer:...} dicts, or plain strings)."""
    out: list[str] = []
    if isinstance(raw, (list, tuple)):
        for item in raw:
            if isinstance(item, dict):
                value = item.get("answer")
                if value is not None:
                    out.append(str(value))
            elif item is not None:
                out.append(str(item))
    elif isinstance(raw, dict) and raw.get("answer") is not None:
        out.append(str(raw["answer"]))
    return out


def load_vqav2_samples(args: argparse.Namespace) -> list[EvalSample]:
    data = _open_dataset(args.vqav2_repo, args.vqav2_split)
    columns = list(getattr(data, "column_names", []) or [])
    limit = args.vqav2_limit if args.vqav2_limit is not None else args.limit
    samples: list[EvalSample] = []
    for index in range(len(data)):
        row = dict(data[index])
        question = str(_first(row, ("question", "query"), "") or "").strip()
        answers = _extract_vqa_answers(_first(row, ("answers", "gt_answers"), None))
        majority = _first(row, ("multiple_choice_answer", "answer"), None)
        if not answers and majority is not None:
            answers = [str(majority)]
        if not question or not answers:
            continue
        display_gt = str(majority) if majority is not None else answers[0]
        answer_type = str(_first(row, ("answer_type",), "") or "").strip() or "all"
        meta = {"benchmark": "vqav2", "answers": answers, "answer_type": answer_type}
        samples.append(
            EvalSample(
                dataset="benchmark/vqav2",
                sample_id=str(_first(row, ("question_id", "id"), index)),
                problem=question,
                ground_truth=display_gt,
                images=extract_images(row),
                image_metadata=[],
                raw={"benchmark_meta": meta},
            )
        )
        if limit is not None and len(samples) >= limit:
            break
    _require_samples(samples, "vqav2", columns)
    return samples


# ----------------------------------------------------------------------- grading
def _attempt_prediction(attempt: dict[str, Any]) -> str:
    """Short answer for one attempt: the boxed/<answer> extraction, else the last
    non-empty line of the raw completion (best-effort for a non-boxing model)."""
    pred = str(attempt.get("extracted_answer") or "").strip()
    if pred:
        return pred
    response = str(attempt.get("response") or "").strip()
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    return lines[-1] if lines else response


def _attempts_of(record: dict[str, Any]) -> list[dict[str, Any]]:
    return record.get("attempts") or [
        {
            "response": record.get("response", ""),
            "extracted_answer": record.get("extracted_answer", ""),
        }
    ]


def _judgment(
    record: dict[str, Any],
    verdicts: list[str],
    predictions: list[str],
    reasoning: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one judgment in the harness schema (matches the LLM-judge output)."""
    correct = sum(verdict == "correct" for verdict in verdicts)
    judgment = {
        "dataset": record["dataset"],
        "sample_id": record["sample_id"],
        "judge_verdict": "correct" if correct > 0 else "incorrect",
        "judge_attempt_verdicts": verdicts,
        "judge_attempt_count": len(verdicts),
        "judge_attempt_correct_count": correct,
        "greedy_correct": bool(verdicts) and verdicts[0] == "correct",
        "avg_at_k": (correct / len(verdicts)) if verdicts else None,
        "judge_extracted_answers": predictions,
        "judge_reasoning": reasoning,
        "judge_error": None,
        "benchmark_meta": record.get("benchmark_meta"),
    }
    if extra:
        judgment.update(extra)
    return judgment


def grade_records_pope(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    judgments = []
    for record in records:
        meta = record.get("benchmark_meta") or {}
        gold = str(meta.get("label") or record.get("ground_truth") or "").strip().lower()
        preds = [pope_label(_attempt_prediction(a)) for a in _attempts_of(record)]
        verdicts = ["correct" if pred == gold else "incorrect" for pred in preds]
        judgments.append(
            _judgment(
                record,
                verdicts,
                preds,
                "rule POPE yes/no mapping (no judge)",
                extra={"greedy_pred": preds[0] if preds else "yes"},
            )
        )
    return judgments


def grade_records_chartqa(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    judgments = []
    for record in records:
        gold = str(record.get("ground_truth") or "")
        preds = [_attempt_prediction(a) for a in _attempts_of(record)]
        verdicts = [
            "correct" if relaxed_correctness(pred, gold) else "incorrect" for pred in preds
        ]
        judgments.append(
            _judgment(record, verdicts, preds, "rule ChartQA relaxed accuracy (no judge)")
        )
    return judgments


def grade_records_vqav2(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    judgments = []
    for record in records:
        meta = record.get("benchmark_meta") or {}
        answers = meta.get("answers") or []
        preds = [_attempt_prediction(a) for a in _attempts_of(record)]
        soft = [vqa_accuracy(pred, answers) for pred in preds]
        verdicts = ["correct" if score > 0 else "incorrect" for score in soft]
        judgments.append(
            _judgment(
                record,
                verdicts,
                preds,
                "rule VQA soft accuracy (no judge)",
                extra={"greedy_vqa_acc": soft[0] if soft else 0.0, "attempt_vqa_acc": soft},
            )
        )
    return judgments


# ----------------------------------------------------------------------- scoring
def score_pope(judgments: list[dict[str, Any]]) -> dict[str, Any]:
    categories: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {"labels": [], "preds": []}
    )
    labels: list[str] = []
    preds: list[str] = []
    for judgment in judgments:
        meta = judgment.get("benchmark_meta") or {}
        category = str(meta.get("category") or "all")
        label = str(meta.get("label") or "").strip().lower()
        pred = str(judgment.get("greedy_pred") or "yes").strip().lower()
        categories[category]["labels"].append(label)
        categories[category]["preds"].append(pred)
        labels.append(label)
        preds.append(pred)
    overall = pope_scores(labels, preds)
    by_category = {
        name: pope_scores(value["labels"], value["preds"])
        for name, value in sorted(categories.items())
    }
    return {
        "benchmark": "pope",
        "questions": len(judgments),
        "metrics": {
            "f1": overall["f1"],
            "accuracy": overall["accuracy"],
            "precision": overall["precision"],
            "recall": overall["recall"],
            "yes_ratio": overall["yes_ratio"],
            "by_category": by_category,
        },
    }


def score_chartqa(judgments: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    correct = 0
    total = 0
    for judgment in judgments:
        meta = judgment.get("benchmark_meta") or {}
        qtype = str(meta.get("type") or "all")
        ok = int(bool(judgment.get("greedy_correct")))
        total += 1
        correct += ok
        by_type[qtype]["total"] += 1
        by_type[qtype]["correct"] += ok
    type_accuracy = {
        name: ratio(value["correct"], value["total"]) for name, value in sorted(by_type.items())
    }
    human, augmented = type_accuracy.get("human"), type_accuracy.get("augmented")
    mean_ha = (human + augmented) / 2 if (human is not None and augmented is not None) else None
    return {
        "benchmark": "chartqa",
        "questions": total,
        "metrics": {
            "relaxed_accuracy": ratio(correct, total),
            "human_augmented_mean": mean_ha,
            "by_type": type_accuracy,
        },
    }


def score_vqav2(judgments: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, dict[str, float]] = defaultdict(lambda: {"acc": 0.0, "total": 0})
    total = 0
    acc_sum = 0.0
    for judgment in judgments:
        meta = judgment.get("benchmark_meta") or {}
        answer_type = str(meta.get("answer_type") or "all")
        acc = float(judgment.get("greedy_vqa_acc") or 0.0)
        total += 1
        acc_sum += acc
        by_type[answer_type]["total"] += 1
        by_type[answer_type]["acc"] += acc
    by_answer_type = {
        name: optional_ratio(value["acc"], value["total"])
        for name, value in sorted(by_type.items())
    }
    return {
        "benchmark": "vqav2",
        "questions": total,
        "metrics": {
            "vqa_accuracy": (acc_sum / total) if total else 0.0,
            "by_answer_type": by_answer_type,
        },
    }


REGISTRY: dict[str, dict[str, Any]] = {
    "pope": {
        "load": load_pope_samples,
        "grade": grade_records_pope,
        "score": score_pope,
        "suffix": POPE_SUFFIX,
        "headline": lambda s: "F1={f1:.4f} acc={accuracy:.4f} yes_ratio={yes_ratio:.4f}".format(
            **s["metrics"]
        ),
    },
    "chartqa": {
        "load": load_chartqa_samples,
        "grade": grade_records_chartqa,
        "score": score_chartqa,
        "suffix": SHORT_SUFFIX,
        "headline": lambda s: "relaxed_acc={relaxed_accuracy:.4f}".format(**s["metrics"]),
    },
    "vqav2": {
        "load": load_vqav2_samples,
        "grade": grade_records_vqav2,
        "score": score_vqav2,
        "suffix": SHORT_SUFFIX,
        "headline": lambda s: "vqa_acc={vqa_accuracy:.4f}".format(**s["metrics"]),
    },
}


# ------------------------------------------------------------------------- main
def main() -> None:
    args = parse_args()
    benchmarks = parse_benchmark_list(args.benchmarks)
    output_dir = Path(args.output_dir)
    for sub in ("responses", "judgments", "benchmark_scores"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=True, use_fast=False
    )
    engine = make_engine(args)
    sampling_params = make_sampling_params(args)

    suffix_override = args.prompt_suffix  # None unless the user forced one for all
    summary: dict[str, Any] = {
        "model_path": args.model_path,
        "model_name": args.model_name or os.path.basename(args.model_path.rstrip("/")),
        "output_dir": str(output_dir),
        "grader": "rule (deterministic, no LLM judge)",
        "prompt": GENERAL_PROMPT_DESCRIPTION,
        "pass_k": max(1, args.pass_k),
        "generation": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
        },
        "benchmarks": {},
    }

    for name in benchmarks:
        spec = REGISTRY[name]
        samples = spec["load"](args)
        args.prompt_suffix = suffix_override if suffix_override is not None else spec["suffix"]
        print(f"[{name}] loaded {len(samples)} samples; suffix={args.prompt_suffix!r}")

        records = generate_records(engine, processor, sampling_params, samples, args)
        response_file = output_dir / "responses" / f"{name}.jsonl"
        write_jsonl(response_file, records)

        judgments = spec["grade"](records)
        judgment_file = output_dir / "judgments" / f"{name}.jsonl"
        write_jsonl(judgment_file, judgments)

        score = spec["score"](judgments)
        score["prompt_suffix"] = args.prompt_suffix
        score["response_file"] = str(response_file)
        score["judgment_file"] = str(judgment_file)
        score_file = output_dir / "benchmark_scores" / f"{name}.json"
        score_file.write_text(
            json.dumps(score, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        score["score_file"] = str(score_file)
        summary["benchmarks"][name] = score
        print(
            f"[{name}] {spec['headline'](score)}  "
            f"({score['questions']} questions) -> {score_file}"
        )

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
