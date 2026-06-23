"""General multi-benchmark evaluation harness for the OPD project.

A lean, dataset-prompt-agnostic evaluator that reuses the generic helpers in
``vigos.eval_utils`` / ``vigos.eval_benchmarks`` (sample extraction, judge
prompts, scoring) but uses the general OPD eval prompt instead of the ViGOS
``<description>`` format. ViGOS code is reused as a library and left untouched.

Pipeline per source (HF dataset or registered benchmark):
  load samples -> vLLM generate pass@k -> extract \\boxed answers ->
  LLM-judge (OpenAI-compatible) -> pass@k / avg@k -> write jsonl + summary.json

Example:
  MODEL_PATH=runs/opd_qwen25_3b_<run> bash scripts/eval_opd.sh
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )

from vigos.eval_benchmarks import (
    avg_at_k_fields,
    load_benchmark_tasks,
    read_jsonl,
    score_benchmark,
)
from vigos.eval_utils import (
    EvalSample,
    build_judge_messages,
    build_passk_judge_messages,
    extract_model_answer,
    parse_dataset_specs,
    parse_judge_output,
    sample_from_record,
    sanitize_dataset_name,
    vllm_request,
)

from baseline.eval.opd_eval_prompt import (
    GENERAL_PROMPT_DESCRIPTION,
    build_general_eval_prompt,
)
from baseline.opd_data_collator import OPD_DEFAULT_PROMPT_SUFFIX


# --------------------------------------------------------------------------- CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="General OPD multi-benchmark eval.")
    p.add_argument("--model-path", required=True)
    p.add_argument("--model-name", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--datasets", default=None, help="Comma-separated id[@split] list.")
    p.add_argument("--benchmarks", default="", help="e.g. vilp-f,vilp-p,cv-bench (optional).")
    p.add_argument("--default-split", default="test")
    p.add_argument("--limit", type=int, default=None, help="Max samples per source.")
    p.add_argument("--prompt-suffix", default=OPD_DEFAULT_PROMPT_SUFFIX)
    # generation
    p.add_argument("--pass-k", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8, help="Questions per vLLM call.")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--limit-images", type=int, default=16)
    p.add_argument("--dtype", default="auto")
    # judge
    p.add_argument("--skip-judge", action="store_true")
    p.add_argument("--judge-model", default="deepseek-v4-flash")
    p.add_argument("--judge-api-url", default="https://api.deepseek.com")
    p.add_argument("--judge-key-env", default="DEEPSEEK_API_KEY")
    p.add_argument("--judge-workers", type=int, default=64)
    p.add_argument("--judge-max-tokens", type=int, default=4096)
    p.add_argument("--judge-timeout", type=float, default=120.0)
    p.add_argument("--judge-retries", type=int, default=2)
    return p.parse_args()


# -------------------------------------------------------------------- sample IO
def dataset_samples(spec: Any, limit: int | None) -> list[EvalSample]:
    from datasets import load_dataset

    data = load_dataset(spec.path, split=spec.split)
    n = len(data) if limit is None else min(limit, len(data))
    samples = []
    for index in range(n):
        samples.append(sample_from_record(spec.path, index, dict(data[index])))
    return samples


def benchmark_samples(task: Any, limit: int | None) -> list[EvalSample]:
    n = task.total if limit is None else min(limit, task.total)
    return [task.load_sample(index) for index in range(n)]


# ------------------------------------------------------------------- generation
def make_engine(args: argparse.Namespace):
    from vllm import LLM

    kwargs: dict[str, Any] = dict(
        model=args.model_path,
        trust_remote_code=True,
        tokenizer_mode="slow",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": args.limit_images},
        dtype=args.dtype,
        seed=args.seed,
    )
    if args.max_model_len is not None:
        kwargs["max_model_len"] = args.max_model_len
    return LLM(**kwargs)


def make_sampling_params(args: argparse.Namespace):
    from vllm import SamplingParams

    return SamplingParams(
        n=max(1, args.pass_k),
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k if args.top_k > 0 else -1,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )


def generate_records(
    engine,
    processor,
    sampling_params,
    samples: list[EvalSample],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    batch = max(1, args.batch_size)
    for start in range(0, len(samples), batch):
        window = samples[start : start + batch]
        requests = []
        for sample in window:
            prompt = build_general_eval_prompt(
                processor, sample.problem, sample.images, suffix=args.prompt_suffix
            )
            requests.append(vllm_request(prompt, sample.images))
        outputs = engine.generate(requests, sampling_params, use_tqdm=False)
        for sample, output in zip(window, outputs, strict=True):
            attempts = []
            for idx, candidate in enumerate(output.outputs):
                text = candidate.text
                attempts.append(
                    {
                        "attempt_index": idx,
                        "response": text,
                        "extracted_answer": extract_model_answer(text),
                    }
                )
            meta = (sample.raw or {}).get("benchmark_meta")
            records.append(
                {
                    "dataset": sample.dataset,
                    "sample_id": sample.sample_id,
                    "problem": sample.problem,
                    "ground_truth": sample.ground_truth,
                    "response": attempts[0]["response"] if attempts else "",
                    "extracted_answer": attempts[0]["extracted_answer"] if attempts else "",
                    "attempts": attempts,
                    "pass_k_requested": max(1, args.pass_k),
                    "pass_k_returned": len(attempts),
                    "image_metadata": sample.image_metadata,
                    "benchmark_meta": meta,
                }
            )
    return records


# ----------------------------------------------------------------------- judge
def judge_records(records: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    from openai import OpenAI

    api_key = os.environ.get(args.judge_key_env) or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            f"No judge API key found in ${args.judge_key_env} (or $OPENAI_API_KEY). "
            "Use --skip-judge to only generate responses."
        )
    client = OpenAI(base_url=args.judge_api_url, api_key=api_key)

    def judge_one(record: dict[str, Any]) -> dict[str, Any]:
        answers = [a.get("extracted_answer", "") for a in record.get("attempts", [])]
        if not answers:
            answers = [record.get("extracted_answer", "")]
        if len(answers) > 1:
            messages = build_passk_judge_messages(answers, record["ground_truth"], record["problem"])
        else:
            messages = build_judge_messages(answers[0], record["ground_truth"], record["problem"])

        parsed: dict[str, Any] = {}
        error = None
        for attempt in range(args.judge_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=args.judge_model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=args.judge_max_tokens,
                    timeout=args.judge_timeout,
                )
                parsed = parse_judge_output(resp.choices[0].message.content)
                error = None
                break
            except Exception as exc:  # noqa: BLE001 - record and retry
                error = f"{type(exc).__name__}: {exc}"
        verdict = parsed.get("verdict", "incorrect")
        attempt_verdicts = parsed.get("attempt_verdicts")
        if not isinstance(attempt_verdicts, list) or not attempt_verdicts:
            attempt_verdicts = [verdict] * len(answers)
        if len(attempt_verdicts) < len(answers):
            attempt_verdicts = attempt_verdicts + ["incorrect"] * (
                len(answers) - len(attempt_verdicts)
            )
        attempt_verdicts = attempt_verdicts[: len(answers)]
        correct_count = sum(1 for v in attempt_verdicts if v == "correct")
        return {
            "dataset": record["dataset"],
            "sample_id": record["sample_id"],
            "judge_verdict": verdict,
            "judge_attempt_verdicts": attempt_verdicts,
            "judge_attempt_count": len(answers),
            "judge_attempt_correct_count": correct_count,
            "avg_at_k": (correct_count / len(answers)) if answers else None,
            "judge_extracted_answers": answers,
            "judge_reasoning": parsed.get("reasoning", ""),
            "judge_error": error,
            "benchmark_meta": record.get("benchmark_meta"),
        }

    with ThreadPoolExecutor(max_workers=max(1, args.judge_workers)) as pool:
        return list(pool.map(judge_one, records))


# ------------------------------------------------------------------------- main
def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    (output_dir / "responses").mkdir(parents=True, exist_ok=True)
    (output_dir / "judgments").mkdir(parents=True, exist_ok=True)

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=True, use_fast=False
    )
    engine = make_engine(args)
    sampling_params = make_sampling_params(args)

    # Build the work list: HF datasets + optional registered benchmarks.
    sources: list[dict[str, Any]] = []
    for spec in parse_dataset_specs(args.datasets, args.default_split):
        sources.append(
            {
                "kind": "dataset",
                "name": spec.path,
                "stem": spec.safe_name,
                "split": spec.split,
                "samples": dataset_samples(spec, args.limit),
            }
        )
    for task in load_benchmark_tasks(args.benchmarks) if args.benchmarks.strip() else []:
        sources.append(
            {
                "kind": "benchmark",
                "name": task.name,
                "stem": task.response_stem,
                "source": task.source,
                "samples": benchmark_samples(task, args.limit),
            }
        )

    summary: dict[str, Any] = {
        "model_path": args.model_path,
        "model_name": args.model_name or os.path.basename(args.model_path.rstrip("/")),
        "output_dir": str(output_dir),
        "pass_k": max(1, args.pass_k),
        "prompt": GENERAL_PROMPT_DESCRIPTION,
        "prompt_suffix": args.prompt_suffix,
        "generation": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
        },
        "datasets": [],
        "benchmarks": [],
    }

    for source in sources:
        stem = sanitize_dataset_name(source["stem"])
        response_file = output_dir / "responses" / f"{stem}.jsonl"
        judgment_file = output_dir / "judgments" / f"{stem}.jsonl"

        records = generate_records(
            engine, processor, sampling_params, source["samples"], args
        )
        write_jsonl(response_file, records)
        print(f"[{source['name']}] generated {len(records)} responses -> {response_file}")

        if args.skip_judge:
            continue

        judgments = judge_records(records, args)
        write_jsonl(judgment_file, judgments)

        if source["kind"] == "benchmark":
            result = score_benchmark(source["name"], response_file, judgment_file, output_dir)
            result["benchmark"] = source["name"]
            summary["benchmarks"].append(result)
            print(
                f"[{source['name']}] pass@k={result.get('pass_at_k')} "
                f"avg@k={result.get('avg_at_k')}"
            )
        else:
            scores = avg_at_k_fields(read_jsonl(judgment_file))
            scores.update(
                {
                    "dataset": source["name"],
                    "split": source.get("split"),
                    "safe_name": stem,
                    "samples": len(records),
                    "response_file": str(response_file),
                    "judgment_file": str(judgment_file),
                }
            )
            summary["datasets"].append(scores)
            print(
                f"[{source['name']}] pass@k={scores.get('pass_at_k')} "
                f"avg@k={scores.get('avg_at_k')}"
            )

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
