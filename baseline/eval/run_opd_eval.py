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

from baseline.eval.grading import attempt_correct
from baseline.eval.opd_eval_prompt import (
    GENERAL_PROMPT_DESCRIPTION,
    build_general_eval_prompt,
)
# (system prompt is baked into baseline.eval.opd_eval_prompt; suffix defaults empty)


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
    p.add_argument("--prompt-suffix", default="")
    # generation
    p.add_argument("--pass-k", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=0,
                   help="0 = feed all prompts to vLLM in one call (recommended; its "
                   "continuous batching saturates the GPU). >0 = chunk size (only to "
                   "bound host memory on very large datasets).")
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
    # grading / judge
    p.add_argument(
        "--grader",
        default="llm",
        choices=["rule", "llm"],
        help="llm = OpenAI-compatible LLM judge (default, same as ViGOS); "
        "rule = mathruler + option/exact match (no API, deterministic/reproducible).",
    )
    p.add_argument("--skip-judge", action="store_true")
    p.add_argument(
        "--judge-only",
        action="store_true",
        help="Skip generation entirely; judge existing responses/*.jsonl (NO GPU). Pair "
        "with a prior --skip-judge run to decouple GPU rollout from the judge — one "
        "controlled --judge-workers pool, not a per-GPU fan-out that swamps the judge.",
    )
    p.add_argument("--judge-model", default="deepseek-v4-flash")
    p.add_argument("--judge-api-url", default="https://api.deepseek.com")
    p.add_argument("--judge-key-env", default="DEEPSEEK_API_KEY")
    p.add_argument("--judge-workers", type=int, default=64)
    p.add_argument("--judge-max-tokens", type=int, default=4096)
    p.add_argument("--judge-timeout", type=float, default=120.0)
    p.add_argument("--judge-retries", type=int, default=2)
    p.add_argument(
        "--judge-extra-body",
        default="",
        help="JSON merged into each judge request's body (OpenAI client extra_body). "
        'E.g. disable Qwen3 thinking: \'{"chat_template_kwargs": {"enable_thinking": false}}\'.',
    )
    return p.parse_args()


# -------------------------------------------------------------------- sample IO
def open_eval_dataset(path: str, split: str):
    """Load a HuggingFace dataset by id OR a LOCAL dir, robust to either on-disk
    layout (``save_to_disk`` arrow, or a hub snapshot of parquet). A local path is
    used as-is, so this works on offline boxes (``HF_HUB_OFFLINE=1``) where only id
    lookups fail — point ``--datasets`` at e.g. ``/data/zli12321/mathvista``.
    Falls back to the only/first split if ``split`` is absent.
    """
    from datasets import load_dataset

    local = Path(path).expanduser()
    if not local.exists():
        return load_dataset(path, split=split)  # treat as a hub id

    def _pick(dataset_dict):
        return dataset_dict[split] if split in dataset_dict else dataset_dict[next(iter(dataset_dict))]

    from datasets import DatasetDict

    try:  # save_to_disk (arrow) dir
        from datasets import load_from_disk

        data = load_from_disk(str(local))
        return _pick(data) if isinstance(data, DatasetDict) else data
    except Exception:
        pass
    try:  # hub snapshot / parquet dir, split present
        return load_dataset(str(local), split=split)
    except Exception:  # split missing -> take what's there
        data = load_dataset(str(local))
        return _pick(data) if isinstance(data, DatasetDict) else data


def dataset_samples(spec: Any, limit: int | None) -> list[EvalSample]:
    data = open_eval_dataset(spec.path, spec.split)
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
    # vLLM v1 launches its EngineCore in a subprocess; if the parent has already
    # touched CUDA, a *forked* child dies with "Cannot re-initialize CUDA in forked
    # subprocess". Force spawn so the child starts from a clean interpreter.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
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
    # Escape hatch for models that crash vLLM's CUDA-graph/torch.compile path with an
    # "illegal memory access" (seen on some finetuned Qwen3-VL checkpoints): run eager.
    if os.environ.get("VLLM_ENFORCE_EAGER", "").strip().lower() in {"1", "true", "yes"}:
        kwargs["enforce_eager"] = True
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
    # Build every request up front and feed vLLM in as FEW generate() calls as
    # possible — its continuous batching then keeps the GPU saturated. The old
    # per-window loop throttled concurrency to batch_size and drained the batch at
    # every boundary (the slowest sequence stalled the next window), which was the
    # main reason eval crawled. batch_size<=0 (default) -> one call with everything;
    # set batch_size>0 only to bound host memory on very large datasets.
    requests = [
        vllm_request(
            build_general_eval_prompt(
                processor, sample.problem, sample.images, suffix=args.prompt_suffix
            ),
            sample.images,
        )
        for sample in samples
    ]
    batch = getattr(args, "batch_size", 0) or 0
    chunk = len(samples) if batch <= 0 else batch
    for start in range(0, len(samples), max(1, chunk)):
        window = samples[start : start + chunk]
        outputs = engine.generate(
            requests[start : start + chunk], sampling_params, use_tqdm=True
        )
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

    extra_body = None
    if args.judge_extra_body.strip():
        try:
            extra_body = json.loads(args.judge_extra_body)
        except json.JSONDecodeError as exc:
            raise ValueError(f"--judge-extra-body must be valid JSON: {exc}") from exc

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
                    extra_body=extra_body,
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


# ------------------------------------------------------------------ rule grading
def grade_records_rule(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministic per-attempt grading (no API), same schema as the LLM judge."""
    judgments: list[dict[str, Any]] = []
    for record in records:
        attempts = record.get("attempts") or []
        texts = [a.get("response", "") for a in attempts] or [record.get("response", "")]
        verdicts = [
            "correct" if attempt_correct(text, record["ground_truth"]) else "incorrect"
            for text in texts
        ]
        correct = sum(v == "correct" for v in verdicts)
        judgments.append(
            {
                "dataset": record["dataset"],
                "sample_id": record["sample_id"],
                "judge_verdict": "correct" if correct > 0 else "incorrect",
                "judge_attempt_verdicts": verdicts,
                "judge_attempt_count": len(verdicts),
                "judge_attempt_correct_count": correct,
                "avg_at_k": (correct / len(verdicts)) if verdicts else None,
                "judge_extracted_answers": [extract_model_answer(t) for t in texts],
                "judge_reasoning": "rule-based (mathruler + option/exact)",
                "judge_error": None,
                "benchmark_meta": record.get("benchmark_meta"),
            }
        )
    return judgments


# ------------------------------------------------------------------------- main
def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def judge_existing_responses(args: argparse.Namespace, output_dir: Path) -> None:
    """Judge previously-saved responses/*.jsonl with NO generation / GPU.

    Decouples the GPU-bound rollout from the network-bound judge: generate once on all
    GPUs with --skip-judge, then judge here in a SINGLE process whose concurrency is
    just --judge-workers (no per-GPU fan-out multiplying it), so the judge can't be
    swamped. Reads the same --datasets/--benchmarks to know which response stems to score.
    """
    sources: list[dict[str, Any]] = []
    for spec in parse_dataset_specs(args.datasets, args.default_split):
        sources.append(
            {"kind": "dataset", "name": spec.path, "stem": spec.safe_name, "split": spec.split}
        )
    for task in load_benchmark_tasks(args.benchmarks) if args.benchmarks.strip() else []:
        sources.append(
            {"kind": "benchmark", "name": task.name, "stem": task.response_stem, "source": task.source}
        )

    summary: dict[str, Any] = {
        "model_path": args.model_path,
        "model_name": args.model_name
        or os.path.basename(str(args.model_path or "").rstrip("/"))
        or "model",
        "output_dir": str(output_dir),
        "pass_k": max(1, args.pass_k),
        "grader": args.grader,
        "prompt": GENERAL_PROMPT_DESCRIPTION,
        "judge_only": True,
        "datasets": [],
        "benchmarks": [],
    }

    for source in sources:
        stem = sanitize_dataset_name(source["stem"])
        response_file = output_dir / "responses" / f"{stem}.jsonl"
        judgment_file = output_dir / "judgments" / f"{stem}.jsonl"
        records = read_jsonl(response_file)
        if not records:
            print(f"[{source['name']}] no responses at {response_file}; skipped")
            continue

        if args.grader == "rule":
            judgments = grade_records_rule(records)
        else:
            judgments = judge_records(records, args)
        write_jsonl(judgment_file, judgments)

        if source["kind"] == "benchmark":
            result = score_benchmark(source["name"], response_file, judgment_file, output_dir)
            result["benchmark"] = source["name"]
            summary["benchmarks"].append(result)
            print(f"[{source['name']}] judged {len(records)} -> pass@k={result.get('pass_at_k')}")
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
            print(f"[{source['name']}] judged {len(records)} -> pass@k={scores.get('pass_at_k')}")

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {output_dir / 'summary.json'}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    (output_dir / "responses").mkdir(parents=True, exist_ok=True)
    (output_dir / "judgments").mkdir(parents=True, exist_ok=True)

    if args.judge_only:
        judge_existing_responses(args, output_dir)
        return

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
        "grader": args.grader,
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

        if args.grader == "rule":
            judgments = grade_records_rule(records)
        else:
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
