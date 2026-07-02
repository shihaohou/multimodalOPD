"""Fast lmms-eval-aligned benchmark runner.

This runner keeps the OPD project's fast vLLM inference path, but takes the
benchmark definition from a local lmms-eval checkout:

* dataset path/name/split
* doc_to_text / doc_to_visual prompt construction
* process_results and aggregation metrics

It intentionally supports the generate-until style VLM tasks used by the OPD
benchmark suite. Unsupported task output types fail early instead of silently
falling back to a non-lmms metric.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )

from baseline.eval.lmms_eval_bridge import (  # noqa: E402
    _collect_entry,
    resolve_benchmarks,
    split_benchmarks,
)
from baseline.eval.opd_eval_prompt import build_general_eval_prompt  # noqa: E402
from baseline.eval.run_opd_eval import make_engine  # noqa: E402
from baseline.opd_data_collator import resolve_opd_system_prompt  # noqa: E402
from vigos.eval_utils import vllm_request  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--model-name", default="")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--benchmarks", default="standard")
    p.add_argument("--lmms-eval-dir", default=os.environ.get("LMMS_EVAL_DIR", ""))
    p.add_argument(
        "--lmms-model-name",
        default=os.environ.get("LMMS_MODEL_NAME", "qwen3_vl"),
        help="Task-specific prompt branch to use inside lmms-eval YAML, e.g. qwen3_vl.",
    )
    p.add_argument("--prompt-mode", choices=["lmms", "opd"], default="lmms")
    p.add_argument("--opd-prompt-style", default=os.environ.get("OPD_PROMPT_STYLE", "think"))
    p.add_argument(
        "--opd-prompt-suffix",
        default=os.environ.get("OPD_PROMPT_SUFFIX", ""),
        help="Only used with --prompt-mode opd. Empty matches scripts/eval_opd.sh.",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=None, help="Override task max_new_tokens.")
    p.add_argument("--temperature", type=float, default=None, help="Override task temperature.")
    p.add_argument("--top-p", type=float, default=None, help="Override task top_p.")
    p.add_argument("--top-k", type=int, default=None, help="Override task top_k.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--limit-images", type=int, default=16)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--tokenizer-mode", default="auto")
    p.add_argument(
        "--skip-score",
        action="store_true",
        help="Generate and save responses only. No lmms-eval process_results/aggregation.",
    )
    p.add_argument(
        "--score-only",
        action="store_true",
        help="Read saved responses and run lmms-eval process_results/aggregation only.",
    )
    p.add_argument(
        "--judge-workers",
        type=int,
        default=int(os.environ.get("JUDGE_WORKERS", "1")),
        help="Concurrent workers for lmms-eval process_results during scoring.",
    )
    p.add_argument(
        "--judge-extra-body",
        default=os.environ.get("JUDGE_EXTRA_BODY", ""),
        help=(
            "JSON merged into each OpenAI-compatible judge request body. "
            'For Qwen3 thinking judges, use: {"chat_template_kwargs":{"enable_thinking":false}}'
        ),
    )
    return p.parse_args()


def add_lmms_eval_to_path(path: str) -> str:
    lmms_dir = path or "/Users/houshihao/project/code/lmms-eval-main"
    lmms_dir = os.path.abspath(os.path.expanduser(lmms_dir))
    if not os.path.isdir(lmms_dir):
        raise SystemExit(
            f"lmms-eval checkout not found: {lmms_dir}. Set LMMS_EVAL_DIR or --lmms-eval-dir."
        )
    if lmms_dir not in sys.path:
        sys.path.insert(0, lmms_dir)
    return lmms_dir


def normalize_lmms_api_type() -> None:
    """Avoid lmms-eval MathVerse import crashes from unsupported API_TYPE values."""
    api_type = os.environ.get("API_TYPE", "openai").strip().lower()
    os.environ["API_TYPE"] = api_type if api_type in {"openai", "azure"} else "openai"
    judge_model = resolve_judge_model()
    if judge_model:
        os.environ.setdefault("MODEL_VERSION", judge_model)
        os.environ.setdefault("JUDGE_MODEL", judge_model)


def resolve_judge_model() -> str:
    for name in ("MODEL_VERSION", "JUDGE_MODEL", "OPENAI_API_MODEL"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def patch_lmms_openai_provider() -> None:
    """Let one OPENAI_API_URL work for both lmms-eval judge call styles.

    Some task utilities post directly to OPENAI_API_URL and therefore need the
    full /v1/chat/completions endpoint. The generic lmms-eval OpenAIProvider uses
    the OpenAI SDK, where that same value would be treated as base_url and get
    /chat/completions appended again. If the user provides a full endpoint, force
    that provider down its requests path, which posts to the URL verbatim.
    """
    try:
        from lmms_eval.llm_judge.providers import openai as openai_provider
    except Exception:
        return

    original_init = openai_provider.OpenAIProvider.__init__
    if getattr(original_init, "_opd_patched", False):
        return

    def patched_init(self, config=None):
        original_init(self, config)
        api_url = str(getattr(self, "api_url", "") or "").rstrip("/")
        if api_url.endswith("/chat/completions"):
            self.use_client = False

    patched_init._opd_patched = True
    openai_provider.OpenAIProvider.__init__ = patched_init


def parse_judge_extra_body(raw: str) -> dict[str, Any]:
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JUDGE_EXTRA_BODY JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit("JUDGE_EXTRA_BODY must be a JSON object.")
    return value


def _merge_dicts(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _judge_endpoints() -> set[str]:
    urls = {
        os.environ.get("OPENAI_API_URL", ""),
        os.environ.get("LLM_JUDGE_URL", ""),
        os.environ.get("JUDGE_API_URL", ""),
    }
    return {url.split("?", 1)[0].rstrip("/") for url in urls if url.strip()}


def _is_judge_post(url: str, endpoints: set[str]) -> bool:
    normalized = str(url).split("?", 1)[0].rstrip("/")
    if normalized in endpoints:
        return True
    # Some lmms-eval providers construct Azure/OpenAI-compatible endpoints after
    # import. Restrict the broad fallback to chat-completions style judge calls.
    return normalized.endswith("/chat/completions") and bool(endpoints)


def patch_judge_request_body(extra_body: dict[str, Any], judge_model: str = "") -> None:
    """Normalize lmms-eval judge request bodies.

    lmms-eval tasks are not consistent about judge transport: some use the
    llm_judge provider, some call requests.post directly, and a few call the
    OpenAI SDK directly. Patch both request paths so one env knob covers
    extra_body and hard-coded task defaults such as HallusionBench's gpt-4.
    """
    if not extra_body and not judge_model:
        return

    try:
        import requests
    except Exception:
        requests = None

    if requests is not None:
        original_post = requests.post
        if not getattr(original_post, "_opd_extra_body_patched", False):
            endpoints = _judge_endpoints()

            def patched_post(url, *args, **kwargs):
                payload = kwargs.get("json")
                if isinstance(payload, dict) and _is_judge_post(str(url), endpoints):
                    payload = _merge_dicts(payload, extra_body)
                    if judge_model:
                        payload["model"] = judge_model
                    kwargs["json"] = payload
                return original_post(url, *args, **kwargs)

            patched_post._opd_extra_body_patched = True
            requests.post = patched_post

    try:
        from openai.resources.chat.completions import Completions
    except Exception:
        return

    original_create = Completions.create
    if getattr(original_create, "_opd_extra_body_patched", False):
        return

    def patched_create(self, *args, **kwargs):
        current = kwargs.get("extra_body")
        if isinstance(current, dict):
            kwargs["extra_body"] = _merge_dicts(current, extra_body)
        elif current is None:
            kwargs["extra_body"] = dict(extra_body)
        if judge_model:
            kwargs["model"] = judge_model
        return original_create(self, *args, **kwargs)

    patched_create._opd_extra_body_patched = True
    Completions.create = patched_create


def flatten_tasks(task_dict: dict[Any, Any]) -> Iterable[tuple[str, Any]]:
    for name, task in task_dict.items():
        if isinstance(task, dict):
            yield from flatten_tasks(task)
        elif isinstance(task, tuple):
            _, inner = task
            if inner is not None:
                yield getattr(inner, "task_name", str(name)), inner
        elif task is not None:
            yield getattr(task, "task_name", str(name)), task


def eval_docs(task: Any):
    if task.has_test_docs():
        return task.test_docs(), task.get_config("test_split")
    if task.has_validation_docs():
        return task.validation_docs(), task.get_config("validation_split")
    raise ValueError(f"{task.task_name} has no test/validation docs")


def task_sampling_params(task: Any, args: argparse.Namespace):
    from vllm import SamplingParams

    gen = dict(task.get_config("generation_kwargs") or {})
    max_tokens = args.max_tokens if args.max_tokens is not None else gen.get("max_new_tokens", 1024)
    temperature = args.temperature if args.temperature is not None else float(gen.get("temperature", 0.0) or 0.0)
    top_p = args.top_p if args.top_p is not None else float(gen.get("top_p", 1.0) or 1.0)
    top_k_raw = args.top_k if args.top_k is not None else gen.get("top_k", -1)
    try:
        top_k = int(top_k_raw)
    except Exception:
        top_k = -1
    stop = gen.get("until")
    if isinstance(stop, str):
        stop = [stop]
    if isinstance(stop, list):
        # lmms-eval injects the fewshot delimiter "\n\n" as a default `until`
        # when a task omits explicit stop strings. Qwen3-VL's lmms-eval model
        # removes that default before post-processing; using it as a vLLM hard
        # stop truncates CoT-style answers after the first paragraph.
        stop = [s for s in stop if s and s != "\n\n"]
        if not stop:
            stop = None
    else:
        stop = None
    return SamplingParams(
        n=1,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k if top_k and top_k > 0 else -1,
        max_tokens=int(max_tokens),
        seed=args.seed,
        stop=stop,
    )


def build_prompt(processor: Any, task_text: str, images: list[Any], args: argparse.Namespace) -> str:
    if args.prompt_mode == "opd":
        system_prompt = resolve_opd_system_prompt(args.opd_prompt_style)
        return build_general_eval_prompt(
            processor,
            task_text,
            images,
            system_prompt=system_prompt,
            suffix=args.opd_prompt_suffix,
        )
    return build_general_eval_prompt(
        processor,
        task_text,
        images,
        system_prompt="",
        suffix="",
    )


def _flatten_scalars(value: Any, prefix: str = "") -> dict[str, float | int | str | None]:
    if isinstance(value, dict):
        flat: dict[str, float | int | str | None] = {}
        for key, item in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_flatten_scalars(item, child_prefix))
        return flat
    if isinstance(value, (int, float, str)) or value is None:
        return {prefix: value}
    return {prefix: str(value)}


def _common_score(value: Any) -> float | int | None:
    if isinstance(value, (int, float)):
        return value
    if not isinstance(value, dict):
        return None
    candidates = (
        ("average", "accuracy"),
        ("overall", "accuracy"),
        ("average",),
        ("accuracy",),
        ("acc",),
        ("score",),
    )
    for path in candidates:
        item = value
        for key in path:
            if not isinstance(item, dict) or key not in item:
                item = None
                break
            item = item[key]
        if isinstance(item, (int, float)):
            return item
    return None


def aggregate_task(task: Any, metric_values: dict[str, list[Any]]) -> dict[str, float | int | str | None]:
    out: dict[str, float | int | str | None] = {}
    for metric, agg_fn in task.aggregation().items():
        values = metric_values.get(metric)
        if not values:
            continue
        if "args" in inspect.signature(agg_fn).parameters:
            value = agg_fn(values, args=task.args)
        else:
            value = agg_fn(values)
        if isinstance(value, dict):
            common = _common_score(value)
            if common is not None:
                out[metric] = common
            for key, item in _flatten_scalars(value).items():
                out[f"{metric}.{key}"] = item
        elif isinstance(value, (int, float, str)) or value is None:
            out[metric] = value
        else:
            out[metric] = str(value)
    return out


def _score_one_record(task: Any, docs: Any, record: dict[str, Any], split: str) -> dict[str, Any]:
    doc_id = int(record["doc_id"])
    doc = docs[doc_id]
    response = str(record.get("response") or "")
    per_doc = task.process_results(doc, [response])
    out = dict(record)
    out.update(
        {
            "task": out.get("task") or task.task_name,
            "split": out.get("split") or split,
            "target": task.doc_to_target(doc),
            "process_results": per_doc,
        }
    )
    return out


def score_records(task: Any, docs: Any, records: list[dict[str, Any]], split: str, workers: int = 1) -> tuple[dict[str, list[Any]], list[dict[str, Any]]]:
    metric_values: dict[str, list[Any]] = {}
    workers = max(1, int(workers or 1))
    if workers == 1:
        scored = [
            _score_one_record(task, docs, record, split)
            for record in tqdm(records, desc=f"score {task.task_name}")
        ]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            scored = list(
                tqdm(
                    pool.map(lambda record: _score_one_record(task, docs, record, split), records),
                    total=len(records),
                    desc=f"score {task.task_name} x{workers}",
                )
            )

    for out in scored:
        per_doc = out["process_results"]
        for metric, value in per_doc.items():
            metric_values.setdefault(metric, []).append(value)
    return metric_values, scored


def load_response_records(path: Path, limit: int | None) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Missing saved responses for score-only run: {path}")
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if limit is not None:
        records = records[:limit]
    return records


def main() -> None:
    args = parse_args()
    if args.skip_score and args.score_only:
        raise SystemExit("--skip-score and --score-only are mutually exclusive.")
    normalize_lmms_api_type()
    lmms_dir = add_lmms_eval_to_path(args.lmms_eval_dir)
    judge_extra_body = parse_judge_extra_body(args.judge_extra_body)
    judge_model = resolve_judge_model()
    patch_lmms_openai_provider()
    patch_judge_request_body(judge_extra_body, judge_model)

    from lmms_eval.tasks import TaskManager, get_task_dict

    output_dir = Path(args.output_dir)
    responses_dir = output_dir / "responses"
    metrics_dir = output_dir / "metrics"
    scored_dir = output_dir / "scored"
    artifacts_dir = output_dir / "lmms_artifacts"
    responses_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    scored_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    specs = resolve_benchmarks(split_benchmarks(args.benchmarks))
    requested_tasks: list[str] = []
    for spec in specs:
        for task in spec.tasks:
            if task not in requested_tasks:
                requested_tasks.append(task)
    task_map = {
        "requested": split_benchmarks(args.benchmarks),
        "tasks": requested_tasks,
        "benchmarks": [asdict(spec) for spec in specs],
    }
    (output_dir / "lmms_task_map.json").write_text(
        json.dumps(task_map, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    task_manager = TaskManager("INFO", model_name=args.lmms_model_name)
    matched_tasks = task_manager.match_tasks(requested_tasks)
    missing = [task for task in requested_tasks if task not in matched_tasks]
    if missing:
        raise SystemExit(f"lmms-eval tasks not found: {', '.join(missing)}")
    task_dict = get_task_dict(matched_tasks, task_manager=task_manager, task_type="simple")
    tasks = list(flatten_tasks(task_dict))
    if not tasks:
        raise SystemExit("No lmms-eval subtasks loaded.")

    processor = None
    engine = None
    if not args.score_only:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
        engine = make_engine(args)

    raw_task_metrics: dict[str, dict[str, Any]] = {}
    raw_task_info: dict[str, Any] = {}
    model_name = args.model_name or os.path.basename(args.model_path.rstrip("/"))
    lmms_args = SimpleNamespace(
        output_path=str(artifacts_dir),
        model=model_name,
        model_args=f"pretrained={args.model_path}",
    )

    for task_name, task in tasks:
        if task.OUTPUT_TYPE != "generate_until":
            raise SystemExit(
                f"{task_name} output_type={task.OUTPUT_TYPE!r} is not supported by "
                "the fast aligned runner yet."
            )
        task.args = lmms_args
        docs, split = eval_docs(task)
        total = len(docs) if args.limit is None else min(args.limit, len(docs))
        records: list[dict[str, Any]] = []
        response_path = responses_dir / f"{task_name}.jsonl"

        if args.score_only:
            records = load_response_records(response_path, args.limit)
            total = len(records)
        else:
            assert processor is not None and engine is not None
            sampling_params = task_sampling_params(task, args)
            requests: list[dict[str, Any]] = []
            request_docs: list[tuple[int, Any, str, list[Any], Any]] = []
            for doc_id in tqdm(range(total), desc=f"build {task_name}"):
                doc = docs[doc_id]
                text = task.doc_to_text(doc)
                visuals = task.doc_to_visual(doc) or []
                prompt = build_prompt(processor, text, visuals, args)
                requests.append(vllm_request(prompt, visuals))
                request_docs.append((doc_id, doc, text, visuals, prompt))

            batch = args.batch_size or 0
            chunk = len(requests) if batch <= 0 else batch
            outputs = []
            for start in range(0, len(requests), max(1, chunk)):
                outputs.extend(
                    engine.generate(
                        requests[start : start + chunk],
                        sampling_params,
                        use_tqdm=True,
                    )
                )

            for (doc_id, _doc, text, visuals, prompt), output in zip(request_docs, outputs, strict=True):
                response = output.outputs[0].text if output.outputs else ""
                records.append(
                    {
                        "task": task_name,
                        "split": split,
                        "doc_id": doc_id,
                        "prompt_mode": args.prompt_mode,
                        "lmms_doc_to_text": text,
                        "prompt": prompt,
                        "response": response,
                        "num_images": len(visuals),
                    }
                )

            with response_path.open("w", encoding="utf-8") as fh:
                for record in records:
                    fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

        if args.skip_score:
            raw_task_info[task_name] = {
                "split": split,
                "output_type": task.OUTPUT_TYPE,
                "num_samples": total,
                "judge_workers": 0,
                "judge_extra_body": {},
                "generation_kwargs": task.get_config("generation_kwargs"),
                "responses": str(response_path),
            }
            print(f"[{task_name}] wrote responses={response_path} samples={total}")
            continue

        metric_values, scored = score_records(task, docs, records, split, args.judge_workers)
        scored_path = scored_dir / f"{task_name}.jsonl"
        with scored_path.open("w", encoding="utf-8") as fh:
            for record in scored:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

        task_metrics = aggregate_task(task, metric_values)
        raw_task_metrics[task_name] = task_metrics
        raw_task_info[task_name] = {
            "split": split,
            "output_type": task.OUTPUT_TYPE,
            "num_samples": total,
            "judge_workers": args.judge_workers,
            "judge_extra_body": judge_extra_body,
            "generation_kwargs": task.get_config("generation_kwargs"),
            "responses": str(response_path),
            "scored": str(scored_path),
            "metrics": task_metrics,
        }
        (metrics_dir / f"{task_name}.json").write_text(
            json.dumps(raw_task_info[task_name], indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        print(f"[{task_name}] metrics={task_metrics} samples={total}")

    if args.skip_score:
        done_path = output_dir / "generation_complete.json"
        done_path.write_text(
            json.dumps(
                {
                    "backend": "lmms-eval-fast",
                    "stage": "generate",
                    "lmms_eval_dir": lmms_dir,
                    "model_path": args.model_path,
                    "model_name": model_name,
                    "output_dir": str(output_dir),
                    "prompt_mode": args.prompt_mode,
                    "lmms_model_name": args.lmms_model_name,
                    "judge_extra_body": judge_extra_body,
                    "task_map": task_map,
                    "task_metrics": raw_task_info,
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {done_path}")
        return

    benchmarks = {
        spec.name: _collect_entry(spec, raw_task_metrics, {})
        for spec in specs
    }
    summary = {
        "backend": "lmms-eval-fast",
        "lmms_eval_dir": lmms_dir,
        "model_path": args.model_path,
        "model_name": model_name,
        "output_dir": str(output_dir),
        "prompt_mode": args.prompt_mode,
        "lmms_model_name": args.lmms_model_name,
        "judge_workers": args.judge_workers,
        "judge_extra_body": judge_extra_body,
        "task_map": task_map,
        "task_metrics": raw_task_info,
        "benchmarks": benchmarks,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
