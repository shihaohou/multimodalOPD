"""Stage 0 driver: evidence-reliance probe for ONE model (needs a GPU / vLLM).

For every sample it renders the image conditions
  full | mask_evidence | mask_rand x N | crop@pad ...
keeping the *question and prompt identical* (only the pixels change), generates an
answer per condition, grades it, and writes one row per (sample, condition) to
``<output-dir>/<model-name>/per_sample.jsonl`` plus ``meta.json``.

Run once per model; ``analyze_stage0.py`` then aggregates across models. Selection
is deterministic so every model sees identical samples.

  MODEL_PATH=... bash scripts/probe_stage0.sh
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from baseline.eval.grading import attempt_correct
from baseline.eval.opd_eval_prompt import GENERAL_PROMPT_DESCRIPTION, build_general_eval_messages
from baseline.opd_data_collator import OPD_SYSTEM_PROMPT
from baseline.probe.image_ops import build_sanity_sheet, crop_box, mask_box, norm_to_px, random_box_same_shape
from baseline.probe.saliency_data import load_saliency_samples
from vigos.eval_utils import extract_model_answer, vllm_request


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 0 evidence-reliance probe (one model).")
    p.add_argument("--model-path", required=True)
    p.add_argument("--model-name", default=None)
    p.add_argument("--output-dir", required=True, help="Parent dir; per-model subdir is created.")
    # data
    p.add_argument("--dataset", default="peterant330/saliency-r1-8k")
    p.add_argument("--split", default="train")
    p.add_argument("--limit", type=int, default=200, help="Per-subset sample cap.")
    p.add_argument("--subsets", default=None, help="Comma list e.g. textvqa,docvqa (default all).")
    p.add_argument("--max-bbox-area", type=float, default=0.5,
                   help="Drop samples whose evidence box exceeds this area frac (random-mask "
                   "control needs room). 0 / negative disables.")
    p.add_argument("--min-bbox-area", type=float, default=None, help="Drop boxes smaller than this area frac.")
    # conditions
    p.add_argument("--mask-fill", default="gray", choices=["gray", "black", "mean", "blur"])
    p.add_argument("--n-rand", type=int, default=3, help="Random equal-shape masks to average.")
    p.add_argument("--crop-pads", default="0,0.1,0.2", help="Comma list of crop padding fractions.")
    p.add_argument("--mask-seed", type=int, default=1234, help="Seed for random-mask placement.")
    p.add_argument("--sanity-dump", type=int, default=8, help="Overlay montages to also dump.")
    p.add_argument("--sanity-only", action="store_true", help="Dump sanity sheets and exit (no GPU).")
    # generation / prompt
    p.add_argument("--system-prompt", default=None,
                   help="Override the system prompt (default: OPD unified CoT+boxed prompt). "
                   "Use a model's native format if its Acc_full looks suppressed.")
    p.add_argument("--no-system-prompt", action="store_true", help="Send no system prompt at all.")
    p.add_argument("--prompt-suffix", default="")
    p.add_argument("--temperature", type=float, default=0.0, help="0 => greedy (clean probe).")
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=-1)
    p.add_argument("--pass-k", type=int, default=1, help="Samples/condition (forced 1 if greedy).")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--limit-images", type=int, default=1, help="Images per prompt (probe uses 1).")
    p.add_argument("--dtype", default="auto")
    # grading
    p.add_argument("--grader", default="rule", choices=["rule", "llm"])
    p.add_argument("--judge-model", default="deepseek-v4-flash")
    p.add_argument("--judge-api-url", default="https://api.deepseek.com")
    p.add_argument("--judge-key-env", default="DEEPSEEK_API_KEY")
    p.add_argument("--judge-workers", type=int, default=32)
    p.add_argument("--judge-timeout", type=float, default=120.0)
    p.add_argument("--judge-retries", type=int, default=2)
    return p.parse_args()


# ------------------------------------------------------------------ conditions
def build_jobs(samples, *, fill: str, n_rand: int, pads, mask_seed: int) -> list[dict[str, Any]]:
    """One job per (sample, condition-variant) with its rendered image."""
    jobs: list[dict[str, Any]] = []
    for s in samples:
        w, h = s.image.size
        ev_px = norm_to_px(s.bbox_norm, w, h)
        # Deterministic per-sample RNG (crc32, NOT salted hash()) so reruns and
        # different model runs draw the SAME random masks -> reproducible + paired.
        rng = np.random.default_rng(mask_seed + zlib.crc32(str(s.sample_id).encode()))

        def add(condition: str, variant: str, image):
            jobs.append(
                {
                    "sample_id": s.sample_id,
                    "subset": s.subset,
                    "problem": s.problem,
                    "gt": s.solution,
                    "condition": condition,
                    "variant": variant,
                    "image": image,
                }
            )

        add("full", "full", s.image)
        add("mask_evidence", "mask_evidence", mask_box(s.image, ev_px, fill=fill))
        for j in range(n_rand):
            rand_px = random_box_same_shape(ev_px, w, h, rng)
            add("mask_rand", f"rand{j}", mask_box(s.image, rand_px, fill=fill))
        for pad in pads:
            add("crop", f"pad{pad}", crop_box(s.image, s.bbox_norm, pad_frac=pad))
    return jobs


# ------------------------------------------------------------------ generation
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

    greedy = args.temperature <= 0.0
    return SamplingParams(
        n=1 if greedy else max(1, args.pass_k),
        temperature=args.temperature,
        top_p=1.0 if greedy else args.top_p,
        top_k=-1 if greedy else (args.top_k if args.top_k > 0 else -1),
        max_tokens=args.max_tokens,
        seed=args.seed,
    )


def generate_rows(engine, processor, sampling_params, jobs, args, system_prompt: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    batch = max(1, args.batch_size)
    n = len(jobs)
    for start in range(0, n, batch):
        window = jobs[start : start + batch]
        requests = []
        for job in window:
            messages = build_general_eval_messages(
                job["problem"], [job["image"]], system_prompt=system_prompt, suffix=args.prompt_suffix
            )
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            requests.append(vllm_request(prompt, [job["image"]]))
        outputs = engine.generate(requests, sampling_params, use_tqdm=False)
        for job, output in zip(window, outputs, strict=True):
            cand = output.outputs[0].text if output.outputs else ""
            rows.append(
                {
                    "sample_id": job["sample_id"],
                    "subset": job["subset"],
                    "condition": job["condition"],
                    "variant": job["variant"],
                    "gt": job["gt"],
                    "problem": job["problem"],
                    "response": cand[:2000],
                    "extracted": extract_model_answer(cand),
                    "correct": None,  # filled by grading
                }
            )
        if (start // batch) % 20 == 0:
            print(f"  generated {min(start + batch, n)}/{n} jobs", flush=True)
    return rows


# --------------------------------------------------------------------- grading
def grade_rule(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        row["correct"] = 1 if attempt_correct(row["response"], row["gt"]) else 0


def grade_llm(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    from openai import OpenAI

    from vigos.eval_utils import build_judge_messages, parse_judge_output

    api_key = os.environ.get(args.judge_key_env) or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(f"No judge key in ${args.judge_key_env}/$OPENAI_API_KEY (use --grader rule).")
    client = OpenAI(base_url=args.judge_api_url, api_key=api_key)

    def judge_one(row: dict[str, Any]) -> int:
        messages = build_judge_messages(row["extracted"], row["gt"], row["problem"])
        for _ in range(args.judge_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=args.judge_model, messages=messages, temperature=0.0,
                    max_tokens=512, timeout=args.judge_timeout,
                )
                verdict = parse_judge_output(resp.choices[0].message.content).get("verdict")
                return 1 if verdict == "correct" else 0
            except Exception:
                continue
        return 0

    with ThreadPoolExecutor(max_workers=max(1, args.judge_workers)) as pool:
        verdicts = list(pool.map(judge_one, rows))
    for row, v in zip(rows, verdicts, strict=True):
        row["correct"] = int(v)


# ------------------------------------------------------------------------- main
def main() -> None:
    args = parse_args()
    subsets = [s for s in (args.subsets or "").split(",") if s.strip()] or None
    pads = [float(x) for x in args.crop_pads.split(",") if x.strip() != ""]
    model_name = args.model_name or os.path.basename(args.model_path.rstrip("/"))

    out_dir = Path(args.output_dir) / model_name
    (out_dir / "sanity").mkdir(parents=True, exist_ok=True)

    max_area = args.max_bbox_area if (args.max_bbox_area and args.max_bbox_area > 0) else None
    samples = load_saliency_samples(
        args.dataset, args.split, limit=args.limit, subsets=subsets,
        max_bbox_area=max_area, min_bbox_area=args.min_bbox_area,
    )
    if not samples:
        raise SystemExit("No samples loaded -- check --dataset / --subsets / --limit.")

    # Sanity montages (cheap; helps catch a bbox-mapping bug before/without GPU).
    rng = np.random.default_rng(args.mask_seed)
    for s in samples[: max(0, args.sanity_dump)]:
        sheet = build_sanity_sheet(s.image, s.bbox_norm, rng, fill=args.mask_fill, pads=tuple(pads))
        sheet.save(out_dir / "sanity" / f"{s.subset}_{s.sample_id}.png")
    if args.sanity_only:
        print(f"[sanity-only] wrote montages to {out_dir / 'sanity'}; exiting before GPU load.")
        return

    jobs = build_jobs(samples, fill=args.mask_fill, n_rand=args.n_rand, pads=pads, mask_seed=args.mask_seed)
    print(f"[stage0] {model_name}: {len(samples)} samples -> {len(jobs)} generation jobs")

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    engine = make_engine(args)
    sampling_params = make_sampling_params(args)

    if args.no_system_prompt:
        system_prompt = ""
    elif args.system_prompt is not None:
        system_prompt = args.system_prompt
    else:
        system_prompt = OPD_SYSTEM_PROMPT

    rows = generate_rows(engine, processor, sampling_params, jobs, args, system_prompt)
    if args.grader == "rule":
        grade_rule(rows)
    else:
        grade_llm(rows, args)

    per_sample = out_dir / "per_sample.jsonl"
    with per_sample.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts: Counter[str] = Counter(s.subset for s in samples)
    meta = {
        "model_name": model_name,
        "model_path": args.model_path,
        "dataset": args.dataset,
        "split": args.split,
        "limit_per_subset": args.limit,
        "subsets": subsets,
        "max_bbox_area": max_area,
        "min_bbox_area": args.min_bbox_area,
        "n_samples": len(samples),
        "subset_counts": dict(counts),
        "sample_ids": [s.sample_id for s in samples],
        "conditions": {"mask_fill": args.mask_fill, "n_rand": args.n_rand, "crop_pads": pads},
        "generation": {
            "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
            "pass_k": args.pass_k, "max_tokens": args.max_tokens, "seed": args.seed,
            "greedy": args.temperature <= 0.0,
        },
        "grader": args.grader,
        "prompt": GENERAL_PROMPT_DESCRIPTION,
        "system_prompt": system_prompt,
        "n_jobs": len(jobs),
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    acc_full = np.mean([r["correct"] for r in rows if r["condition"] == "full"]) if rows else float("nan")
    print(f"[stage0] {model_name}: wrote {len(rows)} rows -> {per_sample}")
    print(f"[stage0] {model_name}: Acc_full = {acc_full:.3f} (n={len(samples)}); run analyze_stage0.py next.")


if __name__ == "__main__":
    main()
