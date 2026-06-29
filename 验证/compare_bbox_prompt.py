#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Deploy two models with vLLM (4 GPUs each) and compare two prompt schemes on
saliency-r1-8k, then report a final accuracy per model x scheme.

Each sample in saliency-r1-8k has a GT *evidence* bounding box. We test whether
handing the model that box in the prompt helps it answer:

  * scheme ``plain`` — original image + question.                      [方案① 原始图片+prompt]
  * scheme ``bbox``  — same, plus an English hint naming the GT box,    [方案② 加 bbox 提示]
                       e.g. "Pay special attention to the region inside
                       the bounding box [x1, y1, x2, y2] ...".
  * scheme ``draw``  — (optional) the GT box drawn on the image.

Deployment: one command launches TWO worker subprocesses, each pinned to its own
4-GPU group (``CUDA_VISIBLE_DEVICES``) running vLLM with ``tensor_parallel_size=4``,
so both models serve concurrently on an 8-GPU box. Each worker generates over the
whole dataset under every scheme, grades the \boxed answer, and writes a JSONL; the
parent merges them and prints the final accuracy table.

Run on the box, in the OPD uv env:

    M=/home/web_server/antispam/project/houshihao/models
    uv run python 验证/compare_bbox_prompt.py \
        --models base=$M/Qwen2.5-VL-3B-Instruct,opd=runs/opd_qwen25_3b_xxx/checkpoint-200 \
        --gpu-groups "0,1,2,3;4,5,6,7" \
        --dataset /home/web_server/antispam/project/houshihao/datasets/saliency-r1-8k \
        --subsets docvqa --limit 200 --output-dir eval_outputs/bbox_ab

``--limit`` is a PER-SUBSET cap (drop it to run the full set). Greedy decoding by
default. Grading is rule-based (mathruler + exact, no API); ``--grader llm`` uses the
DeepSeek judge (better for free-form DocVQA answers; needs $DEEPSEEK_API_KEY).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# --- make the repo importable as a library (this file lives in <repo>/验证/) -----
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_DEFAULT_MODELS_DIR = "/home/web_server/antispam/project/houshihao/models"
_DEFAULT_DATASET = "/home/web_server/antispam/project/houshihao/datasets/saliency-r1-8k"

# English hint appended to the question for the `bbox` scheme. {coords}/{note} filled in.
_DEFAULT_HINT = (
    "Hint: pay special attention to the region of the image inside the bounding "
    "box {coords} ({note}). The evidence needed to answer the question is located there."
)

# Shared eval settings the parent passes to each worker via config.json (everything
# EXCEPT the per-worker model / gpu / output, so both workers run identically).
_CONFIG_KEYS = (
    "dataset", "split", "subsets", "limit", "max_bbox_area", "schemes", "coord_mode",
    "bbox_hint", "noverbalize_hint", "draw_color", "draw_width", "system_prompt", "prompt_suffix",
    "max_new_tokens", "do_sample", "temperature", "top_p", "top_k", "seed",
    "dtype", "gpu_mem_util", "limit_images", "grader", "batch_size", "max_image_side",
)


# --------------------------------------------------------------------------- CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Two vLLM models x two prompt schemes (bbox vs not) on saliency-r1-8k.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # orchestration
    p.add_argument(
        "--models",
        default=(
            f"capcurriculum-8b={_DEFAULT_MODELS_DIR}/CapCurriculum-8B,"
            f"qwen3vl-8b={_DEFAULT_MODELS_DIR}/Qwen3-VL-8B-Instruct"
        ),
        help="Comma-separated name=path pairs (the models to deploy).",
    )
    p.add_argument("--gpu-groups", default="0,1,2,3;4,5,6,7",
                   help="';'-separated GPU groups, one per model (4 GPUs each here).")
    p.add_argument("--output-dir", default="eval_outputs/bbox_ab")
    # data
    p.add_argument("--dataset", default=_DEFAULT_DATASET, help="Local dir or HF id.")
    p.add_argument("--split", default="train")
    p.add_argument("--subsets", default=None, help="Comma list, e.g. docvqa,textvqa (default: all).")
    p.add_argument("--limit", type=int, default=200, help="PER-SUBSET sample cap (None-like: pass -1).")
    p.add_argument("--max-bbox-area", type=float, default=None, help="Drop boxes larger than this (fraction).")
    # schemes / hint
    p.add_argument("--schemes", default="plain,bbox",
                   help="Subset of: plain,bbox,noverbalize,draw. (bbox = verbalize-allowed hint; "
                        "noverbalize = OPD silent hint.)")
    p.add_argument("--coord-mode", default="normalized", choices=["normalized", "pixel"],
                   help="How the bbox coords are written into the hint.")
    p.add_argument("--bbox-hint", default=_DEFAULT_HINT, help="'bbox' scheme template; needs {coords} (+ optional {note}).")
    p.add_argument("--noverbalize-hint", default=None,
                   help="'noverbalize' scheme template; needs {bbox}. Default = baseline.hint HINT_TEMPLATE "
                        "(the strict 'do NOT mention the box/coords/hint/crop'). Pass a softer one to A/B it.")
    p.add_argument("--draw-color", default="red")
    p.add_argument("--draw-width", type=int, default=4)
    # prompt / generation
    p.add_argument("--system-prompt", default="think", help="think|freecot|reason|none (match training).")
    p.add_argument("--prompt-suffix", default="")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--do-sample", action="store_true", help="Default off = greedy (reproducible).")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    # vLLM
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--gpu-mem-util", type=float, default=0.90)
    p.add_argument("--limit-images", type=int, default=4)
    # streaming / memory safety (prevents host-RAM blow-up on the full dataset)
    p.add_argument("--batch-size", type=int, default=64,
                   help="Samples decoded+generated per chunk. Bounds host RAM; raise for throughput.")
    p.add_argument("--max-image-side", type=int, default=1536,
                   help="Downscale any image whose longest side exceeds this (0 = off). Qwen-VL "
                   "smart-resizes anyway, so this mainly caps RAM/preproc on huge DocVQA scans.")
    # grading
    p.add_argument("--grader", default="rule", choices=["rule", "llm"],
                   help="rule = mathruler+exact (no API); llm = DeepSeek judge ($DEEPSEEK_API_KEY).")
    p.add_argument("--judge-model", default="deepseek-v4-flash",
                   help="Judge model name. For a self-hosted server use its --served-model-name.")
    p.add_argument("--judge-api-url", default="https://api.deepseek.com",
                   help="OpenAI-compatible base URL (e.g. http://localhost:8000/v1 for local vLLM).")
    p.add_argument("--judge-key-env", default="DEEPSEEK_API_KEY",
                   help="Env var holding the API key (a self-hosted server needs none → dummy used).")
    p.add_argument("--judge-workers", type=int, default=32)
    p.add_argument("--judge-no-think", action="store_true",
                   help="Disable thinking for the judge (Qwen3 etc.) via chat_template_kwargs — "
                        "prevents <think> from truncating the JSON verdict. Recommended for a local hybrid judge.")
    p.add_argument("--judge-max-tokens", type=int, default=2048,
                   help="Judge max_new_tokens (raise if you keep thinking ON so the verdict isn't cut off).")
    # re-grade an existing run with the LLM judge (no vLLM / no regeneration)
    p.add_argument("--judge-only", action="store_true",
                   help="Skip generation: re-grade --output-dir/records.jsonl with the LLM judge "
                        "and write summary_llm.json + records_llm.jsonl.")
    # internal worker mode
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--model", default=None, help=argparse.SUPPRESS)        # one name=path
    p.add_argument("--tp", type=int, default=4, help=argparse.SUPPRESS)
    p.add_argument("--worker-out", default=None, help=argparse.SUPPRESS)
    p.add_argument("--config", default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "model"


def parse_models(spec: str) -> list[tuple[str, str]]:
    out = []
    for part in spec.split(","):
        item = part.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"--models entry must be name=path, got {item!r}")
        name, path = item.split("=", 1)
        out.append((name.strip(), path.strip()))
    if not out:
        raise ValueError("No models given. Pass --models name=path,name=path.")
    return out


# ------------------------------------------------------------------ prompt schemes
def format_coords(bbox_norm, image_size, mode: str) -> tuple[str, str]:
    x1, y1, x2, y2 = bbox_norm
    if mode == "pixel":
        w, h = image_size
        pts = [round(x1 * w), round(y1 * h), round(x2 * w), round(y2 * h)]
        note = f"in pixels; image size {w}x{h}, top-left origin, [x1, y1, x2, y2]"
    else:
        pts = [round(v, 3) for v in (x1, y1, x2, y2)]
        note = "normalized to [0, 1], top-left origin, [x1, y1, x2, y2]"
    return f"[{pts[0]}, {pts[1]}, {pts[2]}, {pts[3]}]", note


def build_scheme_item(problem: str, bbox_norm, image, scheme: str, args):
    """(problem_text, image) for a scheme, given an already-decoded image. plain/bbox/
    noverbalize keep the image; draw burns the GT box into a copy and keeps the question.

    Schemes:
      * plain       — just the question.
      * bbox        — verbalize-allowed hint ("pay special attention to [box], evidence is there").
      * noverbalize — the OPD-faithful silent hint (baseline.hint HINT_TEMPLATE: "use this only
                      to decide where to look ... do NOT mention the box"). Same box, but forbids
                      the model from reasoning about / mentioning it — what OPD can actually distill.
      * draw        — the GT box drawn on the image.
    """
    from PIL import ImageDraw

    if scheme == "plain":
        return problem, image
    if scheme == "bbox":
        coords, note = format_coords(bbox_norm, image.size, args.coord_mode)
        return f"{problem}\n{args.bbox_hint.format(coords=coords, note=note)}", image
    if scheme == "noverbalize":
        # OPD no-verbalize hint (always normalized [0,1], 2 decimals). --noverbalize-hint
        # overrides the default strict template (e.g. a softer "do NOT mention the box").
        from baseline.hint.opd_hint_collator import HINT_TEMPLATE, format_bbox_hint

        tpl = getattr(args, "noverbalize_hint", None) or HINT_TEMPLATE
        return f"{problem}\n{format_bbox_hint(bbox_norm, tpl, decimals=2)}", image
    if scheme == "draw":
        img = image.convert("RGB").copy()
        w, h = img.size
        x1, y1, x2, y2 = bbox_norm
        ImageDraw.Draw(img).rectangle(
            [x1 * w, y1 * h, x2 * w, y2 * h], outline=args.draw_color, width=args.draw_width
        )
        return problem, img
    raise ValueError(f"unknown scheme {scheme!r}")


def _build_index_viscot(args):
    """Visual-CoT index: per-domain ``metadata/*.jsonl`` → ``kept`` items carrying a
    resolved image PATH (not an HF row). subset = the domain (jsonl stem) so ``--limit``
    stratifies per task; pixel ``bboxs`` are normalized to [0,1]. Reuses the canonical
    Visual-CoT helpers in ``baseline.opd_dataset`` (image-basename index + resolver)."""
    import json
    import os

    from baseline.opd_dataset import (
        _build_viscot_image_index,
        _normalize_viscot_bbox,
        _resolve_viscot_image,
        _viscot_jsonl_files,
    )
    from baseline.probe.saliency_data import bbox_area, parse_bbox_norm

    root = os.environ.get("VISCOT_IMAGE_ROOT") or args.dataset
    index = _build_viscot_image_index(root, verbose=True)
    files = _viscot_jsonl_files(args.dataset)
    subset_filter = {s.strip().lower() for s in args.subsets.split(",")} if args.subsets else None
    limit = None if (args.limit is not None and args.limit < 0) else args.limit

    kept: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for jsonl in files:
        domain = os.path.splitext(os.path.basename(jsonl))[0]
        if subset_filter is not None and domain.lower() not in subset_filter:
            continue
        with open(jsonl, encoding="utf-8") as fh:
            for li, line in enumerate(fh):
                if limit is not None and counts.get(domain, 0) >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                bbox = parse_bbox_norm(_normalize_viscot_bbox(row.get("bboxs"), row.get("width"), row.get("height")))
                if bbox is None:
                    continue
                if args.max_bbox_area is not None and bbox_area(bbox) > args.max_bbox_area:
                    continue
                problem = str(row.get("question", "")).strip()
                solution = str(row.get("answer", "")).strip()
                if not problem or not solution:
                    continue
                img = _resolve_viscot_image(row.get("image"), row.get("dataset"), root, index)
                if img is None:
                    continue
                kept.append({"image_path": img, "id": f"{domain}_{li}", "subset": domain,
                             "problem": problem, "solution": solution, "bbox": bbox})
                counts[domain] = counts.get(domain, 0) + 1
    return kept, counts


def build_index(args):
    """Lazily build the sample list WITHOUT decoding any image (filter on text/bbox
    columns only), so host RAM does not scale with dataset size. Returns
    ``(data, image_column, kept, counts)``: for HF datasets ``data`` decodes images
    on demand (``data[row][image_column]``); for Visual-CoT dirs ``data`` is None and
    each item carries an ``image_path`` (opened on demand)."""
    from baseline.opd_dataset import _looks_like_viscot
    from baseline.probe.saliency_data import _load_hf_split, bbox_area, parse_bbox_norm

    if _looks_like_viscot(args.dataset):
        kept, counts = _build_index_viscot(args)
        return None, None, kept, counts

    data = _load_hf_split(args.dataset, args.split)
    cols = list(data.column_names)
    image_column = "image" if "image" in cols else ("images" if "images" in cols else None)
    text_cols = [c for c in cols if c != image_column]
    meta = data.select_columns(text_cols)  # arrow view; no image bytes materialized
    n = len(meta)

    def col(name, default=None):
        return meta[name] if name in text_cols else [default] * n

    subsets_col, problems_col = col("dataset", "unknown"), col("problem", "")
    solutions_col, bboxes_col, ids_col = col("solution", ""), col("bbox"), col("question_id")
    subset_filter = {s.strip().lower() for s in args.subsets.split(",")} if args.subsets else None
    limit = None if (args.limit is not None and args.limit < 0) else args.limit

    kept: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for i in range(n):
        subset = str(subsets_col[i] or "").strip() or "unknown"
        if subset_filter is not None and subset.lower() not in subset_filter:
            continue
        if limit is not None and counts.get(subset, 0) >= limit:
            continue
        bbox = parse_bbox_norm(bboxes_col[i])
        if bbox is None:
            continue
        if args.max_bbox_area is not None and bbox_area(bbox) > args.max_bbox_area:
            continue
        problem = str(problems_col[i] or "").strip()
        solution = str(solutions_col[i] or "").strip()
        if not problem or not solution:
            continue
        kept.append({"row": i, "id": str(ids_col[i] if ids_col[i] is not None else i),
                     "subset": subset, "problem": problem, "solution": solution, "bbox": bbox})
        counts[subset] = counts.get(subset, 0) + 1
    return data, image_column, kept, counts


def decode_image(data, image_column, item, max_side: int):
    """Decode ONE image (HF row OR Visual-CoT path), downscale if longest side > max_side."""
    from PIL import Image

    from baseline.probe.saliency_data import _to_pil

    if "image_path" in item:  # Visual-CoT: image is a resolved file path
        image = Image.open(item["image_path"]).convert("RGB")
    else:
        image = _to_pil(data[item["row"]][image_column])
    if max_side and max(image.size) > max_side:
        image = image.copy()
        image.thumbnail((max_side, max_side))
    return image


# ------------------------------------------------------- worker: one model in vLLM
def run_worker(args) -> None:
    """Deploy ONE model in vLLM (TP=args.tp on the visible GPUs) and generate over the
    whole dataset under every scheme, STREAMING in chunks so host RAM stays bounded
    (images are decoded a chunk at a time, not all up front). Writes per-sample
    records to args.worker_out incrementally."""
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    from baseline.eval.grading import attempt_correct
    from baseline.eval.opd_eval_prompt import build_general_eval_prompt
    from baseline.opd_data_collator import resolve_opd_system_prompt
    from vigos.eval_utils import extract_model_answer, vllm_request

    name, path = args.model.split("=", 1)
    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    system_prompt = resolve_opd_system_prompt(args.system_prompt)

    data, image_column, kept, counts = build_index(args)
    summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "(none)"
    total = len(kept)
    print(f"[worker:{name}] {total} samples ({summary}); schemes={schemes}; TP={args.tp}; "
          f"batch_size={args.batch_size}; max_image_side={args.max_image_side}", flush=True)
    if total == 0:
        raise SystemExit(f"[worker:{name}] no samples after filtering.")

    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True, use_fast=False)
    engine = LLM(
        model=path, tensor_parallel_size=args.tp, gpu_memory_utilization=args.gpu_mem_util,
        limit_mm_per_prompt={"image": args.limit_images}, trust_remote_code=True,
        dtype=args.dtype, seed=args.seed,
    )
    sp = SamplingParams(
        n=1,
        temperature=(args.temperature if args.do_sample else 0.0),
        top_p=(args.top_p if args.do_sample else 1.0),
        top_k=(args.top_k if (args.do_sample and args.top_k > 0) else -1),
        max_tokens=args.max_new_tokens, seed=args.seed,
    )

    bs = max(1, args.batch_size)
    n_correct = {sc: 0 for sc in schemes}
    n_seen = {sc: 0 for sc in schemes}
    t0 = time.time()
    with open(args.worker_out, "w", encoding="utf-8") as out_fh:
        for start in range(0, total, bs):
            chunk = kept[start:start + bs]
            requests, meta = [], []
            for item in chunk:  # decode only this chunk's images (then let them be freed)
                image = decode_image(data, image_column, item, args.max_image_side)
                for sc in schemes:
                    problem_text, img = build_scheme_item(item["problem"], item["bbox"], image, sc, args)
                    prompt = build_general_eval_prompt(
                        processor, problem_text, [img], system_prompt=system_prompt, suffix=args.prompt_suffix
                    )
                    requests.append(vllm_request(prompt, [img]))
                    meta.append((item, sc, problem_text))
            outputs = engine.generate(requests, sp, use_tqdm=False)
            for (item, sc, problem_text), out in zip(meta, outputs):
                response = out.outputs[0].text
                correct = attempt_correct(response, item["solution"]) if args.grader == "rule" else None
                out_fh.write(json.dumps({
                    "model": name, "sample_id": item["id"], "subset": item["subset"], "scheme": sc,
                    "problem_text": problem_text, "bbox_norm": list(item["bbox"]),
                    "solution": item["solution"], "response": response,
                    "extracted_answer": extract_model_answer(response), "correct": correct,
                }, ensure_ascii=False) + "\n")
                n_seen[sc] += 1
                if correct:
                    n_correct[sc] += 1
            out_fh.flush()
            done = min(start + bs, total)
            rate = done / max(1e-9, time.time() - t0)
            print(f"[worker:{name}] {done}/{total} samples ({rate:.1f}/s)", flush=True)

    if args.grader == "rule":  # quick local readout per worker
        for sc in schemes:
            acc = n_correct[sc] / n_seen[sc] if n_seen[sc] else float("nan")
            print(f"[worker:{name}] acc[{sc}] = {acc:.3f}  (n={n_seen[sc]})", flush=True)
    print(f"[worker:{name}] wrote {args.worker_out} in {time.time() - t0:.0f}s", flush=True)


# -------------------------------------------------------------------------- grading
def grade_llm(records: list[dict[str, Any]], args) -> None:
    """Fill record['correct'] via the DeepSeek judge (same prompts as run_opd_eval)."""
    from concurrent.futures import ThreadPoolExecutor

    from openai import OpenAI

    from vigos.eval_utils import build_judge_messages, parse_judge_output

    # A self-hosted OpenAI-compatible server (vLLM) needs no real key — fall back to a
    # dummy so only a remote API (DeepSeek) requires the env var.
    key = os.environ.get(args.judge_key_env) or os.environ.get("OPENAI_API_KEY") or "EMPTY"
    client = OpenAI(base_url=args.judge_api_url, api_key=key)
    # Disable Qwen3 (and other hybrid) thinking for the judge: a <think> block can eat
    # the token budget and truncate the JSON verdict → parser falls back to "incorrect"
    # (a silent, possibly scheme-asymmetric bias). vLLM honors this chat-template kwarg.
    extra_body = {"chat_template_kwargs": {"enable_thinking": False}} if getattr(args, "judge_no_think", False) else None
    max_tok = getattr(args, "judge_max_tokens", 2048)

    def judge_one(rec):
        msgs = build_judge_messages(rec["extracted_answer"], rec["solution"], rec["problem_text"])
        try:
            resp = client.chat.completions.create(
                model=args.judge_model, messages=msgs, temperature=0.0,
                max_tokens=max_tok, timeout=180, extra_body=extra_body,
            )
            return parse_judge_output(resp.choices[0].message.content).get("verdict") == "correct"
        except Exception as exc:  # noqa: BLE001
            print(f"[judge] error on {rec.get('sample_id')}: {exc}")
            return False

    mode = "no-think" if extra_body else "thinking-on"
    print(f"[bbox-ab] grading {len(records)} responses with the LLM judge "
          f"(model={args.judge_model}, {mode}, max_tokens={max_tok}) ...", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, args.judge_workers)) as pool:
        for rec, ok in zip(records, pool.map(judge_one, records)):
            rec["correct"] = bool(ok)


# ------------------------------------------------------------------------- reporting
def accuracy(rows: list[dict[str, Any]]):
    rows = [r for r in rows if r.get("correct") is not None]
    return (sum(bool(r["correct"]) for r in rows) / len(rows)) if rows else None


def summarize(records, model_names, schemes) -> dict[str, Any]:
    results = {}
    for name in model_names:
        per_scheme = {sc: {"n": len([r for r in records if r["model"] == name and r["scheme"] == sc]),
                           "acc": accuracy([r for r in records if r["model"] == name and r["scheme"] == sc])}
                      for sc in schemes}
        entry = {"schemes": per_scheme}
        # Δ vs plain for EVERY non-plain scheme (bbox, noverbalize, draw, ...).
        if "plain" in per_scheme:
            a0 = per_scheme["plain"]["acc"]
            for sc in schemes:
                if sc == "plain":
                    continue
                a1 = per_scheme[sc]["acc"]
                entry[f"delta_{sc}_minus_plain"] = (a1 - a0) if (a0 is not None and a1 is not None) else None
        results[name] = entry
    return results


def print_table(results, schemes) -> None:
    deltas = [sc for sc in schemes if sc != "plain"] if "plain" in schemes else []
    print("\n" + "=" * 88)
    print("Final answer accuracy  (model x prompt scheme)")
    print("=" * 88)
    head = f"{'model':<18}" + "".join(f"{sc:>14}" for sc in schemes) + "".join(f"{'Δ(' + sc + ')':>14}" for sc in deltas)
    print(head)
    print("-" * len(head))
    for name, entry in results.items():
        row = f"{name:<18}"
        for sc in schemes:
            a = entry["schemes"][sc]["acc"]
            row += f"{(f'{a:.3f}' if a is not None else '—'):>14}"
        for sc in deltas:
            d = entry.get(f"delta_{sc}_minus_plain")
            row += f"{(f'{d:+.3f}' if d is not None else '—'):>14}"
        print(row)
    print("-" * len(head))
    if deltas:
        print("Δ(scheme) = acc(scheme) − acc(plain); > 0 means that hint helped.")


# ----------------------------------------------------------------- orchestrator
def run_orchestrator(args) -> None:
    models = parse_models(args.models)
    groups = [g.strip() for g in args.gpu_groups.split(";") if g.strip()]
    if len(groups) != len(models):
        raise SystemExit(f"{len(models)} models but {len(groups)} GPU groups — they must match "
                         f"(e.g. --models a=..,b=.. --gpu-groups '0,1,2,3;4,5,6,7').")
    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    bad = [s for s in schemes if s not in {"plain", "bbox", "noverbalize", "draw"}]
    if bad:
        raise SystemExit(f"unknown scheme(s) {bad}; pick from plain,bbox,noverbalize,draw")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "config.json"
    cfg_path.write_text(json.dumps({k: getattr(args, k) for k in _CONFIG_KEYS}, ensure_ascii=False, indent=2))

    # Launch one worker subprocess per model, each pinned to its 4-GPU group.
    procs = []
    for (name, path), gpus in zip(models, groups):
        tp = len([g for g in gpus.split(",") if g.strip()])
        wout = out_dir / f"worker_{_safe(name)}.jsonl"
        wlog = out_dir / f"worker_{_safe(name)}.log"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus)
        env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        cmd = [sys.executable, os.path.abspath(__file__), "--worker",
               "--config", str(cfg_path), "--model", f"{name}={path}",
               "--tp", str(tp), "--worker-out", str(wout)]
        print(f"[bbox-ab] deploy '{name}'  GPUs={gpus} (TP={tp})  ->  log: {wlog}", flush=True)
        log_fh = open(wlog, "w", encoding="utf-8")
        procs.append({"name": name, "out": wout, "log": wlog, "fh": log_fh,
                      "p": subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)})

    print(f"[bbox-ab] both models deploying concurrently; waiting ... (tail -f {out_dir}/worker_*.log)", flush=True)
    failed = []
    for d in procs:
        rc = d["p"].wait()
        d["fh"].close()
        status = "ok" if rc == 0 else f"FAILED (exit {rc})"
        print(f"[bbox-ab] worker '{d['name']}' {status}", flush=True)
        if rc != 0:
            failed.append(d)

    for d in failed:
        print(f"\n----- last 30 lines of {d['log']} -----")
        try:
            print("".join(d["log"].read_text(encoding='utf-8').splitlines(keepends=True)[-30:]))
        except Exception:
            pass
    if failed:
        raise SystemExit("[bbox-ab] one or more workers failed; see logs above.")

    # Merge worker JSONLs and report.
    records: list[dict[str, Any]] = []
    for d in procs:
        with open(d["out"], encoding="utf-8") as fh:
            records.extend(json.loads(line) for line in fh if line.strip())
    if args.grader == "llm":
        grade_llm(records, args)

    model_names = [n for n, _ in models]
    results = summarize(records, model_names, schemes)
    subsets_seen = sorted({r["subset"] for r in records})
    per_subset = {sub: summarize([r for r in records if r["subset"] == sub], model_names, schemes)
                  for sub in subsets_seen}

    with open(out_dir / "records.jsonl", "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "dataset": args.dataset, "split": args.split, "subsets": args.subsets,
        "models": [{"name": n, "path": p, "gpus": g} for (n, p), g in zip(models, groups)],
        "schemes": schemes, "coord_mode": args.coord_mode, "bbox_hint": args.bbox_hint,
        "grader": args.grader, "system_prompt": args.system_prompt,
        "generation": {"max_new_tokens": args.max_new_tokens, "do_sample": args.do_sample,
                       "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k},
        "results": results, "per_subset": per_subset,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print_table(results, schemes)
    print(f"\n[bbox-ab] wrote {out_dir / 'records.jsonl'} and {out_dir / 'summary.json'}")


# --------------------------------------------------------------- judge-only mode
def run_judge_only(args) -> None:
    """Re-grade an existing run's records.jsonl with the LLM judge (no vLLM)."""
    out_dir = Path(args.output_dir)
    recs_path = out_dir / "records.jsonl"
    if not recs_path.exists():
        raise SystemExit(f"--judge-only: {recs_path} not found (run the rule pass first).")
    records = [json.loads(line) for line in recs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise SystemExit(f"--judge-only: {recs_path} is empty.")
    args.grader = "llm"
    grade_llm(records, args)  # overwrites record['correct'] with the judge verdict
    model_names = list(dict.fromkeys(r["model"] for r in records))
    schemes = list(dict.fromkeys(r["scheme"] for r in records))
    results = summarize(records, model_names, schemes)
    subsets_seen = sorted({r["subset"] for r in records})
    per_subset = {sub: summarize([r for r in records if r["subset"] == sub], model_names, schemes)
                  for sub in subsets_seen}
    with open(out_dir / "records_llm.jsonl", "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {"grader": "llm", "judge_model": args.judge_model, "schemes": schemes,
               "results": results, "per_subset": per_subset}
    (out_dir / "summary_llm.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print_table(results, schemes)
    print(f"\n[bbox-ab] judge-only wrote {out_dir / 'summary_llm.json'} and records_llm.jsonl")


# ------------------------------------------------------------------------------ main
def main() -> None:
    args = parse_args()
    if args.judge_only:
        run_judge_only(args)
    elif args.worker:
        if args.config:  # inherit the parent's shared eval settings
            cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
            for k, v in cfg.items():
                setattr(args, k, v)
        if not args.model or not args.worker_out:
            raise SystemExit("--worker needs --model name=path and --worker-out.")
        run_worker(args)
    else:
        run_orchestrator(args)


if __name__ == "__main__":
    main()
