"""Standalone vLLM teacher-scoring server for OPD (top-k logprobs).

Loads a teacher VLM with an offline vLLM engine and exposes ``POST /score_topk``.
For each item it scores ``prompt_token_ids + completion_token_ids`` (+ image) with
``prompt_logprobs=top_k`` and returns the teacher's top-k token ids + log-probs at
each completion position. The trainer's :class:`baseline.teacher_client.VLLMServerTeacher`
consumes this and computes a forward top-k KL.

Run on dedicated GPUs, separate from training:
    TEACHER_MODEL=Qwen/Qwen2.5-VL-72B-Instruct bash scripts/serve_teacher_vllm.sh

EXPERIMENTAL: the multimodal ``prompt_token_ids`` + ``prompt_logprobs`` path needs
GPU validation against the installed vLLM version.
"""

from __future__ import annotations

import argparse
import base64
import io
import threading
from typing import Any

from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="vLLM OPD teacher scoring server.")
    p.add_argument("--model", required=True)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8200)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--limit-images", type=int, default=16)
    p.add_argument("--max-num-seqs", type=int, default=64)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from fastapi import FastAPI
    from vllm import LLM, SamplingParams
    import uvicorn

    engine_kwargs: dict[str, Any] = dict(
        model=args.model,
        trust_remote_code=True,
        tokenizer_mode="slow",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": args.limit_images},
        max_num_seqs=args.max_num_seqs,
        dtype=args.dtype,
        seed=args.seed,
    )
    if args.max_model_len is not None:
        engine_kwargs["max_model_len"] = args.max_model_len
    engine = LLM(**engine_kwargs)
    engine_lock = threading.Lock()

    app = FastAPI()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": args.model}

    @app.post("/score_topk")
    def score_topk(body: dict[str, Any]) -> dict[str, Any]:
        top_k = int(body.get("top_k", 32))
        items = body.get("items", [])

        prompts = []
        prompt_lens = []
        comp_lens = []
        for item in items:
            prompt_ids = [int(x) for x in item.get("prompt_token_ids", [])]
            completion_ids = [int(x) for x in item.get("completion_token_ids", [])]
            prompt = {"prompt_token_ids": prompt_ids + completion_ids}
            image = _decode_image(item.get("image_b64"))
            if image is not None:
                prompt["multi_modal_data"] = {"image": image}
            prompts.append(prompt)
            prompt_lens.append(len(prompt_ids))
            comp_lens.append(len(completion_ids))

        sampling = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=top_k)
        with engine_lock:
            outputs = engine.generate(prompts, sampling, use_tqdm=False)

        result_items = []
        for output, prompt_len, comp_len in zip(outputs, prompt_lens, comp_lens, strict=True):
            topk_ids, topk_logprobs = _extract_topk(
                output.prompt_logprobs, prompt_len, comp_len, top_k
            )
            result_items.append({"topk_ids": topk_ids, "topk_logprobs": topk_logprobs})
        return {"items": result_items}

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def _extract_topk(
    prompt_logprobs: Any,
    prompt_len: int,
    comp_len: int,
    top_k: int,
) -> tuple[list[list[int]], list[list[float]]]:
    """Per completion position, the teacher's top-k token ids + log-probs.

    ``prompt_logprobs[i]`` is the predictive distribution for sequence token ``i``
    given tokens ``< i``. The completion token at completion-index ``j`` sits at
    sequence index ``prompt_len + j``, so its predictive distribution is
    ``prompt_logprobs[prompt_len + j]``.
    """
    ids_out: list[list[int]] = []
    logprobs_out: list[list[float]] = []
    if not prompt_logprobs:
        return ids_out, logprobs_out
    for j in range(comp_len):
        index = prompt_len + j
        entries = prompt_logprobs[index] if 0 <= index < len(prompt_logprobs) else None
        if not entries:
            ids_out.append([])
            logprobs_out.append([])
            continue
        ranked = sorted(
            entries.items(), key=lambda kv: _logprob_value(kv[1]), reverse=True
        )[:top_k]
        ids_out.append([int(token_id) for token_id, _ in ranked])
        logprobs_out.append([float(_logprob_value(value)) for _, value in ranked])
    return ids_out, logprobs_out


def _logprob_value(value: Any) -> float:
    # vLLM returns Logprob objects with a .logprob attribute; tolerate plain floats.
    return float(getattr(value, "logprob", value))


def _decode_image(image_b64: str | None) -> Image.Image | None:
    if not image_b64:
        return None
    raw = base64.b64decode(image_b64)
    with Image.open(io.BytesIO(raw)) as image:
        return image.convert("RGB")


if __name__ == "__main__":
    main()
