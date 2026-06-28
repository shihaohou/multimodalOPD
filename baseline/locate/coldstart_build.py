"""Cold-start SFT trace builder for Locate-Once Grounding.

Generic-instruct students (e.g. Qwen3-VL-2B-Instruct) do NOT emit the locate-once
format zero-shot — no ``<think>``, no ``<box>`` — so the RL term never fires
(``box_coverage=0``). This builds a short SFT dataset that teaches the format AND the
``[0,1]`` box convention, so the subsequent RL+OPD run starts from a student that
already emits parseable boxes.

Pipeline (self-distillation cold-start), per ``(image, question, GT answer, GT box)``:

1. vLLM-sample the *student's own* reasoning on a plain reasoning prompt (the model
   reasons + ``\\boxed{}`` well even though it ignores the locate format);
2. keep attempts whose answer matches the GT (rejection sampling — quality, and the
   injected box only ever co-occurs with a correct rationale);
3. rebuild the target as::

       <think>
       <box>[x1, y1, x2, y2]</box>     # the GT box, normalized to [0,1]
       {reasoning}
       </think>
       \\boxed{answer}

4. save a HF dataset (``image`` / ``problem`` / ``target``) to ``--output_dir`` for
   :mod:`baseline.locate.coldstart_sft`.

Only the ``<think>``/``<box>`` scaffold + the GT box are injected; the reasoning is
the model's own (in-distribution), so SFT teaches the FORMAT and box *prediction*
(image+Q -> GT box) without rewriting the reasoning distribution. RL+OPD then refine
grounding. ``--gen_model`` generates with a different/stronger model (e.g. the teacher);
``--gen_hint`` additionally feeds that generator the GT box via the OPD hidden-hint
prompt (no-verbalize) so its reasoning is GROUNDED to the evidence region — the
strongest cold-start. The box is GT-injected either way; these flags only change the
quality/grounding of the *reasoning* text.

Run (single GPU, vLLM):
    uv run python -m baseline.locate.coldstart_build \\
        --model_path $M/Qwen3-VL-2B-Instruct \\
        --dataset_name $D/Visual-CoT --answer_field answer \\
        --output_dir runs/coldstart_locate_traces --max_samples 4000
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Any

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from baseline.eval.opd_eval_prompt import build_general_eval_messages
from baseline.locate.locate_rl import collapse_extra_boxes, parse_student_box
from baseline.locate.prompts import LOCATE_SYSTEM_PROMPT
from baseline.opd_data_collator import (
    _safe_rgb_image,
    resolve_opd_system_prompt,
)
from baseline.opd_dataset import load_opd_dataset
from baseline.probe.saliency_data import parse_bbox_norm
from vigos.answer_utils import extract_boxed_content, normalize_reference_answer

# trace_mode="natural": the teacher is SHOWN the GT box and writes the WHOLE locate trace
# (box woven into the reasoning) itself, used verbatim — vs "inject", which generates plain
# reasoning and bolts a <box>[GT]</box> onto the head. Natural traces match the teacher's
# own thinking pattern (Rethinking-OPD: OPD needs compatible student/teacher patterns), so
# the cold-started student's reasoning is closer to the OPD target.
NATURAL_GEN_TEMPLATE = (
    "Hint: the region that contains the answer is <box>{bbox}</box> (coordinates normalized "
    "to [0,1], top-left origin, [x1, y1, x2, y2]). Inside <think>, follow three steps: "
    "(1) state this region once as <box>{bbox}</box>; (2) describe what is in that region "
    "(the visual details relevant to the question); (3) reason from that description to the "
    "answer. Refer to it as \"that region\" afterwards and do not repeat the coordinates. "
    "Then give the final answer in \\boxed{{}}."
)


JUDGE_PROMPT = (
    "You are grading whether a model's answer to a visual question is correct.\n"
    "Question: {question}\n"
    "Reference answer: {reference}\n"
    "Model answer: {candidate}\n"
    "Does the model answer have the same meaning as the reference answer? "
    "Open-ended phrasing differences are fine; judge the meaning. "
    "Reply with only \"yes\" or \"no\"."
)


def judge_correct_batch(
    items: list[tuple[str, str, str]],
    *,
    base_url: str,
    model: str,
    api_key: str,
    max_workers: int = 16,
) -> list[bool]:
    """LLM-judge a batch of ``(question, reference, candidate)`` -> list[bool].

    Used only for the exact-match FAILURES (open-ended Visual-CoT answers that string
    matching wrongly drops). OpenAI-compatible endpoint (e.g. a local vLLM Kimi). Any
    error on an item -> False (conservative: drop rather than keep a wrong trace).
    """
    from concurrent.futures import ThreadPoolExecutor

    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY")

    def _judge_one(item: tuple[str, str, str]) -> bool:
        question, reference, candidate = item
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": JUDGE_PROMPT.format(
                            question=question, reference=reference, candidate=candidate
                        ),
                    }
                ],
                temperature=0.0,
                max_tokens=8,
            )
            out = (resp.choices[0].message.content or "").strip().lower()
            return out.startswith("yes")
        except Exception:  # noqa: BLE001 - a judge failure just drops the trace
            return False

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(_judge_one, items))

try:  # optional, better math/MCQ matching when available
    from mathruler.grader import grade_answer as _grade_answer
except Exception:  # noqa: BLE001
    _grade_answer = None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="Student to cold-start (also the generator unless --gen_model).")
    ap.add_argument("--gen_model", default=None, help="Model used to GENERATE traces (default: --model_path).")
    ap.add_argument("--dataset_name", required=True)
    ap.add_argument("--dataset_split", default="train")
    ap.add_argument("--answer_field", default="answer")
    ap.add_argument("--bbox_field", default="bbox")
    ap.add_argument("--output_dir", required=True, help="save_to_disk target for the SFT dataset.")
    ap.add_argument("--max_samples", type=int, default=4000, help="Prompts to draw from (boxed rows). Kept traces are fewer (rejection).")
    ap.add_argument("--no_shuffle", action="store_true", help="Take the first max_samples boxed rows in dataset order instead of a seeded random draw (default: shuffle, for representative coverage across domains).")
    ap.add_argument("--num_samples", type=int, default=4, help="vLLM samples per prompt (rejection pool).")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--max_reasoning_chars", type=int, default=1500, help="Truncate the kept reasoning (0=off).")
    ap.add_argument("--bbox_decimals", type=int, default=2)
    ap.add_argument("--keep_incorrect", action="store_true", help="If no attempt is correct, keep one anyway with the GT answer forced (more data, noisier).")
    ap.add_argument("--gen_system_prompt", default="think", help="System-prompt style for GENERATION (think/freecot/reason/none).")
    ap.add_argument("--gen_hint", action="store_true", help="(inject mode) Generate with the hidden-hint teacher prompt (generator silently sees the GT box, forbidden to verbalize it) so the reasoning is GROUNDED. Use with --gen_model <teacher>.")
    ap.add_argument("--trace_mode", default="inject", choices=["inject", "natural"], help="'inject': generate reasoning, bolt a <box>[GT]</box> onto the head. 'natural': the teacher (use --gen_model) is shown the GT box and writes the WHOLE locate trace itself (box woven in) — used verbatim; matches the teacher's thinking pattern.")
    # LLM judge (optional): judge ONLY exact-match failures (open-ended answers) via an
    # OpenAI-compatible endpoint (e.g. a local vLLM Kimi). Exact matches are kept for free.
    ap.add_argument("--judge", default="none", choices=["none", "api"], help="'api': LLM-judge the exact-match failures (recommended for open-ended answers like Visual-CoT).")
    ap.add_argument("--judge_base_url", default="http://10.48.91.210:8000/v1")
    ap.add_argument("--judge_model", default="kimi")
    ap.add_argument("--judge_api_key", default="EMPTY")
    ap.add_argument("--judge_max_workers", type=int, default=16)
    # Data-parallel sharding: run N copies (one per GPU) over disjoint shards of the
    # shuffled dataset, each with its own --output_dir; merge afterwards. max_samples is
    # the GLOBAL target (each shard does max_samples/num_shards).
    ap.add_argument("--shard_index", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    # vLLM
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max_model_len", type=int, default=None)
    ap.add_argument("--limit_images", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def _answer_matches(prediction: Any, reference: Any) -> bool:
    pred = normalize_reference_answer(prediction)
    ref = normalize_reference_answer(reference)
    if not pred or not ref:
        return False
    if _grade_answer is not None:
        try:
            if bool(_grade_answer(pred, ref)):
                return True
        except Exception:  # noqa: BLE001
            pass
    return pred.casefold() == ref.casefold()


def clean_reasoning(text: str, max_chars: int) -> str:
    """The model's reasoning with the boxed answer + any stray tags removed."""
    idx = text.find("\\boxed{")
    if idx >= 0:
        text = text[:idx]
    for tag in ("<think>", "</think>", "<box>", "</box>"):
        text = text.replace(tag, "")
    text = text.strip()
    if max_chars and len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].rstrip() + " ..."
    return text


def build_locate_target(bbox, reasoning: str, answer: str, *, decimals: int) -> str:
    """inject mode: wrap the (grounded, --gen_hint) reasoning, introducing the injected
    GT box in a natural sentence (not a bare line) so it reads as part of the flow."""
    coords = "[" + ", ".join(f"{v:.{decimals}f}" for v in bbox) + "]"
    intro = f"To answer this, I should focus on the region <box>{coords}</box>."
    reasoning = reasoning.strip()
    body = f"{intro}\n{reasoning}" if reasoning else intro
    return f"<think>\n{body}\n</think>\n\\boxed{{{answer}}}"


def build_natural_target(text: str, answer: str, *, max_chars: int) -> str:
    """natural mode: wrap the teacher's box-woven reasoning into the locate format —
    KEEP its ``<box>`` (woven in), drop the trailing ``\\boxed`` + any ``<think>`` tags,
    re-wrap + the GT answer. (Unlike ``clean_reasoning``, does NOT strip ``<box>``.)"""
    idx = text.find("\\boxed{")
    if idx >= 0:
        text = text[:idx]
    text = text.replace("<think>", "").replace("</think>", "").strip()
    text = collapse_extra_boxes(text)  # locate-once: keep only the first <box>
    if max_chars and len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].rstrip() + " ..."
    return f"<think>\n{text}\n</think>\n\\boxed{{{answer}}}"


def main() -> None:
    args = parse_args()
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from datasets import Dataset, Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    gen_model = args.gen_model or args.model_path
    processor = AutoProcessor.from_pretrained(gen_model, trust_remote_code=True, use_fast=True)
    system_prompt = resolve_opd_system_prompt(args.gen_system_prompt)

    # --- collect boxed samples ---------------------------------------------------
    dataset = load_opd_dataset(args.dataset_name, args.dataset_split)
    indices = list(range(len(dataset)))
    if not args.no_shuffle:
        # Representative draw: Visual-CoT is folded from per-domain JSONLs in order, so a
        # first-N slice would be skewed to the first domain(s). Seeded (SAME seed across
        # shards) so the stride-shard below is disjoint and reproducible.
        random.Random(args.seed).shuffle(indices)
    target = args.max_samples
    if args.num_shards > 1:
        indices = indices[args.shard_index :: args.num_shards]  # disjoint stride shard
        target = -(-args.max_samples // args.num_shards)  # ceil; per-shard share of the global target
    samples: list[dict[str, Any]] = []
    for idx in indices:
        row = dataset[int(idx)]
        bbox = parse_bbox_norm(row.get(args.bbox_field))
        if bbox is None:
            continue
        problem = str(row.get("problem", "")).strip()
        answer_raw = row.get(args.answer_field)
        answer = extract_boxed_content(str(answer_raw)) or normalize_reference_answer(answer_raw)
        if not problem or not answer:
            continue
        image = _safe_rgb_image(row.get("images", row.get("image")))
        samples.append({"image": image, "problem": problem, "answer": answer, "bbox": bbox})
        if len(samples) >= target:
            break
    shard_tag = f"shard {args.shard_index}/{args.num_shards}: " if args.num_shards > 1 else ""
    print(f"[coldstart] {shard_tag}{len(samples)} boxed samples to generate (gen_model={gen_model}).", flush=True)
    if not samples:
        raise ValueError("No boxed samples; check --dataset_name / --bbox_field / --answer_field.")

    # --- vLLM generate -----------------------------------------------------------
    llm_kwargs: dict[str, Any] = dict(
        model=gen_model,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": args.limit_images},
        dtype=args.dtype,
        seed=args.seed,
    )
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    engine = LLM(**llm_kwargs)
    sampling = SamplingParams(
        n=max(1, args.num_samples),
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k if args.top_k > 0 else -1,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    if args.trace_mode == "natural":
        # NATURAL: show the generator (teacher) the GT box and have it write the WHOLE
        # locate trace itself (box restated in <box> at the head, reasoning flowing from
        # it). Used verbatim below — no bolt-on, so it matches the teacher's own pattern.
        def gen_messages(s: dict[str, Any]) -> list[dict[str, Any]]:
            coords = "[" + ", ".join(f"{v:.{args.bbox_decimals}f}" for v in s["bbox"]) + "]"
            text = s["problem"] + "\n\n" + NATURAL_GEN_TEMPLATE.format(bbox=coords)
            content = [
                {"type": "image", "image": s["image"]},
                {"type": "text", "text": text},
            ]
            return [
                {"role": "system", "content": [{"type": "text", "text": LOCATE_SYSTEM_PROMPT}]},
                {"role": "user", "content": content},
            ]

        print("[coldstart] NATURAL generation: teacher writes the full locate trace (box shown).", flush=True)
    elif args.gen_hint:
        # GROUNDED inject traces: the generator (teacher) silently sees the GT box via the
        # SAME hidden-hint prompt the OPD teacher uses (no-verbalize clause), so its
        # reasoning is about the evidence region. The GT box is still injected at the head.
        # Lazy import (pulls the trainer stack via baseline.hint) so the module stays light
        # for the CPU sanity check; only hit here, at runtime, when --gen_hint.
        from baseline.hint.opd_hint_collator import (
            HINT_TEMPLATE,
            build_hint_teacher_messages,
            format_bbox_hint,
        )

        def gen_messages(s: dict[str, Any]) -> list[dict[str, Any]]:
            hint = format_bbox_hint(s["bbox"], HINT_TEMPLATE, decimals=args.bbox_decimals)
            return build_hint_teacher_messages(
                s["problem"], s["image"], hint, system_prompt=system_prompt, suffix=""
            )

        print("[coldstart] GROUNDED generation: generator sees the GT box (hidden hint).", flush=True)
    else:
        def gen_messages(s: dict[str, Any]) -> list[dict[str, Any]]:
            return build_general_eval_messages(
                s["problem"], [s["image"]], system_prompt=system_prompt
            )

    requests = [
        {
            "prompt": processor.apply_chat_template(
                gen_messages(s), tokenize=False, add_generation_prompt=True
            ),
            "multi_modal_data": {"image": s["image"]},
        }
        for s in samples
    ]
    outputs = engine.generate(requests, sampling, use_tqdm=True)

    # --- rejection-filter + build targets ---------------------------------------
    natural = args.trace_mode == "natural"
    judge_enabled = args.judge == "api"

    def _valid(cand_text: str) -> bool:
        # usable for a trace: has a boxed answer; natural also needs a parseable <box>.
        if not extract_boxed_content(cand_text):
            return False
        if natural and parse_student_box(cand_text) is None:
            return False
        return True

    def _record(sample: dict[str, Any], cand_text: str) -> dict[str, Any]:
        if natural:
            target = build_natural_target(
                cand_text, sample["answer"], max_chars=args.max_reasoning_chars
            )
        else:
            reasoning = clean_reasoning(cand_text, args.max_reasoning_chars)
            target = build_locate_target(
                sample["bbox"], reasoning, sample["answer"], decimals=args.bbox_decimals
            )
        return {"image": sample["image"], "problem": sample["problem"], "target": target}

    records: list[dict[str, Any]] = []
    n_valid = 0   # samples with >=1 usable candidate (boxed answer [+ <box> for natural])
    n_exact = 0   # kept via exact answer match (free)
    judge_queue: list[tuple[int, str]] = []  # (sample_idx, candidate) — exact-fails to judge
    for si, (sample, output) in enumerate(zip(samples, outputs, strict=True)):
        valid = [c.text for c in output.outputs if _valid(c.text)]
        if valid:
            n_valid += 1
        exact = next(
            (t for t in valid if _answer_matches(extract_boxed_content(t), sample["answer"])),
            None,
        )
        if exact is not None:
            n_exact += 1
            records.append(_record(sample, exact))
        elif judge_enabled and valid:
            judge_queue.append((si, valid[0]))  # judge ONLY exact-fails, one per sample
        elif args.keep_incorrect and not natural and output.outputs:
            records.append(_record(sample, output.outputs[0].text))  # inject force-keep (GT written in)

    # LLM-judge the exact-match failures (open-ended answers string match wrongly drops).
    n_judged = 0
    if judge_queue:
        items = [
            (samples[si]["problem"], samples[si]["answer"], extract_boxed_content(t) or t)
            for si, t in judge_queue
        ]
        verdicts = judge_correct_batch(
            items,
            base_url=args.judge_base_url,
            model=args.judge_model,
            api_key=args.judge_api_key,
            max_workers=args.judge_max_workers,
        )
        for (si, cand_text), ok in zip(judge_queue, verdicts):
            if ok:
                n_judged += 1
                records.append(_record(samples[si], cand_text))

    total = len(samples)
    shard_tag = f"shard {args.shard_index}/{args.num_shards}: " if args.num_shards > 1 else ""
    print(
        f"[coldstart] {shard_tag}{n_valid}/{total} samples had a usable candidate; "
        f"kept {len(records)} = {n_exact} exact-match"
        + (f" + {n_judged} llm-judge (of {len(judge_queue)} exact-fails)" if judge_enabled else "")
        + f"; mode={args.trace_mode}.",
        flush=True,
    )
    if not records:
        raise ValueError(
            "0 cold-start traces kept (see counts above). If n_valid is ~0 the generator "
            "isn't emitting <box> (natural) -> use TRACE_MODE=inject GEN_HINT=true. Else "
            "raise --num_samples, enable --judge api (open-ended answers), or --keep_incorrect."
        )
    out = Dataset.from_list(records).cast_column("image", Image())
    out.save_to_disk(args.output_dir)
    print(f"[coldstart] {shard_tag}saved {len(records)} traces -> {args.output_dir}", flush=True)
    print("[coldstart] example target:\n" + records[0]["target"][:700], flush=True)


if __name__ == "__main__":
    main()
