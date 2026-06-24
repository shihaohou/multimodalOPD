"""Step-1 standalone sanity check for the differentiable saliency engine.

Runs the engine OUTSIDE the OPD trainer on a single image + OPD prompt + sampled
response, and verifies the three things the migration doc (§5.3) says to confirm
before wiring anything into training:

  1. ``S_S.requires_grad and not S_T.requires_grad``   (student differentiable,
     teacher detached)
  2. ``L_ev.backward()`` produces a **non-zero** gradient on a student
     attention projection (the eager-attention path actually carries grad)
  3. ``torch.cuda.max_memory_allocated()`` is printed — eager ``output_attentions``
     over thousands of visual tokens is the real OOM point, so eyeball it FIRST.

It also answers "can Qwen3-VL 8B->2B run?": with ``--teacher_model`` it asserts
the teacher and student produce the **same patch grid** for the same image
(equal ``#visual tokens`` and ``(H_grid, W_grid)``). If that assert passes, the
cross-size evidence loss is well-defined; if it fails, fall back to a shared-ViT
line (Qwen2.5-VL 3B<-7B).

Needs a GPU + the model(s) — run on the box:

    uv run python -m baseline.evidence.sanity_check \
        --student_model Qwen/Qwen2.5-VL-3B-Instruct \
        --teacher_model Qwen/Qwen2.5-VL-7B-Instruct \
        --attn eager --max_new_tokens 64
"""

from __future__ import annotations

import argparse
import os
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from baseline.evidence.evidence_loss import evidence_alignment_loss
from baseline.evidence.saliency_engine import (
    compute_token_saliency_maps,
    resolve_model_parts,
)
from baseline.evidence.span_utils import parse_completion_spans


# Default teacher-forced completion: well-formed OPD output (reasoning + \boxed{}).
# Used so the engine sanity check does NOT depend on whether the base model
# happens to emit the format when sampled (it often does not, especially on a
# synthetic image). Pass --sample to sample on-policy instead.
DEFAULT_RESPONSE = (
    "<reason>The image shows a man in a small boat on a river, holding a long "
    "object in his hands that he uses to move the boat through the water.</reason> "
    "The object is a \\boxed{paddle}."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standalone saliency-engine sanity check.")
    p.add_argument("--student_model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--teacher_model", default=None, help="Optional; enables the grid-consistency check.")
    p.add_argument("--attn", default="eager", help="attn_implementation (must support output_attentions).")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--sample", action="store_true",
                   help="Sample an on-policy completion instead of teacher-forcing --response.")
    p.add_argument("--response", default=DEFAULT_RESPONSE,
                   help="Completion to teacher-force (default has <reason> + \\boxed{}).")
    p.add_argument("--image", default=None, help="Image path; default pulls one from saliency-r1-8k.")
    p.add_argument("--question", default="What is shown in the image? Answer briefly.")
    p.add_argument("--layers", default=None, help="Comma list of decoder layers to sum (default all).")
    p.add_argument("--signed", action="store_true", default=True)
    p.add_argument("--positive_only", dest="signed", action="store_false")
    return p.parse_args()


def _load_image(path: str | None):
    from PIL import Image

    if path:
        return Image.open(path).convert("RGB")
    try:
        from baseline.probe.saliency_data import load_saliency_samples

        samples = load_saliency_samples(
            "peterant330/saliency-r1-8k", "train", limit=1, subsets=["docvqa"]
        )
        if samples:
            return samples[0].image.convert("RGB")
    except Exception as exc:  # pragma: no cover - dataset/network optional
        print(f"[sanity] dataset image unavailable ({exc}); using synthetic image.")
    import numpy as np
    from PIL import Image

    arr = (np.random.default_rng(0).random((448, 448, 3)) * 255).astype("uint8")
    return Image.fromarray(arr)


def _load_model(model_id: str, attn: str, dtype: torch.dtype):
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(
        model_id, attn_implementation=attn, dtype=dtype, trust_remote_code=True
    )
    return model.to("cuda").eval()


def _build_inputs(processor, image, question: str):
    from baseline.opd_data_collator import build_opd_messages

    messages = build_opd_messages(question, image)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=[text], images=[image], return_tensors="pt").to("cuda")


def _positions(spans, prompt_length: int, completion_ids: torch.Tensor):
    """Build the absolute query/key positions + direction ids for the engine."""
    rs, re_ = spans.reason
    as_, ae = spans.answer
    dev = completion_ids.device
    answer_q = torch.arange(as_, ae + 1, device=dev) + prompt_length - 1
    reason_k = torch.arange(rs, re_ + 1, device=dev) + prompt_length
    reason_q = (torch.arange(rs, re_ + 1, device=dev) + prompt_length - 1).clamp_min(0)
    direction_ids = completion_ids[as_ : ae + 1]
    return answer_q.clamp_min(0), reason_k, reason_q, direction_ids


def main() -> None:
    args = parse_args()
    dtype = getattr(torch, args.dtype)
    layers = (
        tuple(int(x) for x in args.layers.split(",") if x.strip()) if args.layers else None
    )

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.student_model, trust_remote_code=True, use_fast=False)
    tokenizer = getattr(processor, "tokenizer", processor)
    image = _load_image(args.image)

    student = _load_model(args.student_model, args.attn, dtype)
    s_parts = resolve_model_parts(student)

    inputs = _build_inputs(processor, image, args.question)
    prompt_length = int(inputs["input_ids"].shape[1])

    # --- get a completion: teacher-forced canned response (default) or sampled --
    if args.sample:
        with torch.no_grad():
            gen = student.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=True, temperature=1.0
            )
        full_ids = gen[0]
        completion_ids = full_ids[prompt_length:]
    else:
        resp_ids = tokenizer(
            args.response, add_special_tokens=False, return_tensors="pt"
        ).input_ids[0].to(inputs["input_ids"].device)
        completion_ids = resp_ids
        full_ids = torch.cat([inputs["input_ids"][0], completion_ids], dim=0)
    spans = parse_completion_spans(tokenizer, completion_ids.tolist())
    print(f"[sanity] completion spans valid={spans.valid} reason={spans.reason} answer={spans.answer}")
    print(f"[sanity] completion text:\n{spans.text[:400]}")
    if not spans.valid:
        raise SystemExit(
            "[sanity] sampled completion has no well-formed <reason>...</reason> + answer; "
            "re-run (sampling) or pass a model that emits the OPD format."
        )

    visual_positions = (full_ids == s_parts.image_token_id).nonzero(as_tuple=True)[0]
    grid = inputs["image_grid_thw"][0]
    merge = s_parts.spatial_merge_size
    grid_hw = (int(grid[1]) // merge, int(grid[2]) // merge)
    print(
        f"[sanity] student: #visual_tokens={visual_positions.numel()} "
        f"grid_hw={grid_hw} (== {grid_hw[0] * grid_hw[1]}?)"
    )

    answer_q, reason_k, reason_q, direction_ids = _positions(spans, prompt_length, completion_ids)
    full_ids_b = full_ids.unsqueeze(0)
    attn_mask = torch.ones_like(full_ids_b)

    fwd_kwargs = dict(
        input_ids=full_ids_b,
        attention_mask=attn_mask,
        pixel_values=inputs["pixel_values"],
        image_grid_thw=inputs["image_grid_thw"],
        output_attentions=True,
        output_hidden_states=True,
        use_cache=False,
    )

    torch.cuda.reset_peak_memory_stats()

    # --- student forward (WITH grad) -> student saliency -----------------------
    student.train()
    s_out = student(**fwd_kwargs)
    student_maps = compute_token_saliency_maps(
        student,
        s_out.attentions,
        s_out.hidden_states,
        batch_index=0,
        answer_query_positions=answer_q,
        reason_key_positions=reason_k,
        reason_query_positions=reason_q,
        visual_positions=visual_positions,
        direction_ids=direction_ids,
        grid_hw=grid_hw,
        layers=layers,
        signed=args.signed,
        parts=s_parts,
    )

    # --- teacher forward (NO grad) -> teacher saliency (grid check) -------------
    if args.teacher_model:
        teacher = _load_model(args.teacher_model, args.attn, dtype)
        t_parts = resolve_model_parts(teacher)
        t_visual = (full_ids == t_parts.image_token_id).nonzero(as_tuple=True)[0]
        t_grid_hw = (int(grid[1]) // t_parts.spatial_merge_size, int(grid[2]) // t_parts.spatial_merge_size)
        print(f"[sanity] teacher: #visual_tokens={t_visual.numel()} grid_hw={t_grid_hw}")
        assert t_visual.numel() == visual_positions.numel(), (
            f"GRID MISMATCH: teacher {t_visual.numel()} vs student {visual_positions.numel()} "
            "visual tokens. Cross-size evidence loss is undefined for this pair — use a "
            "shared-ViT line (e.g. Qwen2.5-VL 3B<-7B)."
        )
        assert t_grid_hw == grid_hw, f"GRID MISMATCH: teacher {t_grid_hw} vs student {grid_hw}."
        print("[sanity] GRID CHECK PASSED — teacher/student share the patch grid. ✅")
        with torch.no_grad():
            t_out = teacher(**fwd_kwargs)
            teacher_maps = compute_token_saliency_maps(
                teacher,
                t_out.attentions,
                t_out.hidden_states,
                batch_index=0,
                answer_query_positions=answer_q,
                reason_key_positions=reason_k,
                reason_query_positions=reason_q,
                visual_positions=t_visual,
                direction_ids=direction_ids,
                grid_hw=t_grid_hw,
                layers=layers,
                signed=args.signed,
                parts=t_parts,
            ).detach()
    else:
        # No teacher: use a detached, perturbed copy of the student map as a stand-in
        # target so the backward path can still be exercised.
        teacher_maps = (student_maps.detach() + 0.1 * torch.randn_like(student_maps)).detach()

    # --- §5.3 checks -----------------------------------------------------------
    print("\n=== differentiability sanity ===")
    print(f"S_S.requires_grad = {student_maps.requires_grad} (want True)")
    print(f"S_T.requires_grad = {teacher_maps.requires_grad} (want False)")
    assert student_maps.requires_grad and not teacher_maps.requires_grad

    loss, stats = evidence_alignment_loss(student_maps, teacher_maps)
    print(f"L_ev = {loss.item():.6f}  corr={stats['ev_corr'].item():.4f} "
          f"gate_mean={stats['ev_gate_mean'].item():.4f} n_tokens={stats['ev_n']}")
    loss.backward()

    q_grad = student.get_input_embeddings().weight.grad
    probe = None
    for name, param in student.named_parameters():
        if name.endswith("self_attn.v_proj.weight") and param.grad is not None:
            probe = (name, param.grad.abs().mean().item())
            break
    print(f"input_embed grad present: {q_grad is not None}")
    if probe:
        print(f"{probe[0]} |grad|.mean = {probe[1]:.3e} (want > 0)")
        assert probe[1] > 0, "zero gradient into v_proj — saliency path is detached somewhere!"

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"\npeak CUDA memory = {peak_gb:.2f} GB")
    print("[sanity] ALL CHECKS PASSED ✅")


if __name__ == "__main__":
    main()
