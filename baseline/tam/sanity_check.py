"""Standalone sanity check for the differentiable TAM engine.

Runs the TAM logit-lens OUTSIDE the OPD trainer on a single image + OPD prompt +
sampled/teacher-forced response, and verifies what must hold before wiring it into
training:

  1. ``S_S.requires_grad and not S_T.requires_grad`` (student differentiable,
     teacher detached).
  2. ``L_tam.backward()`` puts a **non-zero gradient on the student vision tower**
     (``visual.*``) — the whole point: TAM moves *where the student looks*, so the
     gradient must reach the visual representation, not just the LLM.
  3. It runs under **SDPA** (``--attn sdpa``, the default) with **no
     output_attentions** — TAM needs only ``output_hidden_states``, so it is
     FlashAttention/SDPA-compatible (no eager, no hooks). This is the key
     engineering win over the attention-routing saliency engine.
  4. ``torch.cuda.max_memory_allocated()`` is printed.

With ``--teacher_model`` it also asserts the teacher and student produce the
**same patch grid** (equal ``#visual tokens`` and ``(H_grid, W_grid)``) — the
cross-size TAM bridge is well-defined iff they share a tokenizer + grid.

    uv run python -m baseline.tam.sanity_check \
        --student_model Qwen/Qwen3-VL-2B-Instruct \
        --teacher_model Qwen/Qwen3-VL-8B-Instruct \
        --attn sdpa --max_new_tokens 64
"""

from __future__ import annotations

import argparse
import os
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from baseline.tam.tam_engine import (
    compute_tam_token_maps,
    project_correction,
    resolve_tam_parts,
    sparse_correction_topk,
)
from baseline.tam.tam_losses import tam_alignment_loss

# Well-formed OPD completion (reasoning + \boxed{}); used so the check does not
# depend on whether the base model emits the format when sampled. --sample overrides.
DEFAULT_RESPONSE = (
    "<reason>The image shows a man in a small boat on a river, holding a long "
    "object in his hands that he uses to move the boat through the water.</reason> "
    "The object is a \\boxed{paddle}."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standalone TAM-engine sanity check.")
    # Default to None so they can be resolved from $M (the launcher's offline
    # models dir) in main(); explicit flags still win.
    p.add_argument("--student_model", default=None, help="Student ckpt (default: $M/Qwen3-VL-2B-Instruct or the HF id).")
    p.add_argument("--teacher_model", default=None, help="Teacher ckpt; enables the grid-consistency check (default: $M/Qwen3-VL-8B-Instruct if $M is set).")
    p.add_argument("--attn", default="sdpa", help="attn_implementation (TAM needs NO attention weights, so sdpa/flash are fine).")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--sample", action="store_true", help="Sample on-policy instead of teacher-forcing --response.")
    p.add_argument("--response", default=DEFAULT_RESPONSE, help="Completion to teacher-force.")
    p.add_argument("--question", default="What is the man using to move the boat? Answer briefly.")
    p.add_argument("--image", default=None, help="Image path; default = a synthetic image.")
    p.add_argument(
        "--direction",
        default="token",
        choices=["token", "correction", "hybrid"],
        help="TAM base-map readout direction: token (one-hot W[y_i], original) | "
        "correction (W^T sg(p_T-p_S), the OPD residual — needs --teacher_model) | "
        "hybrid (one-hot + alpha*correction).",
    )
    p.add_argument("--corr_top_k", type=int, default=100, help="top-k |p_T-p_S| vocab entries for the correction direction.")
    p.add_argument("--no_corr_normalize", dest="corr_normalize", action="store_false", default=True, help="Disable L1-normalization of the correction direction.")
    p.add_argument("--no_corr_gate", dest="corr_gate", action="store_false", default=True, help="Disable corr_mass per-token loss weighting (the A' variant).")
    p.add_argument("--corr_alpha", type=float, default=1.0, help="hybrid: one-hot + alpha*correction.")
    p.add_argument("--divergence", default="cosine", choices=["cosine", "js", "l1", "mse"])
    p.add_argument(
        "--denoise",
        default="gaussian",
        choices=["gaussian", "rgf", "none"],
        help="Spatial filter on the maps: gaussian (blur) | rgf (the paper's "
        "Rank-Gaussian Filter — the TAM-MSE-RGF ablation) | none.",
    )
    p.add_argument(
        "--rgf_grad",
        default="hard",
        choices=["hard", "detach_sigma", "gaussian", "identity"],
        help="Student RGF gradient surrogate (only with --denoise rgf): hard (true "
        "grad) | detach_sigma | gaussian | identity. Forward is always exact RGF.",
    )
    p.add_argument(
        "--no_gate",
        dest="use_gate",
        action="store_false",
        default=True,
        help="Disable the concentration gate: align ALL tokens with equal weight.",
    )
    p.add_argument("--no_eci", dest="use_eci", action="store_false", default=True)
    p.add_argument("--no_blur", dest="blur", action="store_false", default=True)
    p.add_argument(
        "--allow_no_vision_grad",
        action="store_true",
        help="Downgrade the 'gradient reached the vision tower' assertion to a "
        "warning (use when the ViT is frozen, or no text precedes the image).",
    )
    return p.parse_args()


def _load_image(path: str | None):
    from PIL import Image

    if path:
        return Image.open(path).convert("RGB")
    import numpy as np

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


def _build_positions(full_ids, prompt_length, completion_ids, image_token_id):
    """visual / candidate / context positions + ids for one sample (all completion tokens)."""
    dev = full_ids.device
    visual_positions = (full_ids == image_token_id).nonzero(as_tuple=True)[0]
    n_comp = int(completion_ids.shape[0])
    candidate_positions = prompt_length + torch.arange(n_comp, device=dev)
    candidate_ids = completion_ids
    context_mask = full_ids != image_token_id
    context_positions = context_mask.nonzero(as_tuple=True)[0]
    context_ids = full_ids.index_select(0, context_positions)
    return visual_positions, candidate_positions, candidate_ids, context_positions, context_ids


def main() -> None:
    args = parse_args()
    # Resolve student/teacher from $M (offline models dir, same convention as
    # scripts/train_opd_tam_qwen3_8b_to_2b.sh) when not given explicitly.
    models_dir = os.environ.get("M", "").rstrip("/")
    if args.student_model is None:
        args.student_model = (
            f"{models_dir}/Qwen3-VL-2B-Instruct" if models_dir else "Qwen/Qwen3-VL-2B-Instruct"
        )
    if args.teacher_model is None and models_dir:
        args.teacher_model = f"{models_dir}/Qwen3-VL-8B-Instruct"
    print(f"[sanity] student_model={args.student_model}")
    print(f"[sanity] teacher_model={args.teacher_model}")
    dtype = getattr(torch, args.dtype)

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.student_model, trust_remote_code=True, use_fast=False)
    tokenizer = getattr(processor, "tokenizer", processor)
    image = _load_image(args.image)

    student = _load_model(args.student_model, args.attn, dtype)
    s_parts = resolve_tam_parts(student)

    inputs = _build_inputs(processor, image, args.question)
    prompt_length = int(inputs["input_ids"].shape[1])

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

    print(f"[sanity] attn_implementation={args.attn} (NO output_attentions requested)")
    print(f"[sanity] completion: {tokenizer.decode(completion_ids, skip_special_tokens=False)[:300]}")

    visual_positions, cand_pos, cand_ids, ctx_pos, ctx_ids = _build_positions(
        full_ids, prompt_length, completion_ids, s_parts.image_token_id
    )
    grid = inputs["image_grid_thw"][0]
    merge = s_parts.spatial_merge_size
    t_dim, h_grid, w_grid = int(grid[0]), int(grid[1]) // merge, int(grid[2]) // merge
    print(
        f"[sanity] student: #visual_tokens={visual_positions.numel()} "
        f"grid=({t_dim},{h_grid},{w_grid}) (== {t_dim * h_grid * w_grid}?) "
        f"#candidate_tokens={cand_ids.numel()}"
    )
    assert t_dim * h_grid * w_grid == visual_positions.numel(), "visual-token/grid mismatch"

    full_ids_b = full_ids.unsqueeze(0)
    attn_mask = torch.ones_like(full_ids_b)
    fwd_kwargs = dict(
        input_ids=full_ids_b,
        attention_mask=attn_mask,
        pixel_values=inputs["pixel_values"],
        image_grid_thw=inputs["image_grid_thw"],
        output_hidden_states=True,  # NOTE: no output_attentions — TAM does not need it.
        use_cache=False,
    )

    torch.cuda.reset_peak_memory_stats()

    # --- student forward (WITH grad); teacher forward (NO grad) ----------------
    student.train()
    s_out = student(**fwd_kwargs)

    teacher = t_parts = t_out = t_visual = None
    if args.teacher_model:
        teacher = _load_model(args.teacher_model, args.attn, dtype)
        t_parts = resolve_tam_parts(teacher)
        t_visual = (full_ids == t_parts.image_token_id).nonzero(as_tuple=True)[0]
        t_grid_hw = (int(grid[1]) // t_parts.spatial_merge_size, int(grid[2]) // t_parts.spatial_merge_size)
        print(f"[sanity] teacher: #visual_tokens={t_visual.numel()} grid_hw={t_grid_hw}")
        assert t_visual.numel() == visual_positions.numel(), (
            f"GRID MISMATCH: teacher {t_visual.numel()} vs student {visual_positions.numel()} "
            "visual tokens — cross-size TAM is undefined for this pair."
        )
        assert t_grid_hw == (h_grid, w_grid), f"GRID MISMATCH: teacher {t_grid_hw} vs student {(h_grid, w_grid)}."
        print("[sanity] GRID CHECK PASSED — teacher/student share the patch grid. ✅")
        with torch.no_grad():
            t_out = teacher(**fwd_kwargs)

    # --- correction readout direction (needs BOTH models' logits) --------------
    d_s = d_t = corr_mass = None
    if args.direction != "token":
        if t_out is None:
            raise SystemExit("--direction correction/hybrid needs --teacher_model (teacher logits).")
        # logits[p-1] is the next-token distribution that produced the token at abs
        # position p (the OPD completion logits, here read off the full-sequence fwd).
        s_logits_cand = s_out.logits[0].index_select(0, cand_pos - 1)
        t_logits_cand = t_out.logits[0].index_select(0, cand_pos - 1)
        vocab = min(s_logits_cand.shape[-1], t_logits_cand.shape[-1])
        u_k, idx, corr_mass = sparse_correction_topk(
            s_logits_cand[..., :vocab], t_logits_cand[..., :vocab],
            top_k=args.corr_top_k, normalize=args.corr_normalize,
        )
        d_s = project_correction(u_k, idx, s_parts.lm_head.weight)
        d_t = project_correction(u_k, idx, t_parts.lm_head.weight)
        if args.direction == "hybrid":
            d_s = s_parts.lm_head.weight.detach().index_select(0, cand_ids).float() + args.corr_alpha * d_s
            d_t = t_parts.lm_head.weight.detach().index_select(0, cand_ids).float() + args.corr_alpha * d_t
        print(
            f"[sanity] direction={args.direction} top_k={args.corr_top_k} "
            f"normalize={args.corr_normalize} corr_mass.mean={corr_mass.mean().item():.4e} "
            f"d_s={tuple(d_s.shape)} d_t={tuple(d_t.shape)} "
            f"d_s.requires_grad={d_s.requires_grad} (want False)"
        )
        assert not d_s.requires_grad and not d_t.requires_grad, (
            "correction direction must be detached (sg(u) + detached lm_head) so the "
            "evidence gradient reaches F^v only, not the logits / unembedding."
        )

    # --- student TAM maps (grad) -----------------------------------------------
    student_maps = compute_tam_token_maps(
        s_out.hidden_states[-1][0],
        s_parts.lm_head.weight,
        visual_positions=visual_positions,
        token_ids=cand_ids,
        token_directions=d_s,
        token_positions=cand_pos,
        context_positions=ctx_pos,
        context_ids=ctx_ids,
        use_eci=args.use_eci,
        detach_lm_head=True,
    )

    # --- teacher TAM maps (NO grad) --------------------------------------------
    if args.teacher_model:
        with torch.no_grad():
            teacher_maps = compute_tam_token_maps(
                t_out.hidden_states[-1][0],
                t_parts.lm_head.weight,
                visual_positions=t_visual,
                token_ids=cand_ids,
                token_directions=d_t,
                token_positions=cand_pos,
                context_positions=ctx_pos,
                context_ids=ctx_ids,
                use_eci=args.use_eci,
                detach_lm_head=True,
            ).detach()
    else:
        teacher_maps = (student_maps.detach() + 0.1 * torch.randn_like(student_maps)).detach()

    # --- checks ----------------------------------------------------------------
    print("\n=== differentiability sanity ===")
    print(f"S_S.requires_grad = {student_maps.requires_grad} (want True)")
    print(f"S_T.requires_grad = {teacher_maps.requires_grad} (want False)")
    assert student_maps.requires_grad and not teacher_maps.requires_grad

    token_weights = (
        corr_mass if (args.direction == "correction" and args.corr_gate) else None
    )
    loss, stats = tam_alignment_loss(
        student_maps,
        teacher_maps,
        grid_thw=(t_dim, h_grid, w_grid),
        divergence=args.divergence,
        denoise=(args.denoise if args.blur else "none"),
        rgf_grad=args.rgf_grad,
        use_gate=args.use_gate,
        token_weights=token_weights,
    )
    print(
        f"[sanity] divergence={args.divergence} denoise="
        f"{args.denoise if args.blur else 'none'} rgf_grad={args.rgf_grad} "
        f"use_gate={args.use_gate} corr_weight={'on' if token_weights is not None else 'off'}"
    )
    print(
        f"L_tam = {loss.item():.6f}  div={stats['tam_div'].item():.4f} "
        f"gate_mean={stats['tam_gate_mean'].item():.4f} "
        f"corr_mass={stats['tam_corr_mass'].item():.4e} n_tokens={stats['tam_n']}"
    )
    loss.backward()

    # The core TAM claim: the gradient reaches the student VISION TOWER.
    vis_probe = None
    any_probe = None
    for name, param in student.named_parameters():
        if param.grad is None:
            continue
        g = param.grad.abs().mean().item()
        if any_probe is None and g > 0:
            any_probe = (name, g)
        if "visual." in name and g > 0:
            vis_probe = (name, g)
            break
    assert any_probe is not None, "zero gradient everywhere — the TAM path is detached!"
    if vis_probe is not None:
        print(f"[sanity] vision-tower grad: {vis_probe[0]} |grad|.mean = {vis_probe[1]:.3e} (want > 0) ✅")
    else:
        msg = (
            "NO gradient reached the student vision tower (no `visual.*` param has "
            f"grad; e.g. {any_probe[0]} |grad|.mean={any_probe[1]:.3e} got grad instead). "
            "The TAM term is moving the LLM/projector but NOT the visual representation "
            "— the core TAM claim is unverified."
        )
        if args.allow_no_vision_grad:
            print(f"[sanity] {msg}  (allowed via --allow_no_vision_grad)")
        else:
            raise AssertionError(
                msg + " Re-run with --allow_no_vision_grad if this is expected "
                "(frozen ViT)."
            )

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"\npeak CUDA memory = {peak_gb:.2f} GB")
    print("[sanity] ALL CHECKS PASSED ✅")


if __name__ == "__main__":
    main()
