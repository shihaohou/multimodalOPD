"""EAGLE-G0 driver — faithful causal looking/using records for ONE model.

Per (sample, condition) it:
  1. generates the rollout once (greedy by default) under the OPD prompt;
  2. runs **EAGLE** (:mod:`baseline.g0.eagle_probe`) → causal region attribution
     (``iou_eagle`` / ``pointing_eagle`` / ``area_eagle``) + modality reliance
     (``visual_reliance`` / ``text_reliance`` / ``visual_fraction``) +
     sufficiency / necessity;
  3. optionally (default ON) runs the cheap gradient probes — LH (first + boxed)
     and GLIMPSE (``iou_gl`` + ``vt_ratio*``) — on the *same* rollout so the
     EAGLE-vs-LH comparison is available from one run.

One model per process (``--model``); the multi-launcher fans 4 models across GPU
groups. Conditions are ``plain`` (no hint) and ``hint`` (silent GT-box hint) — the
model identity carries the rest (teacher C1/C2, students C3/C4/C5). Records stream
to ``records*.jsonl`` (sharded), then ``judge_g0`` + ``analyze_eagle_g0``.

Run (one model, sharded across its 2 GPUs handled by the launcher):

    CUDA_VISIBLE_DEVICES=0 uv run python -m baseline.g0.run_eagle_g0 \
        --model $M/CapCurriculum-8B --model-name CapCurriculum-8B \
        --conditions plain,hint --dataset $D/saliency-r1-8k \
        --subsets gqa,openimages,vsr,textvqa --limit 50 \
        --output-dir eval_outputs/eagle_g0/CapCurriculum-8B
"""

from __future__ import annotations

import argparse
import json
import os
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch

from baseline.g0 import eagle_probe as eagle_mod
from baseline.g0 import glimpse as glimpse_mod
from baseline.g0 import localization_heads as lh_mod
from baseline.g0 import salr1_probe as salr1_mod
from baseline.g0.answer_spans import resolve_answer_spans, span_predictor_rows
from baseline.g0.engine import build_inputs, generate_completion, grad_attention_forward, is_correct, load_g0_model, visual_grid
from baseline.g0.run_g0 import _json_safe, resolve_glimpse_layers, save_head_stats
from baseline.probe.saliency_data import bbox_area, load_saliency_samples


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EAGLE-G0 faithful attribution diagnostic (one model).")
    p.add_argument("--model", required=True)
    p.add_argument("--model-name", default=None, help="Tag for records (default: basename of --model).")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dataset", default="peterant330/saliency-r1-8k")
    p.add_argument("--split", default="train")
    p.add_argument("--subsets", default="gqa,openimages,vsr,textvqa",
                   help="Task types (Visual-CoT subsets). Default = object/spatial/OCR signal set.")
    p.add_argument("--limit", type=int, default=50, help="Per-subset cap (EAGLE is expensive; keep small).")
    p.add_argument("--limit-mode", default="per_shard", choices=["per_shard", "global"],
                   help="per_shard keeps the historical behavior: LIMIT applies inside each shard. "
                        "global applies LIMIT before sharding, so extra shards/workers split the same eval set.")
    p.add_argument("--conditions", default="plain,hint", help="plain (no hint) and/or hint (silent GT-box).")
    p.add_argument("--hint-mode", default="generate", choices=["generate", "score_plain_y"],
                   help="generate: the hint condition re-generates under the hint prompt (natural behavior). "
                        "score_plain_y: score the SAME plain rollout under the hint prompt — the OPD-faithful "
                        "'teacher rescoring the student's rollout' setup (table 3 then isolates the hint's effect "
                        "on the identical answer). Forces a plain rollout per sample.")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--max-bbox-area", type=float, default=0.5)
    p.add_argument("--min-bbox-area", type=float, default=None)
    # model / attention
    p.add_argument("--attn", default="eager", help="eager required for grad probes (output_attentions).")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-pixels", type=int, default=602112, help="Grad-probe (full-res) #visual-token cap.")
    p.add_argument("--min-pixels", type=int, default=None)
    # generation
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--sample", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    # EAGLE
    p.add_argument("--eagle-image-size", type=int, default=448, help="Downsize longer side for EAGLE (cost lever).")
    p.add_argument("--n-regions", type=int, default=49, help="#superpixels/tiles to divide the image into.")
    p.add_argument("--search-scope", type=int, default=8)
    p.add_argument("--pending-samples", type=int, default=4)
    p.add_argument("--update-step", type=int, default=10)
    p.add_argument("--eagle-batch-size", type=int, default=8, help="Perturbed images per model forward (VRAM lever).")
    p.add_argument("--region-mode", default="auto", choices=["auto", "slico", "slic", "grid"])
    p.add_argument("--answer-tokens", type=int, default=8, help="Last-K fallback when no \\boxed{} (EAGLE target span).")
    p.add_argument("--eagle-threshold", default="mean", choices=["mean", "top_frac"])
    p.add_argument("--eagle-top-frac", type=float, default=0.25)
    p.add_argument("--explain-span-mode", default="answer", choices=["answer", "sentence"],
                   help="EAGLE target tokens: boxed/last-K answer span, or the whole generated completion.")
    # grad probes (LH + GLIMPSE) on the same rollout
    p.add_argument("--grad-probes", dest="grad_probes", action="store_true", default=True,
                   help="Also compute LH + GLIMPSE (default ON) → EAGLE-vs-LH comparison.")
    p.add_argument("--no-grad-probes", dest="grad_probes", action="store_false")
    p.add_argument("--salr1", dest="salr1", action="store_true", default=True,
                   help="Also compute the Saliency-R1 map (default ON) — secondary attribution baseline.")
    p.add_argument("--no-salr1", dest="salr1", action="store_false")
    p.add_argument("--salr1-layers", default="all",
                   help="Layers summed for the Saliency-R1 map ('all'|'lastN'|comma list). The paper sums all "
                        "layers; 'all' is the faithful setting (last8 = a cheaper 'SalR1-lite').")
    p.add_argument("--salr1-think-row-mode", default="state", choices=["state", "predictor"],
                   help="state: answer attends to thinking tokens' own rows (official intent); "
                        "predictor: shift -1 to the rows that generated each thinking token (strict causal). Ablation.")
    p.add_argument("--debug-mem", action="store_true",
                   help="Print sequence length + CUDA alloc per sample (to find the OOM source).")
    p.add_argument("--calib-limit", type=int, default=30, help="Per-subset LH head-calibration cap.")
    p.add_argument("--top-k-heads", type=int, default=3)
    p.add_argument("--min-layer", type=int, default=2)
    p.add_argument("--lh-sigma", type=float, default=1.0)
    p.add_argument("--glimpse-layers", default="last8")
    p.add_argument("--glimpse-lambda", type=float, default=1.0)
    p.add_argument("--glimpse-lambda-depth", type=float, default=0.1)
    p.add_argument("--threshold", default="mean", choices=["mean", "top_frac"])
    # viz
    p.add_argument("--viz-per-subset", type=int, default=2, help="Save EAGLE+GLIMPSE+GT overlays for first N per subset.")
    return p.parse_args()


def _records_path(out_dir, num_shards=1, shard_index=0):
    if num_shards and num_shards > 1:
        return os.path.join(out_dir, f"records.shard{shard_index}of{num_shards}.jsonl")
    return os.path.join(out_dir, "records.jsonl")


def grad_probe_block(gm, inputs, full_ids, prompt_len, completion_ids, bbox, spans, selected_heads, args, want_viz):
    """LH (first + boxed) + GLIMPSE on the existing rollout. Returns (dict, gl_res, lh_first, lh_boxed)."""
    device = full_ids.device
    comp_len = int(completion_ids.numel())
    layers = resolve_glimpse_layers(args.glimpse_layers, gm.num_layers)
    out = grad_attention_forward(gm, inputs, full_ids)
    gl = glimpse_mod.glimpse_probe(
        gm, inputs, full_ids, prompt_len, completion_ids, bbox,
        layers=layers, answer_k=args.answer_tokens, boxed_span=spans.primary,
        lam=args.glimpse_lambda, lambda_depth=args.glimpse_lambda_depth,
        threshold=args.threshold, keep_map=want_viz, out=out,
    )
    visual_positions, grid_hw = visual_grid(gm, full_ids, inputs["image_grid_thw"])
    maps = lh_mod.head_visual_maps(out.attentions, prompt_len - 1, visual_positions, grid_hw)
    lh_first = lh_mod.localize_from_maps(maps, bbox, selected_heads, sigma=args.lh_sigma, keep_map=want_viz)
    boxed_rows = span_predictor_rows(prompt_len, spans.primary, device=device)
    boxed_maps = lh_mod.head_visual_maps_avg(out.attentions, boxed_rows, visual_positions, grid_hw)
    lh_boxed = lh_mod.localize_from_maps(boxed_maps, bbox, selected_heads, sigma=args.lh_sigma, keep_map=want_viz)
    del out, maps, boxed_maps
    d = {
        "iou_lh": lh_first.iou_lh, "lh_pointing": lh_first.pointing,
        "iou_lh_boxed": lh_boxed.iou_lh, "lh_boxed_pointing": lh_boxed.pointing,
        "iou_gl": gl.iou_gl, "gl_pointing": gl.pointing,
        "vt_ratio": gl.vt_ratio, "vt_ratio_answer": gl.vt_ratio_answer, "vt_ratio_boxed": gl.vt_ratio_boxed,
        "iou_gl_boxed": gl.iou_gl_boxed,
    }
    return d, gl, lh_first, lh_boxed


@torch.no_grad()
def salr1_block(gm, inputs, full_ids, prompt_len, completion_ids, text, bbox, spans, args, want_viz):
    """Saliency-R1 holistic/direct map (signed) on the rollout. Returns (dict, Salr1Result|None).

    Captures per-layer q/k/v with a hook (NO output_attentions — the full [H,S,S]
    tuple over all layers is what OOMs at long S) and computes attention for only
    the answer/think query rows. Its own no-grad forward, independent of grad probes.
    """
    from baseline.g0.engine import _forward_kwargs

    visual_positions, grid_hw = visual_grid(gm, full_ids, inputs["image_grid_thw"])
    think_span = salr1_mod.parse_think_span(text, completion_ids, gm.tokenizer)
    layers = resolve_glimpse_layers(args.salr1_layers, gm.num_layers)
    kwargs = _forward_kwargs(inputs, full_ids)
    kwargs["output_attentions"] = False  # we compute the needed attention rows ourselves
    kwargs["output_hidden_states"] = True
    out = None
    try:
        with salr1_mod._capture_qkv(gm) as qkv:
            out = gm.model(**kwargs)
            hidden_last = out.hidden_states[-1] if getattr(out, "hidden_states", None) else None
            res = salr1_mod.salr1_probe(
                gm, qkv, hidden_last=hidden_last, visual_positions=visual_positions, grid_hw=grid_hw,
                prompt_len=prompt_len, completion_ids=completion_ids, bbox=bbox,
                answer_span=spans.primary, think_span=think_span, layers=layers,
                think_row_mode=args.salr1_think_row_mode, keep_map=want_viz,
            )
    finally:
        # Free the full-attention/hidden-state forward + captured value states even on
        # error, so a single OOM/skip doesn't accumulate across the loop.
        del out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    d = {
        "salr1_span_mode": res.span_mode,
        "salr1_holistic": int(res.span_mode == "holistic"),
        "salr1_valid": int(bool(res.pos.get("valid", False))),
        "salr1_mass_gt": res.pos["mass_gt"], "salr1_mass_enrich": res.pos.get("mass_enrich"),
        "salr1_pointing": res.pos["pointing"],
        "salr1_iou_top20": res.pos["iou_top20"], "salr1_iou_top30": res.pos["iou_top30"],
        "salr1_area_top20": res.pos["area_top20"], "salr1_entropy": res.pos["entropy"],
        "salr1_neg_mass_gt": res.neg["mass_gt"], "salr1_abs_mass_gt": res.abs["mass_gt"],
        "salr1_abs_iou_top20": res.abs["iou_top20"],
    }
    return d, res


def run_condition(gm, sample, *, hint, selected_heads, args, want_viz, reuse_completion=None):
    """One (sample, condition) → (record, viz_handles, (completion_ids, text)).

    ``reuse_completion`` (the plain rollout) skips generation and SCORES that same
    answer under this condition's prompt — the OPD-faithful ``score_plain_y`` mode.
    """
    bbox = sample.bbox_norm
    inputs = build_inputs(gm, sample.image, sample.problem, hint_bbox=bbox if hint else None)
    prompt_len = int(inputs["input_ids"].shape[1])
    if reuse_completion is not None:
        completion_ids, text = reuse_completion
        completion_ids = completion_ids.to(inputs["input_ids"].device)
    else:
        completion_ids, text = generate_completion(
            gm, inputs, max_new_tokens=args.max_new_tokens, do_sample=args.sample,
            temperature=args.temperature, top_p=args.top_p,
            seed=(args.seed + int(sample.sample_id) if str(sample.sample_id).isdigit() else args.seed),
        )
    if int(completion_ids.numel()) == 0:
        raise ValueError("empty completion")
    device = inputs["input_ids"].device
    full_ids = torch.cat([inputs["input_ids"][0], completion_ids.to(device)])
    spans = resolve_answer_spans(completion_ids, gm.tokenizer, args.answer_tokens)
    explain_span_mode = getattr(args, "explain_span_mode", "answer")
    if explain_span_mode == "sentence":
        eagle_span = (0, int(completion_ids.numel()))
        eagle_span_label = "sentence"
    else:
        eagle_span = spans.primary
        eagle_span_label = spans.mode
    if args.debug_mem and torch.cuda.is_available():
        S = int(full_ids.numel())
        P = int((full_ids == gm.parts.image_token_id).sum())
        print(f"[mem] {sample.subset}/{sample.sample_id} S={S} visual={P} "
              f"alloc={torch.cuda.memory_allocated()/1e9:.1f}G", flush=True)
    try:
        from vigos.answer_utils import extract_boxed_content

        pred_boxed = extract_boxed_content(text) or ""
    except Exception:
        pred_boxed = ""

    gl_res = lh_first = lh_boxed = None
    grad = {}
    if args.grad_probes and selected_heads is not None:
        grad, gl_res, lh_first, lh_boxed = grad_probe_block(
            gm, inputs, full_ids, prompt_len, completion_ids, bbox, spans, selected_heads, args, want_viz)

    salr1 = {}
    salr1_res = None
    if args.salr1:
        try:
            salr1, salr1_res = salr1_block(
                gm, inputs, full_ids, prompt_len, completion_ids, text, bbox, spans, args, want_viz)
        except Exception as exc:  # don't let a salr1 hiccup sink the whole sample
            print(f"[eagle_g0] salr1 skip {sample.subset}/{sample.sample_id}: {exc}")

    eg = eagle_mod.eagle_probe(
        gm, sample.image, sample.problem, bbox, completion_ids,
        hint_bbox=bbox if hint else None, boxed_span=eagle_span, answer_k=args.answer_tokens,
        n_regions=args.n_regions, search_scope=args.search_scope, pending_samples=args.pending_samples,
        update_step=args.update_step, batch_size=args.eagle_batch_size, eagle_image_size=args.eagle_image_size,
        region_mode=args.region_mode, threshold=args.eagle_threshold, top_frac=args.eagle_top_frac,
        keep_map=want_viz,
    )
    eg.boxed_span_mode = eagle_span_label

    record = {
        "sample_id": str(sample.sample_id), "subset": sample.subset, "model": gm.name,
        "condition": "hint" if hint else "plain", "hint_mode": args.hint_mode,
        "correct": bool(is_correct(text, sample.solution)),
        "bbox_area": float(bbox_area(bbox)), "gt_bbox": [float(x) for x in bbox],
        "boxed_span_mode": eg.boxed_span_mode, "region_mode_used": eg.region_mode_used,
        "eagle_target_span_mode": explain_span_mode,
        "eagle_target_span": [int(eagle_span[0]), int(eagle_span[1])],
        "prompt_len": prompt_len, "completion_len": int(completion_ids.numel()),
        # EAGLE — looking (causal) + using (reliance) + faithfulness
        "iou_eagle": eg.iou_eagle, "eagle_bbox_iou": eg.bbox_iou, "pointing_eagle": eg.pointing_eagle,
        "area_eagle": eg.area_eagle, "eagle_energy": eg.energy,
        "visual_reliance": eg.visual_reliance, "text_reliance": eg.text_reliance,
        "visual_fraction": eg.visual_fraction, "visual_log_lift": eg.visual_log_lift,
        "org_score": eg.org_score, "baseline_score": eg.baseline_score,
        "org_logp": eg.org_logp, "baseline_logp": eg.baseline_logp,
        "sufficiency": eg.sufficiency, "necessity": eg.necessity, "n_regions": eg.n_regions,
        "eagle_pred_box": list(eg.pred_box_norm) if eg.pred_box_norm else None,
        # judge fields
        "question": sample.problem, "solution": sample.solution, "pred_boxed": pred_boxed,
        "completion": text[:600],
    }
    record.update(grad)
    record.update(salr1)
    return record, (eg, gl_res, lh_first, salr1_res), (completion_ids.detach().cpu(), text)


def save_viz(out_dir, sample, eg, gl_res, lh_first, tag, salr1_res=None, subdir="viz"):
    """Panels: image+GT · EAGLE heatmap · EAGLE keep · EAGLE masked · GLIMPSE(answer) · LH · Saliency-R1.

    GT box drawn green, predicted box red. "keep"/"masked" show the model's view
    if only the important region were kept / removed (EAGLE's causal region)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        from PIL import Image as _Im
    except Exception as exc:
        print(f"[eagle_g0] viz skip {sample.subset}/{sample.sample_id}: matplotlib import failed: {exc}")
        return None
    image = sample.image.convert("RGB")
    W, H = image.size
    bbox = sample.bbox_norm
    img_arr = np.asarray(image)

    heat_panels = [("EAGLE (causal)", eg.attribution_map, eg.pred_box_norm)]
    if gl_res is not None and getattr(gl_res, "visual_map_boxed", None) is not None:
        heat_panels.append(("GLIMPSE (answer)", gl_res.visual_map_boxed, None))
    if lh_first is not None and getattr(lh_first, "assembled_map", None) is not None:
        heat_panels.append(("LH (first-step)", lh_first.assembled_map, lh_first.pred_box_norm))
    if salr1_res is not None and getattr(salr1_res, "pos_map", None) is not None:
        heat_panels.append((f"Saliency-R1 ({salr1_res.span_mode})", salr1_res.pos_map, None))

    # EAGLE keep / masked images (resize the downsized region mask to image size).
    keep_arr = masked_arr = None
    if getattr(eg, "pred_mask", None) is not None:
        mask_img = _Im.fromarray((eg.pred_mask.astype(np.uint8) * 255)).resize((W, H), _Im.NEAREST)
        mfull = (np.asarray(mask_img) > 127)[:, :, None]
        keep_arr = img_arr * mfull
        masked_arr = img_arr * (~mfull)

    def _box(ax, box, color):
        if box is None:
            return
        x1, y1, x2, y2 = box
        ax.add_patch(mpatches.Rectangle((x1 * W, y1 * H), (x2 - x1) * W, (y2 - y1) * H,
                                        fill=False, edgecolor=color, linewidth=2))

    n = 1 + len(heat_panels) + (2 if keep_arr is not None else 0)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5))
    if n == 1:
        axes = [axes]
    ax_i = 0
    axes[ax_i].imshow(image); axes[ax_i].set_title("image + GT(green)"); _box(axes[ax_i], bbox, "lime"); axes[ax_i].axis("off"); ax_i += 1
    for title, m, pred in heat_panels:
        ax = axes[ax_i]; ax_i += 1
        ax.imshow(image)
        if m is not None:
            ax.imshow(np.asarray(m), extent=(0, W, H, 0), cmap="jet", alpha=0.5, interpolation="bilinear")
        _box(ax, bbox, "lime"); _box(ax, pred, "red"); ax.set_title(title); ax.axis("off")
    if keep_arr is not None:
        axes[ax_i].imshow(keep_arr); axes[ax_i].set_title("EAGLE keep"); _box(axes[ax_i], bbox, "lime"); axes[ax_i].axis("off"); ax_i += 1
        axes[ax_i].imshow(masked_arr); axes[ax_i].set_title("EAGLE masked"); _box(axes[ax_i], bbox, "lime"); axes[ax_i].axis("off"); ax_i += 1
    fig.suptitle(f"{sample.subset} {sample.sample_id} | EAGLE IoU={eg.iou_eagle:.2f} "
                 f"vis_frac={eg.visual_fraction:.2f} suff={eg.sufficiency:.2f} nec={eg.necessity:.2f}", fontsize=11)
    fig.tight_layout()
    viz_dir = os.path.join(out_dir, subdir); os.makedirs(viz_dir, exist_ok=True)
    path = os.path.join(viz_dir, f"{tag}.png")
    fig.savefig(path, dpi=90); plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    name = args.model_name or os.path.basename(args.model.rstrip("/"))
    os.makedirs(args.output_dir, exist_ok=True)
    conditions = [c.strip().lower() for c in args.conditions.split(",") if c.strip()]
    subsets = [s for s in args.subsets.split(",") if s.strip()] or None
    # score_plain_y needs the plain rollout, so ensure plain is present and first.
    if args.hint_mode == "score_plain_y" and "plain" not in conditions:
        conditions = ["plain"] + conditions
    conditions = sorted(conditions, key=lambda c: 0 if c == "plain" else 1)
    is_main = args.shard_index == 0
    if is_main:
        with open(os.path.join(args.output_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2, default=_json_safe)

    # Calibration set (unsharded, first calib_limit/subset) — used both to discover
    # LH heads and to hold out from eval. Load it first so we can drop it cleanly.
    calib = []
    if args.grad_probes:
        calib = load_saliency_samples(
            args.dataset, args.split, limit=args.calib_limit, subsets=subsets,
            max_bbox_area=args.max_bbox_area, min_bbox_area=args.min_bbox_area)
    calib_keys = {(s.subset, s.sample_id) for s in calib}

    # Eval set: load with headroom for the calib drop, then re-cap per subset.
    # In per_shard mode, LIMIT is the effective cap inside each shard. In global
    # mode, the loader caps before sharding, so increasing worker count does not
    # increase total eval size.
    raw_limit = None if args.limit <= 0 else (args.limit + (args.calib_limit if args.grad_probes else 0))
    samples = load_saliency_samples(
        args.dataset, args.split, limit=raw_limit, subsets=subsets,
        max_bbox_area=args.max_bbox_area, min_bbox_area=args.min_bbox_area,
        num_shards=args.num_shards, shard_index=args.shard_index,
        limit_before_shard=(args.limit_mode == "global"),
    )
    samples = [s for s in samples if (s.subset, s.sample_id) not in calib_keys]
    if args.limit > 0:
        from collections import Counter

        cnt: Counter = Counter()
        capped = []
        for s in samples:
            if cnt[s.subset] < args.limit:
                capped.append(s)
                cnt[s.subset] += 1
        samples = capped

    pix = dict(min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    gm = load_g0_model(args.model, name, attn=args.attn, dtype=args.dtype, device=args.device, **pix)

    selected_heads = None
    if args.grad_probes:
        stats = lh_mod.calibrate_heads(gm, calib, hint=False, top_k=args.top_k_heads, min_layer=args.min_layer)
        if is_main:
            save_head_stats(args.output_dir, stats, name.replace("/", "_"))
        selected_heads = stats.selected_heads

    print(f"[eagle_g0] model={name} shard {args.shard_index}/{args.num_shards}: {len(samples)} eval samples "
          f"(limit={args.limit}/subset, mode={args.limit_mode}), conditions={conditions}, hint_mode={args.hint_mode}, "
          f"grad_probes={args.grad_probes}")

    rec_path = _records_path(args.output_dir, args.num_shards, args.shard_index)
    n_done = n_skip = 0
    viz_count: dict[str, int] = {}
    with open(rec_path, "w") as rf:
        for idx, sample in enumerate(samples):
            want_viz = is_main and viz_count.get(sample.subset, 0) < args.viz_per_subset
            used_viz = False
            plain_completion = None
            for cond in conditions:
                hint = cond == "hint"
                reuse = plain_completion if (hint and args.hint_mode == "score_plain_y") else None
                try:
                    record, (eg, gl_res, lh_first, salr1_res), comp = run_condition(
                        gm, sample, hint=hint, selected_heads=selected_heads, args=args,
                        want_viz=want_viz, reuse_completion=reuse)
                except Exception as exc:
                    n_skip += 1
                    print(f"[eagle_g0] skip {sample.subset}/{sample.sample_id} {cond}: {exc}")
                    if args.device == "cuda":
                        torch.cuda.empty_cache()
                    continue
                if cond == "plain":
                    plain_completion = comp
                rf.write(json.dumps(record, default=_json_safe) + "\n"); rf.flush()
                n_done += 1
                if want_viz:
                    save_viz(args.output_dir, sample, eg, gl_res, lh_first,
                             tag=f"{sample.subset}_{sample.sample_id}_{cond}", salr1_res=salr1_res)
                    used_viz = True
            if used_viz:
                viz_count[sample.subset] = viz_count.get(sample.subset, 0) + 1
            if args.device == "cuda":
                torch.cuda.empty_cache()
            if idx % 5 == 0:
                print(f"[eagle_g0] {idx}/{len(samples)} ({n_done} records, {n_skip} skips)")

    print(f"[eagle_g0] model={name} shard {args.shard_index}: wrote {n_done} records ({n_skip} skips) → {rec_path}")
    if is_main:
        print(f"[eagle_g0] when shards finish:  uv run python -m baseline.g0.analyze_eagle_g0 --run-dir {args.output_dir}")


if __name__ == "__main__":
    main()
