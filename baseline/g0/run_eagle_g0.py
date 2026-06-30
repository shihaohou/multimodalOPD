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
from baseline.g0.engine import (
    HIDDEN_HINT_TEMPLATE,
    VISIBLE_HINT_TEMPLATE,
    build_inputs,
    build_messages,
    generate_completion,
    grad_attention_forward,
    is_correct,
    load_g0_model,
    visual_grid,
)
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
    p.add_argument("--conditions", default="plain,hint",
                   help="plain, hint (visible bbox hint), and/or hidden_hint (silent no-verbalize bbox hint).")
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
    p.add_argument("--eagle-token-mode", default="span",
                   choices=["span", "per_token_mean", "per_token_max"],
                   help="span = one EAGLE run over the whole target span; per_token_* = run one map per token and aggregate.")
    p.add_argument("--eagle-token-limit", type=int, default=0,
                   help="Max target tokens to explain in per_token_* mode; 0 means all tokens in the target span.")
    p.add_argument("--save-eagle-artifacts", action="store_true",
                   help="Save aggregate/per-token EAGLE maps and per-token reliance details for later plotting.")
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


def _condition_prompt(condition: str, bbox):
    """Return (condition_name, hint_bbox, hint_template) for an eval condition."""
    cond = str(condition or "plain").strip().lower()
    if cond in {"plain", "natural", "none"}:
        return "plain", None, None
    if cond in {"hint", "visible_hint"}:
        return "hint", bbox, VISIBLE_HINT_TEMPLATE
    if cond in {"hidden_hint", "silent_hint"}:
        return "hidden_hint", bbox, HIDDEN_HINT_TEMPLATE
    raise ValueError(f"unknown condition {condition!r}; use plain,hint,hidden_hint")


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


def run_condition(gm, sample, *, hint=False, condition=None, selected_heads=None, args=None,
                  want_viz=False, reuse_completion=None):
    """One (sample, condition) → (record, viz_handles, (completion_ids, text)).

    ``reuse_completion`` (the plain rollout) skips generation and SCORES that same
    answer under this condition's prompt — the OPD-faithful ``score_plain_y`` mode.
    """
    bbox = sample.bbox_norm
    cond_name, hint_bbox, hint_template = _condition_prompt(condition or ("hint" if hint else "plain"), bbox)
    inputs = build_inputs(gm, sample.image, sample.problem, hint_bbox=hint_bbox, hint_template=hint_template)
    prompt_messages = build_messages(sample.image, sample.problem, hint_bbox=hint_bbox, hint_template=hint_template)
    prompt_text = gm.processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
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
        hint_bbox=hint_bbox, boxed_span=eagle_span, answer_k=args.answer_tokens,
        n_regions=args.n_regions, search_scope=args.search_scope, pending_samples=args.pending_samples,
        update_step=args.update_step, batch_size=args.eagle_batch_size, eagle_image_size=args.eagle_image_size,
        region_mode=args.region_mode, threshold=args.eagle_threshold, top_frac=args.eagle_top_frac,
        token_map_mode=getattr(args, "eagle_token_mode", "span"),
        token_limit=getattr(args, "eagle_token_limit", 0),
        hint_template=hint_template,
        keep_map=want_viz,
    )
    eg.boxed_span_mode = eagle_span_label

    record = {
        "sample_id": str(sample.sample_id), "subset": sample.subset, "model": gm.name,
        "condition": cond_name, "hint_mode": args.hint_mode,
        "correct": bool(is_correct(text, sample.solution)),
        "bbox_area": float(bbox_area(bbox)), "gt_bbox": [float(x) for x in bbox],
        "boxed_span_mode": eg.boxed_span_mode, "region_mode_used": eg.region_mode_used,
        "eagle_target_span_mode": explain_span_mode,
        "eagle_target_span": [int(eagle_span[0]), int(eagle_span[1])],
        "prompt_len": prompt_len, "completion_len": int(completion_ids.numel()),
        # EAGLE — looking (causal) + using (reliance) + faithfulness
        "iou_eagle": eg.iou_eagle, "eagle_bbox_iou": eg.bbox_iou, "pointing_eagle": eg.pointing_eagle,
        "area_eagle": eg.area_eagle, "eagle_energy": eg.energy,
        "pointing_at1": eg.pointing_at1, "energy_in_box": eg.energy_in_box,
        "iou_top10": eg.iou_top10, "iou_top20": eg.iou_top20,
        "visual_reliance": eg.visual_reliance, "text_reliance": eg.text_reliance,
        "visual_fraction": eg.visual_fraction, "visual_log_lift": eg.visual_log_lift,
        "org_score": eg.org_score, "baseline_score": eg.baseline_score,
        "org_logp": eg.org_logp, "baseline_logp": eg.baseline_logp,
        "sufficiency": eg.sufficiency, "necessity": eg.necessity,
        "deletion_logp_drop": eg.deletion_logp_drop,
        "insertion_logp_recovery": eg.insertion_logp_recovery,
        "deletion_logp_drop_top10": eg.deletion_logp_drop_top10,
        "deletion_logp_drop_top20": eg.deletion_logp_drop_top20,
        "insertion_logp_recovery_top10": eg.insertion_logp_recovery_top10,
        "insertion_logp_recovery_top20": eg.insertion_logp_recovery_top20,
        "insertion_recovery_frac_top20": eg.insertion_recovery_frac_top20,
        "n_regions": eg.n_regions,
        "eagle_token_mode": eg.token_map_mode,
        "eagle_token_count": eg.token_map_count,
        "eagle_token_indices": eg.token_indices,
        "eagle_token_details": eg.token_details,
        "eagle_pred_box": list(eg.pred_box_norm) if eg.pred_box_norm else None,
        # judge fields
        "question": sample.problem, "solution": sample.solution, "pred_boxed": pred_boxed,
        "prompt": prompt_text,
        "completion": text,
        "completion_preview": text[:600],
        "image_source": getattr(sample, "image_source", None),
    }
    record.update(grad)
    record.update(salr1)
    return record, (eg, gl_res, lh_first, salr1_res), (completion_ids.detach().cpu(), text)


def _token_plot_inputs(eg):
    words, scores = [], []
    for detail in getattr(eg, "token_details", None) or []:
        raw = str(detail.get("token_text", ""))
        label = raw.replace("\n", "\\n").replace("\t", "\\t")
        label = label if label.strip() else "<space>"
        score = detail.get("visual_log_lift", detail.get("visual_reliance"))
        try:
            score = float(score)
        except (TypeError, ValueError):
            continue
        if np.isfinite(score):
            words.append(label)
            scores.append(score)
    return words, scores


def save_viz(out_dir, sample, eg, gl_res, lh_first, tag, salr1_res=None, subdir="viz"):
    """Save the EAGLE official-style explanation heatmap for the final aggregate map."""
    image = sample.image.convert("RGB")
    viz_dir = os.path.join(out_dir, subdir)
    os.makedirs(viz_dir, exist_ok=True)
    path = os.path.join(viz_dir, f"{tag}.png")
    if getattr(eg, "attribution_map", None) is None:
        print(f"[eagle_g0] viz skip {sample.subset}/{sample.sample_id}: no EAGLE attribution_map")
        return None
    token_words, token_scores = _token_plot_inputs(eg)

    eagle_repo = os.environ.get("EAGLE_REPO", "/Users/houshihao/project/code/EAGLE-master")
    eagle_viz_py = os.path.join(eagle_repo, "visualization", "visualization.py")
    if os.path.exists(eagle_viz_py):
        try:
            import importlib.util
            import tempfile

            spec = importlib.util.spec_from_file_location("_original_eagle_visualization", eagle_viz_py)
            if spec is None or spec.loader is None:
                raise ImportError(f"could not load {eagle_viz_py}")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            fd, image_path = tempfile.mkstemp(prefix="eagle_g0_", suffix=".png")
            os.close(fd)
            try:
                image.save(image_path)
                if token_words and hasattr(mod, "visualize_explanation"):
                    amap = np.asarray(eg.attribution_map, dtype=np.float32)
                    amap = amap - float(np.nanmin(amap))
                    amap = amap / (float(np.nanmax(amap)) + 1e-8)
                    image_bgr = mod.cv2.imread(image_path)
                    amap = mod.cv2.resize(
                        amap,
                        (image_bgr.shape[1], image_bgr.shape[0]),
                        interpolation=mod.cv2.INTER_LINEAR,
                    )
                    amap = mod.norm_image(amap)
                    vis_saliency_map, _ = mod.gen_cam(image_path, amap)
                    mod.visualize_explanation(vis_saliency_map, token_words, token_scores)
                    mod.plt.savefig(path, bbox_inches="tight", pad_inches=0, dpi=600)
                    mod.plt.close()
                elif (
                    getattr(eg, "token_map_mode", "span") == "span"
                    and getattr(eg, "eagle_s_set", None) is not None
                    and isinstance(getattr(eg, "eagle_json_file", None), dict)
                    and "smdl_score" in eg.eagle_json_file
                ):
                    mod.visualization_mllm(image_path, eg.eagle_s_set, eg.eagle_json_file, save_path=path)
                else:
                    mod.visualization_mllm(
                        image_path, np.asarray(eg.attribution_map, dtype=np.float32), {}, save_path=path
                    )
            finally:
                try:
                    os.remove(image_path)
                except OSError:
                    pass
            return path
        except Exception as exc:  # noqa: BLE001
            print(f"[eagle_g0] original EAGLE viz fallback {sample.subset}/{sample.sample_id}: {exc}")

    try:
        import cv2
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[eagle_g0] viz skip {sample.subset}/{sample.sample_id}: cv2/matplotlib import failed: {exc}")
        return None
    mask = np.asarray(eg.attribution_map, dtype=np.float32)
    mask = mask - float(np.nanmin(mask))
    mx = float(np.nanmax(mask))
    mask = mask / (mx + 1e-8)
    mask = (mask * 255.0).astype(np.uint8)

    # Same smoothing/colormap recipe as EAGLE visualization.visualization.gen_cam.
    w, h = mask.shape[1], mask.shape[0]
    image_bgr = cv2.resize(np.asarray(image)[:, :, ::-1], (w, h))
    small_w, small_h = max(1, int(w / 20)), max(1, int(h / 20))
    mask = cv2.resize(mask, (small_w, small_h))
    mask = cv2.resize(mask, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(mask), cv2.COLORMAP_VIRIDIS).astype(np.float32)
    cam = (0.5 * heatmap + 0.5 * image_bgr.astype(np.float32)).astype(np.uint8)

    fixed_width = 6
    top_h = fixed_width * h / max(1, w)
    bottom_h = 1.2 if token_words else 0.0
    fig = plt.figure(figsize=(fixed_width, top_h + bottom_h))
    rows = 2 if token_words else 1
    height_ratios = [top_h, bottom_h] if token_words else [top_h]
    gs = fig.add_gridspec(nrows=rows, ncols=2, height_ratios=height_ratios, width_ratios=[1.0, 0.035])
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    ax.set_aspect("equal", adjustable="box")
    im = ax.imshow(cam[:, :, ::-1])
    cax = fig.add_subplot(gs[0, 1])
    cbar = fig.colorbar(im, cax=cax)
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)
    cbar.ax.tick_params(length=0, labelbottom=False, labelleft=False)
    cbar.ax.set_yticklabels([])
    if token_words:
        ax_txt = fig.add_subplot(gs[1, :])
        ax_txt.axis("off")
        ax_txt.set_xlim(0, 1)
        ax_txt.set_ylim(0, 1)
        norm = plt.Normalize(vmin=min(token_scores), vmax=max(token_scores))
        cmap = plt.get_cmap("bwr")
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = ax_txt.transAxes.inverted()
        x, y = 0.02, 0.72
        for word, score in zip(token_words, token_scores):
            t = ax_txt.text(
                x, y, word, transform=ax_txt.transAxes, fontsize=12,
                ha="left", va="center",
                bbox=dict(facecolor=cmap(norm(score)), edgecolor="none", boxstyle="round,pad=0.25"),
            )
            fig.canvas.draw()
            bb = t.get_window_extent(renderer=renderer)
            (x0, _), (x1, _) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
            w_axes = x1 - x0
            if x + w_axes > 0.98:
                y -= 0.28
                x = 0.02
                t.set_position((x, y))
                x += w_axes + 0.02
            else:
                x += w_axes + 0.02
    fig.subplots_adjust(left=0.02, right=0.985, top=0.985, bottom=0.02, wspace=0.04, hspace=0.04)
    fig.savefig(path, bbox_inches="tight", pad_inches=0, dpi=600)
    plt.close(fig)
    return path


def _md_text(value) -> str:
    text = "" if value is None else str(value)
    return text.replace("```", "'''")


def _md_cell(value) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\n", "\\n").replace("\r", "\\r").replace("|", "\\|")
    return text


def _fmt_float(value) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{v:.4f}" if np.isfinite(v) else ""


def save_viz_markdown(out_dir, sample, record, eg, tag, viz_path=None, subdir="viz_md"):
    md_dir = os.path.join(out_dir, subdir)
    img_dir = os.path.join(md_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    image_copy = os.path.join(img_dir, f"{tag}.png")
    sample.image.convert("RGB").save(image_copy)
    md_path = os.path.join(md_dir, f"{tag}.md")

    image_source = getattr(sample, "image_source", None) or (record or {}).get("image_source")
    response_tokens, response_scores = _token_plot_inputs(eg)
    lines = [
        f"# EAGLE Viz: {tag}",
        "",
        "## Files",
        "",
        f"- Heatmap PNG: `{os.path.abspath(viz_path) if viz_path else ''}`",
        f"- Image source: `{image_source or ''}`",
        f"- Image copy: `{os.path.abspath(image_copy)}`",
        "",
        "## Case",
        "",
        f"- Model: `{(record or {}).get('model', '')}`",
        f"- Condition: `{(record or {}).get('condition', '')}`",
        f"- Subset: `{sample.subset}`",
        f"- Sample ID: `{sample.sample_id}`",
        f"- Correct: `{(record or {}).get('correct', '')}`",
        f"- GT bbox: `{(record or {}).get('gt_bbox', list(sample.bbox_norm))}`",
        f"- Pred box: `{(record or {}).get('eagle_pred_box', '')}`",
        f"- Target span mode: `{(record or {}).get('eagle_target_span_mode', '')}`",
        f"- Target span: `{(record or {}).get('eagle_target_span', '')}`",
        f"- Token map mode: `{(record or {}).get('eagle_token_mode', getattr(eg, 'token_map_mode', ''))}`",
        f"- Token count: `{(record or {}).get('eagle_token_count', getattr(eg, 'token_map_count', ''))}`",
        "",
        "## Metrics",
        "",
        f"- IoU_EAGLE: `{_fmt_float((record or {}).get('iou_eagle'))}`",
        f"- Pointing@1: `{_fmt_float((record or {}).get('pointing_at1'))}`",
        f"- Energy-in-box: `{_fmt_float((record or {}).get('energy_in_box'))}`",
        f"- IoU@top10: `{_fmt_float((record or {}).get('iou_top10'))}`",
        f"- IoU@top20: `{_fmt_float((record or {}).get('iou_top20'))}`",
        f"- Deletion drop@top20: `{_fmt_float((record or {}).get('deletion_logp_drop_top20'))}`",
        f"- Insertion recovery@top20: `{_fmt_float((record or {}).get('insertion_logp_recovery_top20'))}`",
        f"- Visual log lift: `{_fmt_float((record or {}).get('visual_log_lift'))}`",
        "",
        "## Question",
        "",
        "```text",
        _md_text((record or {}).get("question", sample.problem)),
        "```",
        "",
        "## Prompt",
        "",
        "```text",
        _md_text((record or {}).get("prompt", "")),
        "```",
        "",
        "## Ground Truth Answer",
        "",
        "```text",
        _md_text((record or {}).get("solution", sample.solution)),
        "```",
        "",
        "## Response",
        "",
        "```text",
        _md_text((record or {}).get("completion", "")),
        "```",
        "",
        "## Response Token Visual Reliance",
        "",
    ]
    if response_tokens:
        lines += [
            "| idx | token | visual_log_lift |",
            "|-----|-------|-----------------|",
        ]
        for idx, (tok, score) in enumerate(zip(response_tokens, response_scores)):
            lines.append(f"| {idx} | `{_md_cell(tok)}` | {_fmt_float(score)} |")
    else:
        lines.append("(no token scores)")

    details = getattr(eg, "token_details", None) or []
    if details:
        lines += [
            "",
            "## Per-token Details",
            "",
            "| token_index | token | visual_log_lift | visual_fraction | Point@1 | Energy | IoU@20 |",
            "|-------------|-------|-----------------|-----------------|---------|--------|--------|",
        ]
        for d in details:
            lines.append(
                f"| {d.get('token_index', '')} | `{_md_cell(d.get('token_text', ''))}` | "
                f"{_fmt_float(d.get('visual_log_lift'))} | {_fmt_float(d.get('visual_fraction'))} | "
                f"{_fmt_float(d.get('pointing_at1'))} | {_fmt_float(d.get('energy_in_box'))} | "
                f"{_fmt_float(d.get('iou_top20'))} |"
            )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return md_path


def save_eagle_artifacts(out_dir, sample, eg, tag, subdir="eagle_artifacts"):
    art_dir = os.path.join(out_dir, subdir)
    os.makedirs(art_dir, exist_ok=True)
    json_path = os.path.join(art_dir, f"{tag}.json")
    npz_path = os.path.join(art_dir, f"{tag}.npz")
    response_tokens, response_scores = _token_plot_inputs(eg)
    meta = {
        "sample_id": str(sample.sample_id),
        "subset": sample.subset,
        "token_map_mode": eg.token_map_mode,
        "token_map_count": eg.token_map_count,
        "token_indices": eg.token_indices,
        "token_details": eg.token_details,
        "response_tokens": response_tokens,
        "response_visual_log_lift": response_scores,
        "question": sample.problem,
        "solution": sample.solution,
        "image_source": getattr(sample, "image_source", None),
        "metrics": {
            "iou_eagle": eg.iou_eagle,
            "pointing_at1": eg.pointing_at1,
            "energy_in_box": eg.energy_in_box,
            "iou_top10": eg.iou_top10,
            "iou_top20": eg.iou_top20,
            "deletion_logp_drop_top10": eg.deletion_logp_drop_top10,
            "deletion_logp_drop_top20": eg.deletion_logp_drop_top20,
            "insertion_logp_recovery_top10": eg.insertion_logp_recovery_top10,
            "insertion_logp_recovery_top20": eg.insertion_logp_recovery_top20,
            "insertion_recovery_frac_top20": eg.insertion_recovery_frac_top20,
            "visual_log_lift": eg.visual_log_lift,
            "org_logp": eg.org_logp,
            "baseline_logp": eg.baseline_logp,
        },
        "eagle_json_file": eg.eagle_json_file,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=_json_safe)
    arrays = {}
    if eg.attribution_map is not None:
        arrays["aggregate_map"] = np.asarray(eg.attribution_map, dtype=np.float32)
    if eg.pred_mask is not None:
        arrays["pred_mask"] = np.asarray(eg.pred_mask, dtype=np.uint8)
    if eg.token_maps is not None:
        arrays["token_maps"] = np.asarray(eg.token_maps, dtype=np.float32)
    if eg.eagle_s_set is not None:
        arrays["eagle_s_set"] = np.asarray(eg.eagle_s_set, dtype=np.uint8)
    if arrays:
        np.savez_compressed(npz_path, **arrays)
    return json_path, npz_path if arrays else None


def main() -> None:
    args = parse_args()
    name = args.model_name or os.path.basename(args.model.rstrip("/"))
    os.makedirs(args.output_dir, exist_ok=True)
    conditions = [_condition_prompt(c, (0.0, 0.0, 1.0, 1.0))[0] for c in args.conditions.split(",") if c.strip()]
    conditions = list(dict.fromkeys(conditions))
    subsets = [s for s in args.subsets.split(",") if s.strip()] or None
    # score_plain_y needs the plain rollout, so ensure plain is present and first.
    if args.hint_mode == "score_plain_y" and "plain" not in conditions:
        conditions = ["plain"] + conditions
    order = {"plain": 0, "hint": 1, "hidden_hint": 2}
    conditions = sorted(conditions, key=lambda c: order.get(c, 99))
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
            keep_eagle = want_viz or args.save_eagle_artifacts
            used_viz = False
            plain_completion = None
            for cond in conditions:
                reuse = plain_completion if (cond != "plain" and args.hint_mode == "score_plain_y") else None
                try:
                    record, (eg, gl_res, lh_first, salr1_res), comp = run_condition(
                        gm, sample, condition=cond, selected_heads=selected_heads, args=args,
                        want_viz=keep_eagle, reuse_completion=reuse)
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
                    tag = f"{sample.subset}_{sample.sample_id}_{cond}_{args.explain_span_mode}_{args.eagle_token_mode}"
                    path = save_viz(args.output_dir, sample, eg, gl_res, lh_first,
                                    tag=tag, salr1_res=salr1_res)
                    save_viz_markdown(args.output_dir, sample, record, eg, tag, viz_path=path)
                    used_viz = True
                if want_viz or args.save_eagle_artifacts:
                    tag = f"{sample.subset}_{sample.sample_id}_{cond}_{args.explain_span_mode}_{args.eagle_token_mode}"
                    save_eagle_artifacts(args.output_dir, sample, eg, tag)
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
