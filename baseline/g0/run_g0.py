"""G0 driver — produce per-sample looking/using records over the 3 conditions.

Conditions (saliency-r1-8k, each row carries a GT evidence box):
  * **C1** teacher (8B), image + question, natural CoT.
  * **C2** teacher (8B), image + question + the silent GT-box hint (no-verbalize),
    natural CoT — the hidden-hint privilege channel from ``baseline.hint``.
  * **C3** student (2B), image + question, natural CoT.

Per condition we discover the model's localization heads once (per-head IoU
calibration vs GT, :mod:`localization_heads`) and then, per sample, run ONE
eager grad forward that feeds BOTH probes:
  * GLIMPSE (:mod:`glimpse`) → ``IoU_GL`` + ``vt_ratio`` (the "using" signal);
  * LocalizationHeads (:mod:`localization_heads`) → ``IoU_LH`` (the "looking"
    signal), reading the same forward's attention values.
Plus rule-based answer correctness. Everything is streamed to ``records.jsonl``;
``analyze_g0`` turns it into the four analyses + figures.

Run (on the GPU box):

    CUDA_VISIBLE_DEVICES=0 uv run python -m baseline.g0.run_g0 \
        --student-model $M/Qwen3-VL-2B-Instruct \
        --teacher-model $M/Qwen3-VL-8B-Instruct \
        --dataset $D/saliency-r1-8k --subsets textvqa,docvqa,gqa,openimages \
        --limit 80 --calib-limit 40 --output-dir eval_outputs/g0/run1
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

from baseline.g0 import glimpse as glimpse_mod
from baseline.g0 import localization_heads as lh_mod
from baseline.g0.engine import build_inputs, generate_completion, grad_attention_forward, is_correct, load_g0_model, visual_grid
from baseline.probe.saliency_data import bbox_area, load_saliency_samples


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="G0 grounding diagnostic (looking vs using).")
    p.add_argument("--student-model", required=True)
    p.add_argument("--teacher-model", default=None, help="8B teacher; omit to run C3 only.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dataset", default="peterant330/saliency-r1-8k")
    p.add_argument("--split", default="train")
    p.add_argument("--subsets", default="textvqa,docvqa,gqa,openimages")
    p.add_argument("--limit", type=int, default=80, help="Per-subset eval cap.")
    p.add_argument("--calib-limit", type=int, default=40, help="Per-subset head-calibration cap (subset of eval).")
    p.add_argument("--max-bbox-area", type=float, default=0.5, help="Drop near-whole-image boxes.")
    p.add_argument("--min-bbox-area", type=float, default=None)
    p.add_argument("--conditions", default="c1,c2,c3", help="Subset of c1,c2,c3 to run.")
    # model / attention
    p.add_argument("--attn", default="eager", help="Must be eager for output_attentions.")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-pixels", type=int, default=602112,
                   help="Cap image resolution (#visual tokens); GLIMPSE grad memory ~ S^2.")
    p.add_argument("--min-pixels", type=int, default=None)
    # generation
    p.add_argument("--max-new-tokens", type=int, default=320)
    p.add_argument("--sample", action="store_true", help="Sample completions instead of greedy.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    # LH
    p.add_argument("--top-k-heads", type=int, default=3)
    p.add_argument("--min-layer", type=int, default=2, help="Ignore layers < this when selecting heads.")
    p.add_argument("--lh-sigma", type=float, default=1.0, help="Gaussian sigma for LH assembly.")
    # GLIMPSE
    p.add_argument("--glimpse-layers", default=None, help="Comma layer list (default all).")
    p.add_argument("--glimpse-lambda", type=float, default=1.0, help="Head-weight temperature (Eq.6).")
    p.add_argument("--glimpse-lambda-depth", type=float, default=0.1, help="Layer depth prior (Eq.10).")
    p.add_argument("--threshold", default="mean", choices=["mean", "top_frac"])
    # viz
    p.add_argument("--viz-n", type=int, default=8, help="Save heatmap overlays for the first N eval samples.")
    return p.parse_args()


def _records_path(out_dir: str) -> str:
    return os.path.join(out_dir, "records.jsonl")


def _json_safe(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_head_stats(out_dir: str, stats: lh_mod.HeadStats, tag: str) -> None:
    with open(os.path.join(out_dir, f"head_stats_{tag}.json"), "w") as f:
        json.dump(stats.to_json(), f, indent=2)
    np.savez(
        os.path.join(out_dir, f"head_stats_{tag}.npz"),
        mean_iou=stats.mean_iou,
        mean_attn_sum=stats.mean_attn_sum,
        selection_freq=stats.selection_freq,
        selected_heads=np.array(stats.selected_heads, dtype=np.int64),
    )


def save_viz(out_dir, image, bbox, lh_res, gl_res, tag: str) -> None:
    """Overlay LH + GLIMPSE maps on the image with GT (green) and pred (red) boxes."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
    except Exception:
        return
    W, H = image.size

    def _box(ax, box, color):
        if box is None:
            return
        x1, y1, x2, y2 = box
        ax.add_patch(mpatches.Rectangle(
            (x1 * W, y1 * H), (x2 - x1) * W, (y2 - y1) * H,
            fill=False, edgecolor=color, linewidth=2))

    panels = [("LH (looking)", lh_res.assembled_map, lh_res.pred_box_norm),
              ("GLIMPSE (using)", gl_res.visual_map, gl_res.pred_box_norm)]
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(5 * (len(panels) + 1), 5))
    axes[0].imshow(image); axes[0].set_title("image + GT(green)"); _box(axes[0], bbox, "lime"); axes[0].axis("off")
    for ax, (title, m, pred) in zip(axes[1:], panels):
        ax.imshow(image)
        if m is not None:
            ax.imshow(np.asarray(m), extent=(0, W, H, 0), cmap="jet", alpha=0.5, interpolation="bilinear")
        _box(ax, bbox, "lime"); _box(ax, pred, "red")
        ax.set_title(title); ax.axis("off")
    fig.tight_layout()
    viz_dir = os.path.join(out_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)
    fig.savefig(os.path.join(viz_dir, f"{tag}.png"), dpi=90)
    plt.close(fig)


def run_condition(gm, sample, *, hint, selected_heads, args, glimpse_layers, want_viz):
    """One (sample, condition) → record dict (+ optional viz handles)."""
    bbox = sample.bbox_norm
    inputs = build_inputs(gm, sample.image, sample.problem, hint_bbox=bbox if hint else None)
    prompt_len = int(inputs["input_ids"].shape[1])
    completion_ids, text = generate_completion(
        gm, inputs, max_new_tokens=args.max_new_tokens, do_sample=args.sample,
        temperature=args.temperature, top_p=args.top_p,
        seed=(args.seed + int(sample.sample_id) if str(sample.sample_id).isdigit() else args.seed),
    )
    full_ids = torch.cat([inputs["input_ids"][0], completion_ids.to(inputs["input_ids"].device)])
    correct = is_correct(text, sample.solution)

    out = grad_attention_forward(gm, inputs, full_ids)
    gl_res = glimpse_mod.glimpse_probe(
        gm, inputs, full_ids, prompt_len, completion_ids, bbox,
        layers=glimpse_layers, lam=args.glimpse_lambda, lambda_depth=args.glimpse_lambda_depth,
        threshold=args.threshold, keep_map=want_viz, out=out,
    )
    visual_positions, grid_hw = visual_grid(gm, full_ids, inputs["image_grid_thw"])
    maps = lh_mod.head_visual_maps(out.attentions, prompt_len - 1, visual_positions, grid_hw)
    lh_res = lh_mod.localize_from_maps(maps, bbox, selected_heads, sigma=args.lh_sigma, keep_map=want_viz)
    del out, maps

    record = {
        "sample_id": str(sample.sample_id),
        "subset": sample.subset,
        "model": gm.name,
        "correct": bool(correct),
        "bbox_area": float(bbox_area(bbox)),
        "grid_hw": list(grid_hw),
        "prompt_len": prompt_len,
        "completion_len": int(completion_ids.numel()),
        # looking
        "iou_lh": lh_res.iou_lh,
        "lh_bbox_iou": lh_res.bbox_iou,
        "lh_pointing": lh_res.pointing,
        "lh_energy": lh_res.energy,
        "best_single_iou": lh_res.best_single_iou,
        # using
        "iou_gl": gl_res.iou_gl,
        "gl_bbox_iou": gl_res.bbox_iou,
        "gl_pointing": gl_res.pointing,
        "gl_energy": gl_res.energy,
        "vt_ratio": gl_res.vt_ratio,
        "visual_mass": gl_res.visual_mass,
        "textual_mass": gl_res.textual_mass,
        "self_mass": gl_res.self_mass,
        "completion": text[:600],
    }
    return record, (lh_res, gl_res)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    conditions = [c.strip().lower() for c in args.conditions.split(",") if c.strip()]
    subsets = [s for s in args.subsets.split(",") if s.strip()] or None
    glimpse_layers = (
        tuple(int(x) for x in args.glimpse_layers.split(",") if x.strip())
        if args.glimpse_layers else None
    )

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=_json_safe)

    samples = load_saliency_samples(
        args.dataset, args.split, limit=args.limit, subsets=subsets,
        max_bbox_area=args.max_bbox_area, min_bbox_area=args.min_bbox_area,
    )
    # Calibration subset = first calib_limit per subset (deterministic order).
    seen: dict[str, int] = {}
    calib_samples = []
    for s in samples:
        if seen.get(s.subset, 0) < args.calib_limit:
            calib_samples.append(s)
            seen[s.subset] = seen.get(s.subset, 0) + 1
    print(f"[g0] {len(samples)} eval samples, {len(calib_samples)} calibration samples")

    # ----- load models -----
    pix = dict(min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    student = load_g0_model(args.student_model, "student", attn=args.attn, dtype=args.dtype,
                            device=args.device, **pix)
    teacher = None
    if args.teacher_model and ({"c1", "c2"} & set(conditions)):
        teacher = load_g0_model(args.teacher_model, "teacher", attn=args.attn, dtype=args.dtype,
                                device=args.device, **pix)

    # ----- calibrate localization heads (analysis 1) -----
    calib_kwargs = dict(top_k=args.top_k_heads, min_layer=args.min_layer)
    student_stats = lh_mod.calibrate_heads(student, calib_samples, hint=False, **calib_kwargs)
    save_head_stats(args.output_dir, student_stats, "student")
    teacher_stats = None
    if teacher is not None:
        teacher_stats = lh_mod.calibrate_heads(teacher, calib_samples, hint=False, **calib_kwargs)
        save_head_stats(args.output_dir, teacher_stats, "teacher")

    # condition → (model, hint, selected_heads, tag)
    cond_specs = {}
    if teacher is not None:
        cond_specs["c1"] = (teacher, False, teacher_stats.selected_heads)
        cond_specs["c2"] = (teacher, True, teacher_stats.selected_heads)
    cond_specs["c3"] = (student, False, student_stats.selected_heads)
    conditions = [c for c in conditions if c in cond_specs]

    # ----- eval loop -----
    rec_path = _records_path(args.output_dir)
    n_done, n_skip = 0, 0
    with open(rec_path, "w") as rf:
        for idx, sample in enumerate(samples):
            want_viz = idx < args.viz_n
            for cond in conditions:
                gm, hint, heads = cond_specs[cond]
                try:
                    record, (lh_res, gl_res) = run_condition(
                        gm, sample, hint=hint, selected_heads=heads, args=args,
                        glimpse_layers=glimpse_layers, want_viz=want_viz,
                    )
                except Exception as exc:
                    n_skip += 1
                    print(f"[g0] skip {sample.subset}/{sample.sample_id} {cond}: {exc}")
                    if args.device == "cuda":
                        torch.cuda.empty_cache()
                    continue
                record["condition"] = cond
                rf.write(json.dumps(record, default=_json_safe) + "\n")
                rf.flush()
                n_done += 1
                if want_viz:
                    save_viz(args.output_dir, sample.image, sample.bbox_norm, lh_res, gl_res,
                             tag=f"{sample.subset}_{sample.sample_id}_{cond}")
            if args.device == "cuda" and idx % 4 == 0:
                torch.cuda.empty_cache()
            if idx % 10 == 0:
                print(f"[g0] {idx}/{len(samples)} samples ({n_done} records, {n_skip} skips)")

    summary = quick_summary(rec_path)
    with open(os.path.join(args.output_dir, "summary_quick.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[g0] wrote {n_done} records ({n_skip} skips) → {rec_path}")
    print(json.dumps(summary, indent=2))
    print(f"[g0] next: uv run python -m baseline.g0.analyze_g0 --run-dir {args.output_dir}")


def quick_summary(rec_path: str) -> dict:
    """Per-condition n / accuracy / mean IoU_LH / mean IoU_GL / mean vt_ratio."""
    by_cond: dict[str, dict] = {}
    with open(rec_path) as f:
        for line in f:
            r = json.loads(line)
            c = r["condition"]
            d = by_cond.setdefault(c, {"n": 0, "correct": 0, "iou_lh": [], "iou_gl": [], "vt": []})
            d["n"] += 1
            d["correct"] += int(r["correct"])
            d["iou_lh"].append(r["iou_lh"])
            d["iou_gl"].append(r["iou_gl"])
            if r["vt_ratio"] == r["vt_ratio"]:  # not NaN
                d["vt"].append(r["vt_ratio"])
    out = {}
    for c, d in sorted(by_cond.items()):
        out[c] = {
            "n": d["n"],
            "accuracy": d["correct"] / d["n"] if d["n"] else 0.0,
            "mean_iou_lh": float(np.mean(d["iou_lh"])) if d["iou_lh"] else 0.0,
            "mean_iou_gl": float(np.mean(d["iou_gl"])) if d["iou_gl"] else 0.0,
            "mean_vt_ratio": float(np.mean(d["vt"])) if d["vt"] else 0.0,
        }
    return out


if __name__ == "__main__":
    main()
