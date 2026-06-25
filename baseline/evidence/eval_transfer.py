"""Evidence-TRANSFER eval — measure the thing the method actually claims.

Accuracy (MMVP / MMMU) is a *downstream proxy*: it can move because the evidence
loss acts as a regularizer, without any visual evidence actually being
transferred. This eval measures the transfer **directly**, on a held-out set, via
the saliency map itself (migration doc §8):

  1. **energy-in-bbox** — fraction of the student's (positive) saliency mass that
     lands inside the GT evidence box (Saliency_R1's `think_saliency_reward`
     overlap). Higher = the student draws support from the right region.
  2. **pointing-game** — is the argmax saliency patch inside the GT box?
  3. **teacher–student saliency corr** (with `--teacher-model`) — on the SAME
     completion, does the student's map correlate with the teacher's? This is the
     most direct "did the student learn to look where the teacher looks".

The publishable chain needs **energy-in-bbox / corr up on ev_opd vs opd** AND the
accuracy gain concentrated on perception (MMVP). Run this on before / opd /
opd_ev / teacher and compare. Uses `peterant330/saliency-r1-8k` (GT bbox), the
same dataset as the reliance probe.

  uv run python -m baseline.evidence.eval_transfer \
      --model-path runs/opd_ev_qwen3_8b_to_2b/checkpoint-75 --model-name opd_ev \
      --dataset $D/saliency-r1-8k --subsets textvqa,docvqa,gqa,openimages \
      --limit 150 --output-dir eval_outputs/transfer/opd_ev
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from baseline.evidence.evidence_loss import signed_pearson_corr
from baseline.evidence.saliency_engine import compute_token_saliency_maps, resolve_model_parts
from baseline.evidence.sanity_check import _build_inputs, _load_model
from baseline.evidence.span_utils import parse_completion_spans
from baseline.probe.saliency_data import load_saliency_samples


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evidence-transfer eval (saliency vs GT bbox).")
    p.add_argument("--model-path", required=True)
    p.add_argument("--model-name", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--teacher-model", default=None, help="Enables teacher-student saliency corr.")
    p.add_argument("--dataset", default="peterant330/saliency-r1-8k")
    p.add_argument("--split", default="train")
    p.add_argument("--subsets", default="textvqa,textcap,docvqa,infographicsvqa,gqa,openimages")
    p.add_argument("--limit", type=int, default=150, help="Per-subset cap.")
    p.add_argument("--max-bbox-area", type=float, default=0.5)
    p.add_argument("--min-bbox-area", type=float, default=None)
    p.add_argument("--attn", default="eager")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=4, help="Last-N decoder layers (match training).")
    p.add_argument("--layers", default=None, help="Explicit comma layer list (overrides --num-layers).")
    return p.parse_args()


def resolve_layers(parts, num_layers: int, explicit: str | None) -> tuple[int, ...]:
    n = len(parts.text_model.layers)
    if explicit:
        return tuple(l for l in (int(x) for x in explicit.split(",") if x.strip()) if 0 <= l < n)
    if num_layers > 0:
        return tuple(range(max(0, n - num_layers), n))
    return tuple(range(n))


def _bbox_patches(bbox, h_grid: int, w_grid: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    c1 = max(0, int(x1 * w_grid))
    c2 = min(w_grid, max(c1 + 1, int(round(x2 * w_grid))))
    r1 = max(0, int(y1 * h_grid))
    r2 = min(h_grid, max(r1 + 1, int(round(y2 * h_grid))))
    return r1, r2, c1, c2


def energy_and_pointing(map_2d: torch.Tensor, bbox) -> tuple[float | None, bool | None]:
    """Positive-saliency energy-in-bbox + argmax-pointing hit. ``map_2d`` [Hp,Wp]."""
    pos = map_2d.clamp_min(0)
    total = float(pos.sum())
    if total <= 0:
        return None, None
    h_grid, w_grid = pos.shape
    r1, r2, c1, c2 = _bbox_patches(bbox, h_grid, w_grid)
    energy = float(pos[r1:r2, c1:c2].sum()) / total
    flat = int(pos.argmax())
    pr, pc = divmod(flat, w_grid)
    pointing = (r1 <= pr < r2) and (c1 <= pc < c2)
    return energy, bool(pointing)


def _extract_boxed(text: str) -> str | None:
    m = list(re.finditer(r"\\boxed\{", text))
    if not m:
        return None
    i = m[-1].end()
    depth, k = 1, i
    while k < len(text) and depth:
        depth += text[k] == "{"
        depth -= text[k] == "}"
        if not depth:
            break
        k += 1
    return text[i:k].strip()


def _correct(pred_text: str, solution: str) -> bool:
    pred = _extract_boxed(pred_text) or pred_text
    norm = lambda s: " ".join(str(s).strip().lower().split())
    return norm(pred) == norm(solution) or norm(solution) in norm(pred)


def _spans_to_positions(spans, prompt_len, completion_ids, device):
    rs, re_ = spans.reason
    a_start, a_end = spans.answer
    answer_q = (torch.arange(a_start, a_end + 1, device=device) + prompt_len - 1).clamp_min(0)
    reason_k = torch.arange(rs, re_ + 1, device=device) + prompt_len
    reason_q = (torch.arange(rs, re_ + 1, device=device) + prompt_len - 1).clamp_min(0)
    direction = completion_ids[a_start : a_end + 1]
    return answer_q, reason_k, reason_q, direction


def _saliency_map(model, parts, layers, full_ids, inputs, positions, signed=True):
    """One summed saliency map [Hp,Wp] for the completion (no grad)."""
    answer_q, reason_k, reason_q, direction = positions
    visual = (full_ids == parts.image_token_id).nonzero(as_tuple=True)[0]
    grid = inputs["image_grid_thw"][0]
    grid_hw = (int(grid[1]) // parts.spatial_merge_size, int(grid[2]) // parts.spatial_merge_size)
    attn_mask = torch.ones_like(full_ids).unsqueeze(0)
    with torch.no_grad():
        out = model(
            input_ids=full_ids.unsqueeze(0),
            attention_mask=attn_mask,
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"],
            output_attentions=True,
            output_hidden_states=True,
            use_cache=False,
        )
        maps = compute_token_saliency_maps(
            model, out.attentions, out.hidden_states, batch_index=0,
            answer_query_positions=answer_q, reason_key_positions=reason_k,
            reason_query_positions=reason_q, visual_positions=visual,
            direction_ids=direction, grid_hw=grid_hw, layers=layers, signed=signed, parts=parts,
        )
    return maps.sum(0)  # [Hp, Wp]


def main() -> None:
    args = parse_args()
    dtype = getattr(torch, args.dtype)
    subsets = [s for s in args.subsets.split(",") if s.strip()] or None
    name = args.model_name or os.path.basename(args.model_path.rstrip("/"))
    os.makedirs(args.output_dir, exist_ok=True)

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    tokenizer = getattr(processor, "tokenizer", processor)
    model = _load_model(args.model_path, args.attn, dtype)
    parts = resolve_model_parts(model)
    layers = resolve_layers(parts, args.num_layers, args.layers)

    teacher = t_parts = t_layers = t_proc = None
    if args.teacher_model:
        t_proc = AutoProcessor.from_pretrained(args.teacher_model, trust_remote_code=True, use_fast=False)
        teacher = _load_model(args.teacher_model, args.attn, dtype)
        t_parts = resolve_model_parts(teacher)
        t_layers = resolve_layers(t_parts, args.num_layers, args.layers)

    samples = load_saliency_samples(
        args.dataset, args.split, limit=args.limit, subsets=subsets,
        max_bbox_area=args.max_bbox_area, min_bbox_area=args.min_bbox_area,
    )
    print(f"[transfer] {name}: {len(samples)} samples; layers={layers}")

    per_subset = defaultdict(lambda: {"energy": [], "pointing": [], "corr": [], "correct": [], "valid": 0, "n": 0})
    for s in samples:
        st = per_subset[s.subset]
        st["n"] += 1
        inputs = _build_inputs(processor, s.image, s.problem)
        prompt_len = int(inputs["input_ids"].shape[1])
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        full_ids = gen[0]
        completion_ids = full_ids[prompt_len:]
        spans = parse_completion_spans(tokenizer, completion_ids.tolist())
        if not spans.valid:
            continue
        st["valid"] += 1
        st["correct"].append(float(_correct(spans.text, s.solution)))

        pos = _spans_to_positions(spans, prompt_len, completion_ids, full_ids.device)
        smap = _saliency_map(model, parts, layers, full_ids, inputs, pos)
        energy, pointing = energy_and_pointing(smap.float(), s.bbox_norm)
        if energy is not None:
            st["energy"].append(energy)
            st["pointing"].append(float(pointing))

        if teacher is not None:
            t_inputs = _build_inputs(t_proc, s.image, s.problem)
            # teacher scores the SAME completion (student prompt+completion ids reused)
            t_full = torch.cat([t_inputs["input_ids"][0], completion_ids.to(t_inputs["input_ids"].device)], dim=0)
            t_prompt_len = int(t_inputs["input_ids"].shape[1])
            t_pos = _spans_to_positions(spans, t_prompt_len, completion_ids, t_full.device)
            tmap = _saliency_map(teacher, t_parts, t_layers, t_full, t_inputs, t_pos)
            if tmap.shape == smap.shape:
                corr = float(signed_pearson_corr(
                    smap.reshape(1, -1).float().to(tmap.device), tmap.reshape(1, -1).float())[0])
                st["corr"].append(corr)

    def agg(vals):
        return (sum(vals) / len(vals)) if vals else None

    report = {"model_name": name, "model_path": args.model_path, "layers": list(layers),
              "teacher_model": args.teacher_model, "subsets": {}, "overall": {}}
    all_e, all_p, all_c, all_a = [], [], [], []
    for sub, st in sorted(per_subset.items()):
        report["subsets"][sub] = {
            "n": st["n"], "valid": st["valid"],
            "energy_in_bbox": agg(st["energy"]), "pointing": agg(st["pointing"]),
            "teacher_student_corr": agg(st["corr"]), "answer_accuracy": agg(st["correct"]),
        }
        all_e += st["energy"]; all_p += st["pointing"]; all_c += st["corr"]; all_a += st["correct"]
    report["overall"] = {
        "n_valid": len(all_a),
        "energy_in_bbox": agg(all_e), "pointing": agg(all_p),
        "teacher_student_corr": agg(all_c), "answer_accuracy": agg(all_a),
    }
    out = os.path.join(args.output_dir, "transfer.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    o = report["overall"]
    def fmt(x):
        return f"{x:.4f}" if isinstance(x, float) else "—"
    print(f"\n[transfer] {name}  n_valid={o['n_valid']}")
    print(f"  energy_in_bbox = {fmt(o['energy_in_bbox'])}   pointing = {fmt(o['pointing'])}")
    print(f"  teacher_student_corr = {fmt(o['teacher_student_corr'])}   answer_acc = {fmt(o['answer_accuracy'])}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
