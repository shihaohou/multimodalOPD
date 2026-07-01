"""Score a fixed student rollout under teacher prompts.

This is the OPD-faithful hidden-hint mechanism probe:

  student rollout y = S(I, Q)
  teacher scores the SAME y under plain / hint / hidden_hint prompts

No teacher free-generation happens here. Optional ``--run-eagle`` runs EAGLE on
that fixed student response under each teacher prompt, which is much slower.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Optional

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch

from baseline.g0 import eagle_probe as eagle_mod
from baseline.g0.analyze_g0 import load_records
from baseline.g0.answer_spans import resolve_answer_spans
from baseline.g0.engine import build_inputs, build_messages, is_correct, load_g0_model
from baseline.g0.run_eagle_g0 import _condition_prompt, save_eagle_artifacts
from baseline.g0.run_g0 import _json_safe
from baseline.probe.saliency_data import canon_subset, load_saliency_samples


@dataclass
class ScoreResult:
    condition: str
    prompt: str
    target_span: tuple[int, int]
    target_len: int
    mean_logp: float
    sum_logp: float
    mean_prob: float
    token_logps: list[float]
    token_probs: list[float]
    token_ids: list[int]
    token_texts: list[str]
    logits: Optional[torch.Tensor] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--teacher-name", default=None)
    parser.add_argument("--student-run-dir", required=True)
    parser.add_argument("--student-condition", default="plain")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--analyze-only", action="store_true", help="Only summarize existing fixed_rollout_records*.jsonl.")
    parser.add_argument("--dataset", default="peterant330/saliency-r1-8k")
    parser.add_argument("--split", default="train")
    parser.add_argument("--subsets", default="", help="Optional comma list. Default: infer from student records.")
    parser.add_argument("--max-records", type=int, default=0, help="0 = all matched records.")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-bbox-area", type=float, default=0.5)
    parser.add_argument("--min-bbox-area", type=float, default=None)
    parser.add_argument("--teacher-conditions", default="plain,hidden_hint")
    parser.add_argument("--target-span-mode", default="sentence", choices=["sentence", "answer"])
    parser.add_argument("--answer-tokens", type=int, default=8)
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=602112)
    parser.add_argument("--no-kl", action="store_true", help="Skip token-distribution KL to reduce memory.")
    parser.add_argument("--kl-token-limit", type=int, default=0, help="0 = all target tokens.")
    parser.add_argument("--hint-delta-threshold", type=float, default=0.5)
    parser.add_argument("--run-eagle", action="store_true")
    parser.add_argument("--eagle-image-size", type=int, default=448)
    parser.add_argument("--n-regions", type=int, default=49)
    parser.add_argument("--search-scope", type=int, default=8)
    parser.add_argument("--pending-samples", type=int, default=4)
    parser.add_argument("--update-step", type=int, default=10)
    parser.add_argument("--eagle-batch-size", type=int, default=128)
    parser.add_argument("--region-mode", default="auto", choices=["auto", "slico", "slic", "grid"])
    parser.add_argument("--eagle-threshold", default="mean", choices=["mean", "top_frac"])
    parser.add_argument("--eagle-top-frac", type=float, default=0.25)
    parser.add_argument("--save-eagle-artifacts", action="store_true")
    return parser.parse_args()


def _mean(values) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    return float(arr.mean()) if arr.size else float("nan")


def _completion_ids(gm, text: str) -> torch.Tensor:
    ids = gm.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    return ids.to(gm.device)


def _target_span(gm, completion_ids: torch.Tensor, mode: str, answer_tokens: int) -> tuple[int, int]:
    n = int(completion_ids.numel())
    if mode == "sentence":
        return 0, n
    span = resolve_answer_spans(completion_ids.detach().cpu(), gm.tokenizer, answer_tokens).primary
    s, e = int(span[0]), int(span[1])
    return max(0, min(s, n)), max(1, min(e, n))


def _score_completion(
    gm,
    sample,
    completion_ids: torch.Tensor,
    *,
    condition: str,
    target_span: tuple[int, int],
    return_logits: bool,
) -> ScoreResult:
    cond_name, hint_bbox, hint_template = _condition_prompt(condition, sample.bbox_norm)
    inputs = build_inputs(gm, sample.image, sample.problem, hint_bbox=hint_bbox, hint_template=hint_template)
    messages = build_messages(sample.image, sample.problem, hint_bbox=hint_bbox, hint_template=hint_template)
    prompt = gm.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    s, e = target_span
    target_len = int(e - s)
    if target_len <= 0:
        raise ValueError("empty target span")

    prompt_ids = inputs["input_ids"][0]
    prefix_completion = completion_ids[: max(e - 1, 0)].to(gm.device)
    eval_ids = torch.cat([prompt_ids, prefix_completion])
    kwargs = {
        "input_ids": eval_ids.unsqueeze(0),
        "attention_mask": torch.ones_like(eval_ids).unsqueeze(0),
        "use_cache": False,
    }
    for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
        if key in inputs:
            kwargs[key] = inputs[key]

    with torch.no_grad():
        try:
            out = gm.model(**kwargs, logits_to_keep=target_len)
            logits = out.logits[0, -target_len:, :].float()
        except TypeError:
            out = gm.model(**kwargs)
            logits = out.logits[0, -target_len:, :].float()
        target_ids = completion_ids[s:e].to(gm.device)
        log_probs = torch.log_softmax(logits, dim=-1)
        token_logps_t = log_probs.gather(1, target_ids.view(-1, 1)).squeeze(1)
        token_probs_t = token_logps_t.exp()

    token_logps = [float(v) for v in token_logps_t.detach().cpu()]
    token_probs = [float(v) for v in token_probs_t.detach().cpu()]
    token_ids = [int(v) for v in target_ids.detach().cpu()]
    token_texts = [
        gm.tokenizer.decode([tid], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for tid in token_ids
    ]
    return ScoreResult(
        condition=cond_name,
        prompt=prompt,
        target_span=(int(s), int(e)),
        target_len=target_len,
        mean_logp=float(np.mean(token_logps)),
        sum_logp=float(np.sum(token_logps)),
        mean_prob=float(np.mean(token_probs)),
        token_logps=token_logps,
        token_probs=token_probs,
        token_ids=token_ids,
        token_texts=token_texts,
        logits=logits.detach() if return_logits else None,
    )


def _kl_rows(p_logits: torch.Tensor, q_logits: torch.Tensor) -> list[float]:
    with torch.no_grad():
        log_p = torch.log_softmax(p_logits.float(), dim=-1)
        log_q = torch.log_softmax(q_logits.float(), dim=-1)
        kl = (log_p.exp() * (log_p - log_q)).sum(dim=-1)
    return [float(v) for v in kl.detach().cpu()]


def _student_records(run_dir: str, condition: str) -> list[dict]:
    records = [r for r in load_records(run_dir) if str(r.get("condition", "")) == condition]
    dedup: dict[tuple[str, str], dict] = {}
    for record in records:
        dedup[(canon_subset(str(record.get("subset", ""))), str(record.get("sample_id", "")))] = record
    return [dedup[key] for key in sorted(dedup)]


def _load_sample_map(args, records: list[dict]) -> dict[tuple[str, str], object]:
    needed = {
        (canon_subset(str(record.get("subset", ""))), str(record.get("sample_id", "")))
        for record in records
    }
    subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]
    if not subsets:
        subsets = sorted({key[0] for key in needed})
    samples = load_saliency_samples(
        args.dataset,
        args.split,
        subsets=subsets,
        max_bbox_area=args.max_bbox_area,
        min_bbox_area=args.min_bbox_area,
    )
    return {
        (canon_subset(sample.subset), str(sample.sample_id)): sample
        for sample in samples
        if (canon_subset(sample.subset), str(sample.sample_id)) in needed
    }


def _eagle_metrics(args, gm, sample, completion_ids: torch.Tensor, condition: str, target_span: tuple[int, int]):
    cond_name, hint_bbox, hint_template = _condition_prompt(condition, sample.bbox_norm)
    eg = eagle_mod.eagle_probe(
        gm,
        sample.image,
        sample.problem,
        sample.bbox_norm,
        completion_ids.detach().cpu(),
        hint_bbox=hint_bbox,
        boxed_span=target_span,
        answer_k=args.answer_tokens,
        n_regions=args.n_regions,
        search_scope=args.search_scope,
        pending_samples=args.pending_samples,
        update_step=args.update_step,
        batch_size=args.eagle_batch_size,
        eagle_image_size=args.eagle_image_size,
        region_mode=args.region_mode,
        threshold=args.eagle_threshold,
        top_frac=args.eagle_top_frac,
        token_map_mode="span",
        token_limit=0,
        hint_template=hint_template,
        keep_map=args.save_eagle_artifacts,
    )
    if args.save_eagle_artifacts:
        tag = f"{sample.subset}_{sample.sample_id}_{cond_name}_fixed_student_{args.target_span_mode}_span"
        save_eagle_artifacts(args.output_dir, sample, eg, tag)
    return {
        "iou_eagle": eg.iou_eagle,
        "pointing_at1": eg.pointing_at1,
        "energy_in_box": eg.energy_in_box,
        "iou_top10": eg.iou_top10,
        "iou_top20": eg.iou_top20,
        "visual_log_lift": eg.visual_log_lift,
        "visual_fraction": eg.visual_fraction,
        "deletion_logp_drop_top20": eg.deletion_logp_drop_top20,
        "insertion_logp_recovery_top20": eg.insertion_logp_recovery_top20,
    }


def _analysis(records: list[dict], teacher_conditions: list[str]) -> dict:
    out: dict[str, object] = {"n": len(records), "teacher_conditions": teacher_conditions}
    pairs = []
    if "plain" in teacher_conditions:
        pairs = [condition for condition in teacher_conditions if condition != "plain"]
    for condition in pairs:
        key = f"{condition}_vs_plain"
        deltas = [r.get("deltas", {}).get(key, {}).get("mean_logp_delta", float("nan")) for r in records]
        kls = [r.get("deltas", {}).get(key, {}).get("kl_condition_to_plain", float("nan")) for r in records]
        ratios = [r.get("deltas", {}).get(key, {}).get("hint_sensitive_token_ratio", float("nan")) for r in records]
        out[key] = {
            "mean_logp_delta": _mean(deltas),
            "kl_condition_to_plain": _mean(kls),
            "hint_sensitive_token_ratio": _mean(ratios),
            "n_positive_delta": int(sum(1 for v in deltas if np.isfinite(v) and v > 0)),
        }
        for correct in (False, True):
            sub = [r for r in records if bool(r.get("student_correct_rule")) is correct]
            sub_deltas = [r.get("deltas", {}).get(key, {}).get("mean_logp_delta", float("nan")) for r in sub]
            out[key][f"mean_logp_delta_student_correct_{str(correct).lower()}"] = _mean(sub_deltas)
    return out


def _load_fixed_records(output_dir: str) -> list[dict]:
    import glob

    paths = sorted(glob.glob(os.path.join(output_dir, "fixed_rollout_records*.jsonl")))
    records = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    return records


def _write_report(output_dir: str, analysis: dict, args) -> str:
    lines = [
        "# Fixed Student Rollout Teacher Scoring",
        "",
        f"- Teacher: `{args.teacher_name or os.path.basename(args.teacher_model.rstrip('/'))}`",
        f"- Student run: `{args.student_run_dir}`",
        f"- Student condition: `{args.student_condition}`",
        f"- Target span mode: `{args.target_span_mode}`",
        f"- Records: `{analysis.get('n', 0)}`",
        f"- EAGLE run: `{bool(args.run_eagle)}`",
        "",
    ]
    for key, value in analysis.items():
        if not key.endswith("_vs_plain") or not isinstance(value, dict):
            continue
        lines.extend(
            [
                f"## {key}",
                "",
                f"- Mean Δlogp: `{value.get('mean_logp_delta'):.4f}`",
                f"- Mean KL(condition || plain): `{value.get('kl_condition_to_plain'):.4f}`",
                f"- Hint-sensitive token ratio: `{value.get('hint_sensitive_token_ratio'):.4f}`",
                f"- Positive Δlogp cases: `{value.get('n_positive_delta')}`",
                f"- Mean Δlogp when student wrong: `{value.get('mean_logp_delta_student_correct_false'):.4f}`",
                f"- Mean Δlogp when student correct: `{value.get('mean_logp_delta_student_correct_true'):.4f}`",
                "",
            ]
        )
    path = os.path.join(output_dir, "fixed_rollout_report.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


def main() -> None:
    args = parse_args()
    args.teacher_name = args.teacher_name or os.path.basename(args.teacher_model.rstrip("/"))
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, default=_json_safe)

    if args.analyze_only:
        output_records = _load_fixed_records(args.output_dir)
        if not output_records:
            raise SystemExit(f"[fixed_rollout] no fixed_rollout_records*.jsonl in {args.output_dir}")
        teacher_conditions = list(output_records[0].get("teacher_conditions", []))
        analysis = _analysis(output_records, teacher_conditions)
        with open(os.path.join(args.output_dir, "fixed_rollout_analysis.json"), "w", encoding="utf-8") as handle:
            json.dump(analysis, handle, indent=2, default=_json_safe)
        report_path = _write_report(args.output_dir, analysis, args)
        print(f"[fixed_rollout] summarized {len(output_records)} records -> {report_path}")
        return

    records = _student_records(args.student_run_dir, args.student_condition)
    if args.max_records > 0:
        records = records[: args.max_records]
    if args.num_shards > 1:
        records = [record for idx, record in enumerate(records) if idx % args.num_shards == args.shard_index]
    if not records:
        raise SystemExit("[fixed_rollout] no student records selected")
    sample_map = _load_sample_map(args, records)
    gm = load_g0_model(
        args.teacher_model,
        args.teacher_name,
        attn=args.attn,
        dtype=args.dtype,
        device=args.device,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    teacher_conditions = [
        _condition_prompt(condition, (0.0, 0.0, 1.0, 1.0))[0]
        for condition in args.teacher_conditions.split(",")
        if condition.strip()
    ]
    teacher_conditions = list(dict.fromkeys(teacher_conditions))
    out_path = os.path.join(
        args.output_dir,
        (
            f"fixed_rollout_records.shard{args.shard_index}of{args.num_shards}.jsonl"
            if args.num_shards > 1
            else "fixed_rollout_records.jsonl"
        ),
    )
    wrote = skipped = 0
    output_records: list[dict] = []
    with open(out_path, "w", encoding="utf-8") as handle:
        for idx, student_record in enumerate(records):
            key = (
                canon_subset(str(student_record.get("subset", ""))),
                str(student_record.get("sample_id", "")),
            )
            sample = sample_map.get(key)
            if sample is None:
                skipped += 1
                print(f"[fixed_rollout] skip missing sample {key[0]}/{key[1]}")
                continue
            completion = str(student_record.get("completion", ""))
            completion_ids = _completion_ids(gm, completion)
            if int(completion_ids.numel()) == 0:
                skipped += 1
                print(f"[fixed_rollout] skip empty completion {sample.subset}/{sample.sample_id}")
                continue
            target_span = _target_span(gm, completion_ids, args.target_span_mode, args.answer_tokens)
            scored: dict[str, ScoreResult] = {}
            for condition in teacher_conditions:
                scored[condition] = _score_completion(
                    gm,
                    sample,
                    completion_ids,
                    condition=condition,
                    target_span=target_span,
                    return_logits=not args.no_kl,
                )

            score_payload = {}
            for condition, score in scored.items():
                score_payload[condition] = {
                    "prompt": score.prompt,
                    "target_span": list(score.target_span),
                    "target_len": score.target_len,
                    "mean_logp": score.mean_logp,
                    "sum_logp": score.sum_logp,
                    "mean_prob": score.mean_prob,
                    "token_logps": score.token_logps,
                    "token_probs": score.token_probs,
                    "token_ids": score.token_ids,
                    "token_texts": score.token_texts,
                }
                if args.run_eagle:
                    score_payload[condition]["eagle"] = _eagle_metrics(
                        args, gm, sample, completion_ids, condition, target_span
                    )

            deltas = {}
            plain = scored.get("plain")
            if plain is not None:
                for condition, score in scored.items():
                    if condition == "plain":
                        continue
                    delta_logps = np.asarray(score.token_logps) - np.asarray(plain.token_logps)
                    pair_key = f"{condition}_vs_plain"
                    deltas[pair_key] = {
                        "mean_logp_delta": float(score.mean_logp - plain.mean_logp),
                        "sum_logp_delta": float(score.sum_logp - plain.sum_logp),
                        "mean_prob_delta": float(score.mean_prob - plain.mean_prob),
                        "hint_sensitive_token_ratio": float(
                            np.mean(np.abs(delta_logps) >= args.hint_delta_threshold)
                        ),
                        "token_logp_deltas": [float(v) for v in delta_logps],
                    }
                    if not args.no_kl and score.logits is not None and plain.logits is not None:
                        limit = args.kl_token_limit if args.kl_token_limit > 0 else score.target_len
                        kl_cp = _kl_rows(score.logits[:limit], plain.logits[:limit])
                        kl_pc = _kl_rows(plain.logits[:limit], score.logits[:limit])
                        deltas[pair_key]["kl_condition_to_plain"] = _mean(kl_cp)
                        deltas[pair_key]["kl_plain_to_condition"] = _mean(kl_pc)
                        deltas[pair_key]["token_kl_condition_to_plain"] = kl_cp
                        deltas[pair_key]["token_kl_plain_to_condition"] = kl_pc

            record = {
                "subset": sample.subset,
                "sample_id": str(sample.sample_id),
                "teacher_model": args.teacher_name,
                "student_model": student_record.get("model"),
                "student_condition": args.student_condition,
                "student_correct_rule": bool(is_correct(completion, sample.solution)),
                "student_record_correct_rule": student_record.get("correct"),
                "question": sample.problem,
                "solution": sample.solution,
                "gt_bbox": list(sample.bbox_norm),
                "image_source": getattr(sample, "image_source", None),
                "student_completion": completion,
                "student_completion_preview": completion[:600],
                "teacher_conditions": teacher_conditions,
                "target_span_mode": args.target_span_mode,
                "teacher_token_completion_len": int(completion_ids.numel()),
                "target_span": list(target_span),
                "scores": score_payload,
                "deltas": deltas,
            }
            handle.write(json.dumps(record, default=_json_safe) + "\n")
            handle.flush()
            output_records.append(record)
            wrote += 1
            if idx % 10 == 0:
                print(f"[fixed_rollout] {idx}/{len(records)} wrote={wrote} skipped={skipped}", flush=True)
            if args.device == "cuda":
                torch.cuda.empty_cache()

    analysis = _analysis(output_records, teacher_conditions)
    with open(os.path.join(args.output_dir, "fixed_rollout_analysis.json"), "w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2, default=_json_safe)
    report_path = _write_report(args.output_dir, analysis, args)
    print(f"[fixed_rollout] wrote {wrote} records ({skipped} skipped) -> {out_path}")
    print(f"[fixed_rollout] wrote analysis/report -> {report_path}")


if __name__ == "__main__":
    main()
