"""Post-hoc EAGLE-G0 visualization for selected records.

Reads an existing EAGLE-G0 run directory, selects a small stratified set of cases
from records*.jsonl, then re-runs only those cases with ``want_viz=True`` to write
overlay PNGs. This avoids redoing judge or the full evaluation.

Example:

    python -m baseline.g0.viz_eagle_g0 \
      --run-dir eval_outputs/eagle_g0/qwen3vl-2b-opd \
      --selects wrong,correct --span-modes answer,sentence \
      --rank-condition plain --conditions plain,hint --per-subset 1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from argparse import Namespace
from collections import defaultdict

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from baseline.g0.analyze_g0 import apply_judge, load_records
from baseline.g0.engine import load_g0_model
from baseline.g0.run_eagle_g0 import (
    _condition_prompt,
    run_condition,
    save_eagle_artifacts,
    save_viz,
    save_viz_markdown,
)
from baseline.probe.saliency_data import (
    SaliencySample,
    _avoid_eager_image_decode,
    _load_hf_split,
    _image_source,
    _to_pil,
    bbox_area,
    canon_subset,
    parse_bbox_norm,
)


def _finite(value, default: float = float("inf")) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(v) else v


SELECTORS = {
    "wrong": (lambda r: not bool(r.get("correct", False)), lambda r: _finite(r.get("iou_eagle"))),
    "correct": (lambda r: bool(r.get("correct", False)), lambda r: -_finite(r.get("visual_log_lift"), 0.0)),
    "low_iou_eagle": (lambda r: True, lambda r: _finite(r.get("iou_eagle"))),
    "high_iou_eagle": (lambda r: True, lambda r: -_finite(r.get("iou_eagle"), 0.0)),
    "low_visual_log_lift": (lambda r: True, lambda r: _finite(r.get("visual_log_lift"))),
    "high_visual_log_lift": (lambda r: True, lambda r: -_finite(r.get("visual_log_lift"), 0.0)),
    "low_visual_fraction": (lambda r: True, lambda r: _finite(r.get("visual_fraction"))),
    "high_visual_fraction": (lambda r: True, lambda r: -_finite(r.get("visual_fraction"), 0.0)),
    "random": (lambda r: True, None),
}


def _load_config(run_dir: str) -> dict:
    path = os.path.join(run_dir, "config.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _available_conditions(records: list[dict]) -> list[str]:
    return sorted({str(r.get("condition", "")) for r in records if r.get("condition")})


def _dedupe_records(records: list[dict]) -> list[dict]:
    """Keep one record per model/condition/subset/sample_id.

    Old exploratory runs can leave stale shard files in the same directory. For
    visualization that only needs case selection, duplicate logical records are
    unhelpful; later records in the sorted glob/load order win.
    """
    out: dict[tuple, dict] = {}
    for r in records:
        key = (r.get("model", ""), r.get("condition", ""), r.get("subset", ""), str(r.get("sample_id", "")))
        out[key] = r
    return list(out.values())


def select_cases(
    records: list[dict],
    *,
    select: str,
    rank_condition: str,
    subsets: set[str] | None,
    per_subset: int,
    max_cases: int | None,
    seed: int,
) -> list[dict]:
    filt, key_fn = SELECTORS[select]
    candidates = [
        r for r in records
        if r.get("condition") == rank_condition
        and (subsets is None or canon_subset(r.get("subset", "")) in subsets)
        and filt(r)
    ]
    by_subset: dict[str, list[dict]] = defaultdict(list)
    for r in candidates:
        by_subset[str(r.get("subset", "unknown"))].append(r)

    chosen: list[dict] = []
    for subset, rows in sorted(by_subset.items()):
        if key_fn is None:
            rng = random.Random(f"{seed}:{subset}")
            rng.shuffle(rows)
            picked = rows[:per_subset]
        else:
            picked = sorted(rows, key=key_fn)[:per_subset]
        chosen.extend(picked)

    if max_cases is not None and max_cases >= 0:
        chosen = chosen[:max_cases]
    return chosen


def load_selected_samples(cfg: dict, keys: set[tuple[str, str]]) -> dict[tuple[str, str], SaliencySample]:
    """Load only the images needed for selected ``(subset, sample_id)`` keys."""
    data = _avoid_eager_image_decode(_load_hf_split(cfg["dataset"], cfg.get("split", "train")))
    wanted = {(canon_subset(subset), str(sample_id)) for subset, sample_id in keys}
    out: dict[tuple[str, str], SaliencySample] = {}
    max_bbox_area = cfg.get("max_bbox_area")
    min_bbox_area = cfg.get("min_bbox_area")

    for index in range(len(data)):
        record = data[index]
        subset = str(record.get("dataset", "")).strip() or "unknown"
        sample_id = str(record.get("question_id", index))
        lookup = (canon_subset(subset), sample_id)
        if lookup not in wanted:
            continue

        bbox = parse_bbox_norm(record.get("bbox"))
        if bbox is None:
            continue
        area = bbox_area(bbox)
        if (max_bbox_area is not None and area > max_bbox_area) or (
            min_bbox_area is not None and area < min_bbox_area
        ):
            continue
        problem = str(record.get("problem", "")).strip()
        solution = str(record.get("solution", "")).strip()
        if not problem or not solution:
            continue
        image_field = record.get("image")
        sample = SaliencySample(
            sample_id, subset, problem, solution, _to_pil(image_field), bbox, _image_source(image_field)
        )
        out[(canon_subset(subset), sample_id)] = sample
        if len(out) == len(wanted):
            break
    return out


def _viz_args_from_config(cfg: dict, cli: argparse.Namespace) -> Namespace:
    ns = Namespace(**cfg)
    # Keep post-hoc visualization cheap and compatible with runs made under sdpa.
    ns.grad_probes = False
    ns.salr1 = bool(cli.with_salr1)
    ns.debug_mem = bool(cli.debug_mem)
    if cli.eagle_batch_size is not None:
        ns.eagle_batch_size = cli.eagle_batch_size
    if cli.eagle_image_size is not None:
        ns.eagle_image_size = cli.eagle_image_size
    if cli.max_new_tokens is not None:
        ns.max_new_tokens = cli.max_new_tokens
    if cli.attn is not None:
        ns.attn = cli.attn
    if cli.eagle_token_mode is not None:
        ns.eagle_token_mode = cli.eagle_token_mode
    if cli.eagle_token_limit is not None:
        ns.eagle_token_limit = cli.eagle_token_limit
    if not hasattr(ns, "eagle_token_mode"):
        ns.eagle_token_mode = "span"
    if not hasattr(ns, "eagle_token_limit"):
        ns.eagle_token_limit = 0
    ns.save_eagle_artifacts = bool(cli.save_eagle_artifacts)
    ns.explain_span_mode = "answer"
    if not hasattr(ns, "sample"):
        ns.sample = False
    if not hasattr(ns, "temperature"):
        ns.temperature = 1.0
    if not hasattr(ns, "top_p"):
        ns.top_p = 1.0
    if not hasattr(ns, "hint_mode"):
        ns.hint_mode = "generate"
    return ns


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Render selected post-hoc EAGLE-G0 visualizations.")
    ap.add_argument("--run-dir", required=True, help="One model directory under eval_outputs/eagle_g0.")
    ap.add_argument("--select", default="wrong", choices=sorted(SELECTORS))
    ap.add_argument("--selects", default="", help="Comma list of selectors; overrides --select.")
    ap.add_argument("--span-modes", default="answer",
                    help="Comma list: answer,sentence. answer = boxed/last-K; sentence = whole completion.")
    ap.add_argument("--rank-condition", default="plain")
    ap.add_argument("--conditions", default="plain,hint", help="Conditions to render for each selected sample.")
    ap.add_argument("--subsets", default="", help="Optional comma list; default uses all subsets present in records.")
    ap.add_argument("--per-subset", type=int, default=3)
    ap.add_argument("--max-cases", type=int, default=None, help="Optional global cap after per-subset selection.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-subdir", default=None, help="Default: viz_<select>.")
    ap.add_argument("--no-use-judge", dest="use_judge", action="store_false",
                    help="Do not overlay judgments.jsonl before selecting cases.")
    ap.set_defaults(use_judge=True)
    ap.add_argument("--with-salr1", action="store_true", help="Also render Saliency-R1; slower.")
    ap.add_argument("--attn", default=None, help="Override config attn implementation, e.g. eager or sdpa.")
    ap.add_argument("--eagle-batch-size", type=int, default=None)
    ap.add_argument("--eagle-image-size", type=int, default=None)
    ap.add_argument("--eagle-token-mode", default=None, choices=["span", "per_token_mean", "per_token_max"])
    ap.add_argument("--eagle-token-limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--save-eagle-artifacts", action="store_true")
    ap.add_argument("--debug-mem", action="store_true")
    return ap.parse_args()


def _split_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _resolve_selects(args: argparse.Namespace) -> list[str]:
    selects = _split_csv(args.selects) if args.selects else [args.select]
    bad = [s for s in selects if s not in SELECTORS]
    if bad:
        raise SystemExit(f"[eagle.viz] bad selector(s): {','.join(bad)}")
    return list(dict.fromkeys(selects))


def _resolve_span_modes(args: argparse.Namespace) -> list[str]:
    modes = _split_csv(args.span_modes) or ["answer"]
    bad = [m for m in modes if m not in {"answer", "sentence"}]
    if bad:
        raise SystemExit(f"[eagle.viz] bad span mode(s): {','.join(bad)}")
    return list(dict.fromkeys(modes))


def _subdir_name(base: str | None, select: str, span_mode: str, multi_span: bool) -> str:
    if base:
        if multi_span:
            return f"{base}_{select}_{span_mode}"
        return f"{base}_{select}"
    if span_mode == "answer" and not multi_span:
        return f"viz_{select}"
    return f"viz_{select}_{span_mode}"


def main() -> None:
    args = parse_args()
    cfg = _load_config(args.run_dir)
    records = _dedupe_records(load_records(args.run_dir))
    if args.use_judge:
        apply_judge(args.run_dir, records)

    available = _available_conditions(records)
    if not available:
        raise SystemExit(f"[eagle.viz] no records found in {args.run_dir}")
    rank_condition = args.rank_condition if args.rank_condition in available else available[0]
    render_conditions = [_condition_prompt(c, (0.0, 0.0, 1.0, 1.0))[0]
                         for c in args.conditions.split(",") if c.strip()]
    render_conditions = list(dict.fromkeys(render_conditions))
    if not render_conditions:
        render_conditions = [rank_condition]
    if cfg.get("hint_mode") == "score_plain_y" and any(c != "plain" for c in render_conditions) and "plain" not in render_conditions:
        render_conditions = ["plain"] + render_conditions

    subset_filter = {canon_subset(s) for s in args.subsets.split(",") if s.strip()} or None
    selects = _resolve_selects(args)
    span_modes = _resolve_span_modes(args)
    chosen_by_select: dict[str, list[dict]] = {}
    for select in selects:
        chosen = select_cases(
            records, select=select, rank_condition=rank_condition, subsets=subset_filter,
            per_subset=args.per_subset, max_cases=args.max_cases, seed=args.seed,
        )
        if not chosen:
            print(f"[eagle.viz] WARNING: no cases selected for {select}/{rank_condition}")
            continue
        chosen_by_select[select] = chosen
    if not chosen_by_select:
        raise SystemExit(f"[eagle.viz] no cases selected for {','.join(selects)}/{rank_condition}")

    keys = {
        (str(r["subset"]), str(r["sample_id"]))
        for chosen in chosen_by_select.values()
        for r in chosen
    }
    samples = load_selected_samples(cfg, keys)
    missing = sorted({(canon_subset(subset), sample_id) for subset, sample_id in keys} - set(samples))
    if missing:
        print(f"[eagle.viz] WARNING: {len(missing)} selected sample(s) were not found after dataset filters.")

    ns = _viz_args_from_config(cfg, args)
    gm = load_g0_model(
        cfg["model"], cfg.get("model_name") or os.path.basename(str(cfg["model"]).rstrip("/")),
        attn=ns.attn, dtype=ns.dtype, device=ns.device,
        min_pixels=getattr(ns, "min_pixels", None), max_pixels=getattr(ns, "max_pixels", None),
    )

    explicit_subdir = args.output_subdir
    multi_span = len(span_modes) > 1 or any(mode != "answer" for mode in span_modes)
    wrote = 0
    completion_cache: dict[tuple[str, str, str], tuple] = {}
    output_dirs: set[str] = set()
    for select, chosen in chosen_by_select.items():
        for rec in chosen:
            sample_key = (canon_subset(rec["subset"]), str(rec["sample_id"]))
            sample = samples.get(sample_key)
            if sample is None:
                continue
            for cond in render_conditions:
                cache_key = (sample_key[0], sample_key[1], cond)
                for span_mode in span_modes:
                    ns.explain_span_mode = span_mode
                    reuse = completion_cache.get(cache_key)
                    if cond != "plain" and ns.hint_mode == "score_plain_y":
                        reuse = completion_cache.get((sample_key[0], sample_key[1], "plain"))
                    try:
                        record, (eg, gl_res, lh_first, salr1_res), comp = run_condition(
                            gm, sample, condition=cond, selected_heads=None, args=ns,
                            want_viz=True, reuse_completion=reuse,
                        )
                    except Exception as exc:  # noqa: BLE001
                        print(f"[eagle.viz] skip {sample.subset}/{sample.sample_id} {cond}/{span_mode}: {exc}")
                        continue
                    if cache_key not in completion_cache and not (cond != "plain" and ns.hint_mode == "score_plain_y"):
                        completion_cache[cache_key] = comp

                    subdir = _subdir_name(explicit_subdir, select, span_mode, multi_span)
                    path = save_viz(
                        args.run_dir, sample, eg, gl_res, lh_first,
                        tag=f"{sample.subset}_{sample.sample_id}_{cond}_{span_mode}_{ns.eagle_token_mode}",
                        salr1_res=salr1_res, subdir=subdir,
                    )
                    save_viz_markdown(
                        args.run_dir, sample, record, eg,
                        tag=f"{sample.subset}_{sample.sample_id}_{cond}_{span_mode}_{ns.eagle_token_mode}",
                        viz_path=path,
                    )
                    if ns.save_eagle_artifacts:
                        save_eagle_artifacts(
                            args.run_dir, sample, eg,
                            tag=f"{sample.subset}_{sample.sample_id}_{cond}_{span_mode}_{ns.eagle_token_mode}",
                        )
                    out_dir = os.path.join(args.run_dir, subdir)
                    output_dirs.add(out_dir)
                    if path and os.path.exists(path):
                        print(f"[eagle.viz] wrote {path} "
                              f"(correct={record['correct']}, IoU={record['iou_eagle']:.3f})")
                        wrote += 1
                    else:
                        print(f"[eagle.viz] WARNING: save_viz did not create a PNG for "
                              f"{sample.subset}/{sample.sample_id} {cond}/{span_mode}")

    for out_dir in sorted(output_dirs):
        print(f"[eagle.viz] output -> {os.path.abspath(out_dir)}/")
    print(f"[eagle.viz] wrote {wrote} PNG(s)")


if __name__ == "__main__":
    main()
