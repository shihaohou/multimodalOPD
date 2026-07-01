"""EAGLE-G0 analysis — the four faithful-attribution tables across models.

Reads one or more EAGLE-G0 run dirs (:mod:`baseline.g0.run_eagle_g0`; one model
each) and writes ``eagle_report.md`` + ``eagle_analysis.json``. The four tables
(GPT's EAGLE-G0 plan):

  1. **Region accuracy** — per model: EAGLE ``iou_eagle`` / ``pointing`` / ``area``
     / ``sufficiency`` / ``necessity`` (causal), with ``iou_lh`` alongside when the
     gradient probes were run (is EAGLE's region better-localized than LH?).
  2. **Looking vs using** (per model, plain) — ``corr(correct, iou_eagle)`` vs
     ``corr(correct, visual_fraction)``: which causal axis tracks correctness?
  3. **Hint mechanism** (models with plain+hint, paired) — Δacc / Δiou_eagle /
     Δvisual_reliance: does the silent hint move the *causal* region, or just the
     answer's image-reliance / accuracy?
  4. **OPD / hint training** (students, plain) — accuracy, ``visual_reliance``,
     ``iou_eagle``, ``necessity`` across base vs OPD vs hint-OPD: did training make
     the answer *depend on the image more* (the success story)?

Plus a per-task-type (Visual-CoT subset) breakdown of table 2.

Run: ``uv run python -m baseline.g0.analyze_eagle_g0 --run-dirs eval_outputs/eagle_g0/* --use-judge``
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from baseline.g0.analyze_g0 import _fmt, _mean, _vals, apply_judge, load_records, safe_corr


def apply_sentence_span_metrics(run_dir: str, records: list[dict]) -> tuple[int, int]:
    path = os.path.join(run_dir, "sentence_span_metrics.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found; run baseline.g0.backfill_sentence_span_metrics first"
        )
    metric_map = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            metric = json.loads(line)
            key = (
                metric.get("model", ""),
                metric.get("condition", ""),
                metric.get("subset", ""),
                str(metric.get("sample_id", "")),
            )
            metric_map[key] = metric

    matched = eligible = 0
    for record in records:
        if str(record.get("eagle_target_span_mode", "")) != "sentence":
            continue
        eligible += 1
        key = (
            record.get("model", ""),
            record.get("condition", ""),
            record.get("subset", ""),
            str(record.get("sample_id", "")),
        )
        metric = metric_map.get(key)
        if metric is None:
            continue
        for field in (
            "iou_eagle",
            "eagle_bbox_iou",
            "pointing_eagle",
            "pointing_at1",
            "area_eagle",
            "eagle_energy",
            "energy_in_box",
            "iou_top10",
            "iou_top20",
            "eagle_pred_box",
        ):
            if field in metric:
                record[field] = metric[field]
        # These top-area interventions were scored against the old aggregate map
        # and cannot be reconstructed without model forwards.
        for field in (
            "deletion_logp_drop",
            "insertion_logp_recovery",
            "deletion_logp_drop_top10",
            "deletion_logp_drop_top20",
            "insertion_logp_recovery_top10",
            "insertion_logp_recovery_top20",
            "insertion_recovery_frac_top20",
        ):
            record[field] = float("nan")
        record["spatial_metric_source"] = "sentence_span_artifact"
        matched += 1
    print(f"[eagle.analyze] applied sentence-span spatial metrics to {matched}/{eligible} records.")
    return matched, eligible


def _load_all(run_dirs: list[str], use_judge: bool, use_sentence_span_metrics: bool) -> list[dict]:
    recs: list[dict] = []
    for d in run_dirs:
        rs = load_records(d)
        if use_judge:
            judged = apply_judge(d, rs)
            if judged != len(rs):
                raise RuntimeError(
                    f"{d}: LLM judge only matched {judged}/{len(rs)} records; "
                    "finish judge_g0 before generating the final report"
                )
        if use_sentence_span_metrics:
            matched, eligible = apply_sentence_span_metrics(d, rs)
            if matched != eligible:
                raise RuntimeError(
                    f"{d}: sentence-span metrics only matched {matched}/{eligible}; "
                    "rerun backfill without --allow-missing"
                )
        for r in rs:
            r.setdefault("_run_dir", d)
        recs.extend(rs)
    return recs


def _by_model(records):
    out = defaultdict(list)
    for r in records:
        out[r.get("model", "?")].append(r)
    return out


def _plain(recs):
    return [r for r in recs if r.get("condition", "plain") == "plain"]


# Canonical subset groups (saliency-r1-8k = Visual-CoT sources). PRIMARY = the
# local-evidence subsets where "looking" is meaningful (object/spatial + scene text);
# the rest are OCR/doc/caption controls. The user asked for a first-5 average and an
# all-10 average — these define the first-5.
PRIMARY5 = ["gqa", "openimages", "v7w", "textvqa", "vsr"]
from baseline.probe.saliency_data import canon_subset  # noqa: E402


def _group_subsets(by_sub: dict, which: str) -> list[str]:
    """Subset names present in ``by_sub`` belonging to a group ('primary5'|'all')."""
    if which == "all":
        return sorted(by_sub)
    prim = {canon_subset(s) for s in PRIMARY5}
    return sorted(s for s in by_sub if canon_subset(s) in prim)


def _pooled_and_macro(recs, subset_names, metrics) -> dict:
    """Two averages over a group of subsets, per metric.

    * **pooled** = mean over all records in the group (size-weighted — dominated by
      big subsets like gqa/flickr30k).
    * **macro**  = mean over per-subset means (each subset counts equally — the
      right "average across task types" when sizes differ 25× across subsets).
    """
    p = _plain(recs)
    in_group = [r for r in p if r.get("subset") in subset_names]
    by_sub = defaultdict(list)
    for r in in_group:
        by_sub[r["subset"]].append(r)
    out = {"subsets": sorted(by_sub), "n_subsets": len(by_sub), "n": len(in_group)}
    for m in metrics:
        pooled = _mean(in_group, m) if m != "accuracy" else (
            float(np.mean([r["correct"] for r in in_group])) if in_group else float("nan"))
        per_sub = []
        for s, rs in by_sub.items():
            v = (float(np.mean([r["correct"] for r in rs])) if m == "accuracy" else _mean(rs, m))
            if v == v:  # not NaN
                per_sub.append(v)
        macro = float(np.mean(per_sub)) if per_sub else float("nan")
        out[f"{m}_pooled"] = pooled
        out[f"{m}_macro"] = macro
    return out


_AVG_METRICS = ["accuracy", "iou_eagle", "pointing_at1", "energy_in_box", "iou_top10", "iou_top20",
                "visual_reliance", "visual_log_lift", "visual_fraction", "sufficiency", "necessity",
                "deletion_logp_drop", "insertion_logp_recovery"]


def group_averages(by_model) -> dict:
    """Per-model first-5 (primary) and all-10 averages (pooled + macro)."""
    out = {}
    for model, recs in by_model.items():
        by_sub = defaultdict(list)
        for r in _plain(recs):
            by_sub[r.get("subset", "?")].append(r)
        if not by_sub:
            continue
        out[model] = {
            "primary5": _pooled_and_macro(recs, _group_subsets(by_sub, "primary5"), _AVG_METRICS),
            "all": _pooled_and_macro(recs, _group_subsets(by_sub, "all"), _AVG_METRICS),
        }
    return out


# --------------------------------------------------- table 1: region accuracy
def table1_region_accuracy(by_model) -> dict:
    out = {}
    for model, recs in by_model.items():
        p = _plain(recs)
        if not p:
            continue
        out[model] = {
            "n": len(p),
            "accuracy": float(np.mean([r["correct"] for r in p])),
            "iou_eagle": _mean(p, "iou_eagle"),
            "pointing_eagle": _mean(p, "pointing_eagle"),
            "pointing_at1": _mean(p, "pointing_at1") if any("pointing_at1" in r for r in p) else _mean(p, "pointing_eagle"),
            "energy_in_box": _mean(p, "energy_in_box") if any("energy_in_box" in r for r in p) else _mean(p, "eagle_energy"),
            "iou_top10": _mean(p, "iou_top10") if any("iou_top10" in r for r in p) else None,
            "iou_top20": _mean(p, "iou_top20") if any("iou_top20" in r for r in p) else None,
            "area_eagle": _mean(p, "area_eagle"),
            "sufficiency": _mean(p, "sufficiency"),
            "necessity": _mean(p, "necessity"),
            "deletion_logp_drop": _mean(p, "deletion_logp_drop") if any("deletion_logp_drop" in r for r in p) else None,
            "insertion_logp_recovery": _mean(p, "insertion_logp_recovery") if any("insertion_logp_recovery" in r for r in p) else None,
            "visual_reliance": _mean(p, "visual_reliance"),
            "visual_fraction": _mean(p, "visual_fraction"),
            "visual_log_lift": _mean(p, "visual_log_lift"),
            "iou_lh": _mean(p, "iou_lh") if any("iou_lh" in r for r in p) else None,
            "iou_lh_boxed": _mean(p, "iou_lh_boxed") if any("iou_lh_boxed" in r for r in p) else None,
            "salr1_mass_gt": _mean(p, "salr1_mass_gt") if any("salr1_mass_gt" in r for r in p) else None,
            "salr1_mass_enrich": _mean(p, "salr1_mass_enrich") if any("salr1_mass_enrich" in r for r in p) else None,
            "salr1_iou_top20": _mean(p, "salr1_iou_top20") if any("salr1_iou_top20" in r for r in p) else None,
            "salr1_holistic_rate": _mean(p, "salr1_holistic") if any("salr1_holistic" in r for r in p) else None,
            "salr1_valid_rate": _mean(p, "salr1_valid") if any("salr1_valid" in r for r in p) else None,
        }
    return out


# ----------------------------------------------- table 2: looking vs using
def table2_looking_vs_using(recs, *, min_n=10) -> dict:
    p = _plain(recs)
    if len(p) < 3:
        return {}
    iou = _vals(p, "iou_eagle", drop_nan=False)
    correct = np.array([float(r["correct"]) for r in p])
    vfrac = np.array([r.get("visual_fraction", np.nan) for r in p], dtype=np.float64)
    vrel = np.array([r.get("visual_reliance", np.nan) for r in p], dtype=np.float64)
    vlog = np.array([r.get("visual_log_lift", np.nan) for r in p], dtype=np.float64)
    # Saliency-R1 cross-check: does the salr1 localization metric track correctness?
    salm = np.array([r.get("salr1_mass_gt", np.nan) for r in p], dtype=np.float64)
    saliou = np.array([r.get("salr1_iou_top20", np.nan) for r in p], dtype=np.float64)
    m = ~np.isnan(vfrac)
    ml = ~np.isnan(vlog)
    lc = safe_corr(iou, correct)
    uc = safe_corr(vfrac[m], correct[m])
    # log-lift is the PRIMARY using axis (raw-prob mean is diluted by easy tokens).
    ulc = safe_corr(vlog[ml], correct[ml])
    res = {
        "n": len(p),
        "accuracy": float(correct.mean()),
        "mean_iou_eagle": float(np.nanmean(iou)) if iou.size else float("nan"),
        "corr_correct_iou_eagle": lc,
        "corr_correct_visual_fraction": uc,
        "corr_correct_visual_log_lift": ulc,
        "corr_correct_visual_reliance": safe_corr(vrel[~np.isnan(vrel)], correct[~np.isnan(vrel)]),
        # Saliency-R1 cross-check (secondary baseline): a localization metric.
        "corr_correct_salr1_mass": safe_corr(salm[~np.isnan(salm)], correct[~np.isnan(salm)]),
        "corr_correct_salr1_iou": safe_corr(saliou[~np.isnan(saliou)], correct[~np.isnan(saliou)]),
        "mean_iou_eagle_right": float(iou[correct >= 0.5].mean()) if (correct >= 0.5).any() else float("nan"),
        "mean_iou_eagle_wrong": float(iou[correct < 0.5].mean()) if (correct < 0.5).any() else float("nan"),
        "mean_vlog_right": float(np.nanmean(vlog[correct >= 0.5])) if (correct >= 0.5).any() else float("nan"),
        "mean_vlog_wrong": float(np.nanmean(vlog[correct < 0.5])) if (correct < 0.5).any() else float("nan"),
    }
    lcv = lc if not np.isnan(lc) else 0.0
    # use the stronger of the two using signals (log-lift primary, fraction backup).
    ucv = max(ulc if not np.isnan(ulc) else -9, uc if not np.isnan(uc) else -9)
    ucv = ucv if ucv > -9 else 0.0
    if lcv >= 0.15:
        res["verdict"] = (f"looking matters: corr(correct, IoU_EAGLE)={lcv:+.2f} → the causally-needed region "
                          "tracks correctness; region/grounding supervision has headroom")
    elif ucv >= 0.10 and lcv <= 0.05:
        res["verdict"] = (f"using bottleneck: IoU_EAGLE corr≈0 ({lcv:+.2f}) but visual-reliance corr={ucv:+.2f} "
                          "(log-lift) → image-reliance, not localization, predicts correctness (output-level)")
    else:
        res["verdict"] = (f"mixed (corr(correct,IoU_EAGLE)={lcv:+.2f}, corr(correct,visual_log_lift)="
                          f"{ulc if not np.isnan(ulc) else float('nan'):+.2f})")
    return res


def table2_by_subset(recs, *, min_n=10) -> dict:
    p = _plain(recs)
    by_sub = defaultdict(list)
    for r in p:
        by_sub[r.get("subset", "?")].append(r)
    out = {}
    for sub, rs in sorted(by_sub.items()):
        a = table2_looking_vs_using(rs)
        if not a:
            continue
        out[sub] = {
            "n": a["n"], "accuracy": a["accuracy"],
            "corr_correct_iou_eagle": a["corr_correct_iou_eagle"],
            "corr_correct_visual_log_lift": a["corr_correct_visual_log_lift"],
            "mean_iou_eagle": a["mean_iou_eagle"],
            "low_n": a["n"] < min_n,
        }
    return out


# ----------------------------------------------- table 3: hint mechanism
def table3_hint_mechanism(recs) -> dict:
    p = {(r["subset"], r["sample_id"]): r for r in recs if r.get("condition") == "plain"}
    h = {(r["subset"], r["sample_id"]): r for r in recs if r.get("condition") == "hint"}
    common = sorted(set(p) & set(h))
    if not common:
        return {}

    def paired(key):
        a = np.array([p[i].get(key, np.nan) for i in common], dtype=np.float64)
        b = np.array([h[i].get(key, np.nan) for i in common], dtype=np.float64)
        mm = ~(np.isnan(a) | np.isnan(b))
        return a[mm], b[mm]

    acc_p = np.array([float(p[i]["correct"]) for i in common])
    acc_h = np.array([float(h[i]["correct"]) for i in common])
    iou_p, iou_h = paired("iou_eagle")
    vr_p, vr_h = paired("visual_reliance")
    logp_p, logp_h = paired("org_logp")
    vlog_p, vlog_h = paired("visual_log_lift")
    nec_p, nec_h = paired("necessity")
    # In score_plain_y the hint condition re-scores the SAME plain rollout, so its
    # completion (hence correctness) is identical → Δaccuracy is structurally 0 and
    # must NOT be read as "no help". We branch the verdict on the hint mode.
    hint_mode = next((r.get("hint_mode", "generate") for r in recs if r.get("condition") == "hint"), "generate")
    res = {
        "n_paired": len(common),
        "hint_mode": hint_mode,
        "delta_accuracy": float(acc_h.mean() - acc_p.mean()),
        "acc_plain": float(acc_p.mean()), "acc_hint": float(acc_h.mean()),
        "delta_iou_eagle": float((iou_h - iou_p).mean()) if iou_p.size else float("nan"),
        "delta_visual_reliance": float((vr_h - vr_p).mean()) if vr_p.size else float("nan"),
        "delta_org_logp": float((logp_h - logp_p).mean()) if logp_p.size else float("nan"),
        "delta_visual_log_lift": float((vlog_h - vlog_p).mean()) if vlog_p.size else float("nan"),
        "delta_necessity": float((nec_h - nec_p).mean()) if nec_p.size else float("nan"),
    }
    moves = (not np.isnan(res["delta_iou_eagle"])) and res["delta_iou_eagle"] >= 0.03
    if hint_mode == "score_plain_y":
        # Same rollout → judge by whether the hint raises the model's SUPPORT for it.
        d_logp = res["delta_org_logp"]
        lifts = (not np.isnan(d_logp)) and d_logp > 0
        if lifts and moves:
            res["verdict"] = ("score_plain_y: hint raises teacher support (Δorg_logp>0) AND moves the causal "
                              "region toward GT → the silent hint redirects attention to the evidence")
        elif lifts:
            res["verdict"] = ("score_plain_y: hint raises teacher support for the SAME rollout (Δorg_logp>0) with "
                              "~unchanged region → output-level reweighting (text-routing on the hint coords); "
                              "this is the distillable hidden-hint signal")
        else:
            res["verdict"] = "score_plain_y: hint does not raise support for the plain rollout (Δorg_logp≤0)"
        res["note"] = "Δaccuracy is 0 by construction here (same completion) — ignore it; read Δorg_logp / Δvisual_log_lift."
    else:
        helps = res["delta_accuracy"] >= 0.01
        if moves and helps:
            res["verdict"] = "attentional+causal: hint moves the needed region toward GT and helps → region distillation motivated"
        elif helps and not moves:
            res["verdict"] = "output-level: hint helps with ~unchanged causal region (likely text-routing on the hint coords)"
        elif moves and not helps:
            res["verdict"] = "region moved but accuracy flat → looking wasn't the bottleneck"
        else:
            res["verdict"] = "no effect"
    return res


# ----------------------------------------------- table 4: OPD/hint training
def table4_training(by_model, base_keys, opd_keys, hint_keys) -> dict:
    def pick(keys, *, avoid=(), exclude=()):
        for k in keys:
            for m in by_model:
                ml = m.lower()
                if m in exclude or any(a in ml for a in avoid):
                    continue
                if k.lower() in ml:
                    return m
        return None

    # Resolve hint-OPD first; the vanilla-OPD name often contains the OPD substring
    # too ("opd_qwen…" ⊂ "hint_opd_qwen…"), so exclude the hint model + avoid "hint".
    hint_model = pick(hint_keys)
    # base = the raw student: its name ("qwen3vl-2b") is a substring of the -opd/-hint
    # variants, so avoid both so we don't grab a trained checkpoint as "base".
    base_model = pick(base_keys, avoid=("opd", "hint"))
    opd_model = pick(opd_keys, avoid=("hint",), exclude=(hint_model,) if hint_model else ())
    roles = {"base": base_model, "opd": opd_model, "hint_opd": hint_model}
    out = {}
    for role, model in roles.items():
        if not model:
            continue
        p = _plain(by_model[model])
        if not p:
            continue
        out[role] = {
            "model": model, "n": len(p),
            "accuracy": float(np.mean([r["correct"] for r in p])),
            "visual_reliance": _mean(p, "visual_reliance"),
            "visual_fraction": _mean(p, "visual_fraction"),
            "visual_log_lift": _mean(p, "visual_log_lift"),
            "iou_eagle": _mean(p, "iou_eagle"),
            "necessity": _mean(p, "necessity"),
            "sufficiency": _mean(p, "sufficiency"),
        }
    return out


# ----------------------------------------------------------------- report
def write_report(out_dir, analysis) -> None:
    L = ["# EAGLE-G0 — faithful causal attribution report", ""]
    L.append(f"_correctness source: **{analysis.get('correctness_source','rule')}** · "
             f"spatial metric source: **{analysis.get('spatial_metric_source','record')}** · "
             f"{analysis.get('n_records',0)} records across {analysis.get('n_models',0)} models_")
    L.append("")

    t1 = analysis.get("table1_region_accuracy", {})
    if t1:
        L.append("## Table 1 — region accuracy (plain)")
        L.append("")
        L.append("| model | n | acc | IoU_EAGLE | Point@1 | Energy | IoU@10 | IoU@20 | DelDrop | InsRec | vis_frac |")
        L.append("|-------|---|-----|-----------|---------|--------|--------|--------|---------|--------|----------|")
        for m, e in t1.items():
            L.append(f"| {m} | {e['n']} | {_fmt(e['accuracy'])} | {_fmt(e['iou_eagle'])} | "
                     f"{_fmt(e.get('pointing_at1'))} | {_fmt(e.get('energy_in_box'))} | "
                     f"{_fmt(e.get('iou_top10'))} | {_fmt(e.get('iou_top20'))} | "
                     f"{_fmt(e.get('deletion_logp_drop'))} | {_fmt(e.get('insertion_logp_recovery'))} | "
                     f"{_fmt(e['visual_fraction'])} |")
        L.append("")
        L.append("_Point@1 = max heat patch in GT. Energy = heatmap mass inside GT. "
                 "IoU@10/20 threshold the top 10%/20% area of the final aggregate map. "
                 "DelDrop/InsRec are target-span logp drop/recovery after deleting/keeping top-20% attributed area._")
        if analysis.get("spatial_metric_source") == "sentence_span_artifact":
            L.append("")
            L.append("_DelDrop/InsRec are left blank because offline sentence-map reconstruction has no model "
                     "forwards; Acc, IoU, Point@1, Energy, visual reliance, and span AUC metrics remain valid._")
        # Saliency-R1 health: holistic-rate (went through the thinking bottleneck) and
        # valid-rate (non-empty positive map). Low → read salr1_abs_*, not salr1_pos_*.
        health = [f"{m}: holistic={_fmt(e.get('salr1_holistic_rate'))} valid={_fmt(e.get('salr1_valid_rate'))} "
                  f"mass_enrich={_fmt(e.get('salr1_mass_enrich'))}"
                  for m, e in t1.items() if e.get("salr1_holistic_rate") is not None]
        if health:
            L.append("")
            L.append("_SalR1 health (low holistic/valid ⇒ map is direct-answer/unstable — trust abs over pos): "
                     + "; ".join(health) + "_")
        L.append("")

    ga = analysis.get("group_averages", {})
    if ga:
        L.append("## Group averages (plain) — first-5 (primary local-evidence) vs all-10")
        L.append("")
        L.append("_macro = mean over per-subset means (each task type counts equally); "
                 "pooled = size-weighted mean over records (dominated by big subsets gqa/flickr30k). "
                 "**Prefer macro** — subset sizes differ ~25× (gqa 1765 vs vsr 70)._")
        L.append("")
        for grp, label in (("primary5", "first-5 (gqa/openimages/v7w/textvqa/vsr)"), ("all", "all-10")):
            L.append(f"### {label} average")
            L.append("")
            L.append("| model | #sub | n | acc | IoU_EAGLE | Point@1 | Energy | IoU@20 | DelDrop | InsRec | vis_log_lift |")
            L.append("|-------|------|---|-----|-----------|---------|--------|--------|---------|--------|--------------|")
            for m, e in ga.items():
                g = e.get(grp)
                if not g or not g.get("n"):
                    continue
                def mm(metric):  # "macro (pooled)"
                    return f"{_fmt(g.get(metric + '_macro'))} ({_fmt(g.get(metric + '_pooled'))})"
                L.append(f"| {m} | {g['n_subsets']} | {g['n']} | {mm('accuracy')} | {mm('iou_eagle')} | "
                         f"{mm('pointing_at1')} | {mm('energy_in_box')} | {mm('iou_top20')} | "
                         f"{mm('deletion_logp_drop')} | {mm('insertion_logp_recovery')} | {mm('visual_log_lift')} |")
            L.append("")
        L.append("_cells are `macro (pooled)`._")
        L.append("")

    t2 = analysis.get("table2_looking_vs_using", {})
    if t2:
        L.append("## Table 2 — looking vs using (per model, plain)")
        L.append("")
        L.append("| model | n | acc | corr(c,IoU_EAGLE) | corr(c,vlog) | corr(c,vfrac) | IoU r/w | verdict |")
        L.append("|-------|---|-----|-------------------|--------------|---------------|---------|---------|")
        for m, a in t2.items():
            L.append(f"| {m} | {a['n']} | {_fmt(a['accuracy'])} | {_fmt(a['corr_correct_iou_eagle'])} | "
                     f"{_fmt(a['corr_correct_visual_log_lift'])} | {_fmt(a['corr_correct_visual_fraction'])} | "
                     f"{_fmt(a['mean_iou_eagle_right'])}/{_fmt(a['mean_iou_eagle_wrong'])} | {a['verdict'].split(':')[0]} |")
        L.append("")
        for m, a in t2.items():
            L.append(f"- **{m}**: {a['verdict']}")
        L.append("")
        # Saliency-R1 cross-check: does the secondary map agree on the looking axis?
        if any(a.get("corr_correct_salr1_mass") == a.get("corr_correct_salr1_mass") for a in t2.values()):
            L.append("**Saliency-R1 cross-check** (does the secondary map agree the looking axis is flat?):")
            L.append("")
            L.append("| model | corr(c,IoU_EAGLE) | corr(c,SalR1_mass) | corr(c,SalR1_IoU) | corr(c,vlog) |")
            L.append("|-------|-------------------|--------------------|-------------------|--------------|")
            for m, a in t2.items():
                L.append(f"| {m} | {_fmt(a['corr_correct_iou_eagle'])} | {_fmt(a.get('corr_correct_salr1_mass'))} | "
                         f"{_fmt(a.get('corr_correct_salr1_iou'))} | {_fmt(a['corr_correct_visual_log_lift'])} |")
            L.append("")
            L.append("_If EAGLE and Saliency-R1 both show localization corr≈0 while the reliance/log-lift corr is "
                     "positive, the 'using-bottleneck' verdict is robust to the attribution method (not an LH artifact)._")
            L.append("")

    t2s = analysis.get("table2_by_subset", {})
    if t2s:
        L.append("## Table 2b — looking vs using by task type (plain)")
        for m, subs in t2s.items():
            L.append(f"\n**{m}**\n")
            L.append("| subset | n | acc | corr(c,IoU_EAGLE) | corr(c,vlog) |")
            L.append("|--------|---|-----|-------------------|--------------|")
            for sub, e in subs.items():
                flag = " ⚠" if e.get("low_n") else ""
                L.append(f"| {sub}{flag} | {e['n']} | {_fmt(e['accuracy'])} | {_fmt(e['corr_correct_iou_eagle'])} | "
                         f"{_fmt(e['corr_correct_visual_log_lift'])} |")
        L.append("")

    t3 = analysis.get("table3_hint_mechanism", {})
    if t3:
        L.append("## Table 3 — hint mechanism (plain vs hint, paired)")
        for m, a in t3.items():
            if a.get("hint_mode") == "score_plain_y":
                # same rollout → Δacc is 0 by construction; report support deltas.
                L.append(f"- **{m}** [score_plain_y]: n={a['n_paired']}, "
                         f"Δorg_logp={_fmt(a.get('delta_org_logp'))}, "
                         f"Δvisual_log_lift={_fmt(a.get('delta_visual_log_lift'))}, "
                         f"ΔIoU_EAGLE={_fmt(a['delta_iou_eagle'])}, Δnecessity={_fmt(a.get('delta_necessity'))} "
                         f"→ {a['verdict']}")
            else:
                L.append(f"- **{m}** [generate]: n={a['n_paired']}, Δacc={_fmt(a['delta_accuracy'])} "
                         f"(plain {_fmt(a['acc_plain'])}→hint {_fmt(a['acc_hint'])}), "
                         f"ΔIoU_EAGLE={_fmt(a['delta_iou_eagle'])}, Δvisual_reliance={_fmt(a['delta_visual_reliance'])} "
                         f"→ {a['verdict']}")
        L.append("")

    t4 = analysis.get("table4_training", {})
    if t4:
        L.append("## Table 4 — OPD / hint training raises image-reliance? (plain)")
        L.append("")
        L.append("| role | model | n | acc | visual_log_lift | visual_reliance | visual_fraction | IoU_EAGLE | necessity |")
        L.append("|------|-------|---|-----|-----------------|-----------------|-----------------|-----------|-----------|")
        for role in ("base", "opd", "hint_opd"):
            e = t4.get(role)
            if not e:
                continue
            L.append(f"| {role} | {e['model']} | {e['n']} | {_fmt(e['accuracy'])} | {_fmt(e['visual_log_lift'])} | "
                     f"{_fmt(e['visual_reliance'])} | {_fmt(e['visual_fraction'])} | {_fmt(e['iou_eagle'])} | "
                     f"{_fmt(e['necessity'])} |")
        L.append("")
        L.append("_Success story = training ↑ accuracy AND ↑ visual_log_lift / visual_reliance (answer depends on "
                 "the image more), even if IoU_EAGLE barely moves._")
        L.append("")

    with open(os.path.join(out_dir, "eagle_report.md"), "w") as f:
        f.write("\n".join(L) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze EAGLE-G0 run(s).")
    ap.add_argument("--run-dirs", nargs="+", required=True, help="One or more model run dirs.")
    ap.add_argument("--output-dir", default=None, help="Where to write the report (default: first run dir).")
    ap.add_argument("--use-judge", action="store_true", help="Overlay judgments.jsonl per dir (LLM correctness).")
    ap.add_argument(
        "--use-sentence-span-metrics",
        action="store_true",
        help="Overlay sentence_span_metrics.jsonl geometry before reporting.",
    )
    ap.add_argument("--base-keys", default="qwen3vl-2b,2b-instruct,base",
                    help="Substrings to identify the RAW base student (table 4); -opd/-hint variants are excluded.")
    ap.add_argument("--opd-keys", default="opd,vanilla", help="Substrings for the vanilla-OPD student.")
    ap.add_argument("--hint-keys", default="hint", help="Substrings for the hint-OPD student.")
    args = ap.parse_args()

    records = _load_all(args.run_dirs, args.use_judge, args.use_sentence_span_metrics)
    if not records:
        raise SystemExit(f"[eagle.analyze] no records in {args.run_dirs}")
    by_model = _by_model(records)
    out_dir = args.output_dir or args.run_dirs[0]
    os.makedirs(out_dir, exist_ok=True)

    analysis = {
        "n_records": len(records),
        "n_models": len(by_model),
        "models": sorted(by_model),
        "correctness_source": "llm_judge" if args.use_judge else "rule",
        "spatial_metric_source": "sentence_span_artifact" if args.use_sentence_span_metrics else "record",
        "table1_region_accuracy": table1_region_accuracy(by_model),
        "group_averages": group_averages(by_model),
        "table2_looking_vs_using": {m: table2_looking_vs_using(recs) for m, recs in by_model.items()},
        "table2_by_subset": {m: table2_by_subset(recs) for m, recs in by_model.items()},
        "table3_hint_mechanism": {m: table3_hint_mechanism(recs) for m, recs in by_model.items()
                                  if any(r.get("condition") == "hint" for r in recs)},
        "table4_training": table4_training(
            by_model,
            [k for k in args.base_keys.split(",") if k],
            [k for k in args.opd_keys.split(",") if k],
            [k for k in args.hint_keys.split(",") if k],
        ),
    }
    # Drop empty per-model entries for readability.
    analysis["table2_looking_vs_using"] = {m: a for m, a in analysis["table2_looking_vs_using"].items() if a}
    analysis["table2_by_subset"] = {m: a for m, a in analysis["table2_by_subset"].items() if a}
    analysis["table3_hint_mechanism"] = {m: a for m, a in analysis["table3_hint_mechanism"].items() if a}

    with open(os.path.join(out_dir, "eagle_analysis.json"), "w") as f:
        json.dump(analysis, f, indent=2)
    write_report(out_dir, analysis)
    print(json.dumps(analysis, indent=2))
    print(f"\n[eagle.analyze] wrote eagle_analysis.json + eagle_report.md → {out_dir}")


if __name__ == "__main__":
    main()
