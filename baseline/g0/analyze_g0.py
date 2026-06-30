"""G0 analysis — turn ``records.jsonl`` into the four diagnostic analyses.

Reads a run directory produced by :mod:`baseline.g0.run_g0` (``records.jsonl`` +
``head_stats_{student,teacher}.{json,npz}``) and writes ``analysis.json``,
``report.md`` and figures under ``figs/``. The four analyses follow the G0 manual:

  1. **Head usability** (8B & 2B) — can either model's attention localize at all?
     Best per-head mean IoU, the selected localization heads and their layer
     distribution, and the assembled ``IoU_LH`` on the natural condition. Gates the
     LH-box / label-free plans.
  2. **Student looking-vs-using** (the key one) — on C3, the 2×2 of ``IoU_LH``
     (high/low) × correctness, plus ``vt_ratio`` for right vs wrong. A heavy
     "looked-right-but-wrong + low vt_ratio" mass is the *using-failure* signature
     (attention/map interventions can't help; stay at the OPD output level). The
     opposite — low IoU that correlates with correctness — is *looking failure*.
  3. **Hint mechanism** (C1 vs C2, paired) — does the silent hint move attention
     (ΔIoU_LH ≫ 0, "attentional") or just the answer (Δacc > 0 with ΔIoU_LH ≈ 0,
     "non-attentional / output-level")?
  4. **Teacher-vs-student gap** (C1 vs C3) — is the gap mainly localization
     (ΔIoU_LH) or attribution pattern (Δvt_ratio)? Tells us what OPD must transfer.

Run: ``uv run python -m baseline.g0.analyze_g0 --run-dir eval_outputs/g0/run1``
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Optional

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np


# --------------------------------------------------------------------- loading
def load_records(run_dir: str) -> list[dict]:
    """Read all record files (single ``records.jsonl`` or sharded ``records.shard*.jsonl``).

    Tolerant of a half-written trailing line so it can be run MID-RUN for a live
    trend preview while shards are still appending.
    """
    paths = sorted(glob.glob(os.path.join(run_dir, "records*.jsonl")))
    records = []
    bad = 0
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    bad += 1  # partially-flushed last line during a live run
    if bad:
        print(f"[g0.analyze] skipped {bad} unparseable line(s) (mid-run preview?).")
    return records


def by_condition(records: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        out[r["condition"]].append(r)
    return out


def load_head_stats(run_dir: str, tag: str) -> Optional[dict]:
    path = os.path.join(run_dir, f"head_stats_{tag}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def apply_judge(run_dir: str, records: list[dict]) -> int:
    """Overlay ``judgments.jsonl`` (from :mod:`baseline.g0.judge_g0`) onto records.

    Replaces each matched record's rule ``correct`` with the LLM ``correct_judge``
    (the trustworthy correctness for the looking-vs-using correlations). Returns the
    number of records overridden; 0 (with a warning) if there is no judgments file.
    """
    path = os.path.join(run_dir, "judgments.jsonl")
    if not os.path.exists(path):
        print(f"[g0.analyze] --use-judge but {path} not found — run baseline.g0.judge_g0 first. "
              "Falling back to the rule grader.")
        return 0
    jmap: dict[tuple, bool] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            j = json.loads(line)
            jmap[(j.get("model", ""), j.get("condition", ""), j.get("subset", ""), str(j.get("sample_id", "")))] = bool(
                j.get("correct_judge", False))
    n = 0
    for r in records:
        k = (r.get("model", ""), r.get("condition", ""), r.get("subset", ""), str(r.get("sample_id", "")))
        if k in jmap:
            r["correct"] = jmap[k]
            n += 1
    print(f"[g0.analyze] applied LLM judge to {n}/{len(records)} records (correct_judge overrides rule).")
    return n


# ------------------------------------------------------------------- stats utils
def _vals(records, key, *, drop_nan=True) -> np.ndarray:
    arr = np.array([r[key] for r in records], dtype=np.float64)
    if drop_nan:
        arr = arr[~np.isnan(arr)]
    return arr


def _mean(records, key) -> float:
    arr = _vals(records, key)
    return float(arr.mean()) if arr.size else float("nan")


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation with constant-array / length guards."""
    if x.size < 3 or y.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


# ----------------------------------------------------- analysis 1: head usability
def analysis_1_head_usability(run_dir, conds) -> dict:
    out = {}
    natural = {"teacher": "c1", "student": "c3"}
    for tag, cond in natural.items():
        stats = load_head_stats(run_dir, tag)
        recs = conds.get(cond, [])
        if stats is None and not recs:
            continue
        iou_lh = _vals(recs, "iou_lh") if recs else np.array([])
        best_single = _vals(recs, "best_single_iou") if recs else np.array([])
        # pointing (argmax-in-box) / energy (mass-in-box) are more forgiving than
        # IoU for small GT boxes on a coarse grid: high pointing + low IoU = "points
        # roughly right but diffuse"; ~chance pointing = genuinely mislocalized.
        pointing = _vals(recs, "lh_pointing") if recs else np.array([])
        energy = _vals(recs, "lh_energy") if recs else np.array([])
        entry = {
            "condition": cond,
            "best_head_mean_iou": stats["best_mean_iou"] if stats else None,
            "selected_heads": stats["selected_heads"] if stats else None,
            "selected_head_layers": sorted({h[0] for h in stats["selected_heads"]}) if stats else None,
            "assembled_iou_lh_mean": float(iou_lh.mean()) if iou_lh.size else None,
            "best_single_iou_mean": float(best_single.mean()) if best_single.size else None,
            "assembled_pointing_mean": float(pointing.mean()) if pointing.size else None,
            "assembled_energy_mean": float(energy.mean()) if energy.size else None,
            "n": len(recs),
        }
        # Heuristic verdict (numbers are what matter; this is a quick label).
        score = entry["assembled_iou_lh_mean"] or entry["best_head_mean_iou"] or 0.0
        entry["verdict"] = "clean" if score >= 0.30 else ("weak" if score >= 0.15 else "noisy")
        out[tag] = entry
    return out


# ------------------------------------------ analysis 2: student looking-vs-using
def _two_by_two(looked_right: np.ndarray, right: np.ndarray) -> dict:
    cell = lambda lr, c: int(np.sum((looked_right == lr) & (right == c)))
    return {
        "high_iou_correct": cell(True, True),
        "high_iou_wrong": cell(True, False),   # ⚠ looked right but wrong (using-failure flavor)
        "low_iou_correct": cell(False, True),  # guessed right
        "low_iou_wrong": cell(False, False),
    }


def analysis_2_looking_vs_using(records_c3, *, iou_threshold=None, abs_iou_threshold=0.30,
                                iou_key="iou_lh") -> dict:
    """Looking-vs-using on the student.

    The verdict is **correlation-driven** (threshold-independent): does IoU_LH
    predict correctness (looking-failure) or does vt_ratio predict it while IoU_LH
    does not (using-failure)? We report THREE 2x2s for transparency — a relative
    (median) split, an ABSOLUTE split (IoU >= abs_iou_threshold = genuinely
    "looked right"), and a pointing split (argmax patch in GT box) — because a
    median split mechanically calls half the samples "high IoU" even when all IoU
    are low, which would inflate the using-failure cell. ``iou_key`` lets the
    caller run this on the answer-span LH (``iou_lh_answer``) too.
    """
    if not records_c3:
        return {}
    iou = _vals(records_c3, iou_key, drop_nan=False)
    correct = np.array([float(r["correct"]) for r in records_c3])
    # Match the pointing + vt keys to the IoU variant so each looking-vs-using
    # flavor is internally consistent (boxed IoU ↔ boxed pointing ↔ boxed vt).
    _POINTING = {"iou_lh": "lh_pointing", "iou_lh_answer": "lh_answer_pointing", "iou_lh_boxed": "lh_boxed_pointing"}
    _VT = {"iou_lh": "vt_ratio", "iou_lh_answer": "vt_ratio_answer", "iou_lh_boxed": "vt_ratio_boxed"}
    pointing_key = _POINTING.get(iou_key, "lh_pointing")
    vt_key = _VT.get(iou_key, "vt_ratio")
    vt = np.array([r.get(vt_key, r.get("vt_ratio", float("nan"))) for r in records_c3], dtype=np.float64)
    pointing = np.array([float(r.get(pointing_key, 0.0)) for r in records_c3])
    right = correct >= 0.5
    n_wrong = int(np.sum(~right))

    med = float(np.median(iou))
    rel_thr = med if iou_threshold is None else iou_threshold
    tables = {
        "relative_median": {"threshold": rel_thr, **_two_by_two(iou >= rel_thr, right)},
        "absolute": {"threshold": abs_iou_threshold, **_two_by_two(iou >= abs_iou_threshold, right)},
        "pointing": {"threshold": 1.0, **_two_by_two(pointing >= 0.5, right)},
    }
    vt_right = vt[right & ~np.isnan(vt)]
    vt_wrong = vt[(~right) & ~np.isnan(vt)]
    look_corr = safe_corr(iou, correct)
    use_corr = safe_corr(vt[~np.isnan(vt)], correct[~np.isnan(vt)])

    res = {
        "n": len(records_c3),
        "iou_key": iou_key,
        "accuracy": float(right.mean()),
        "median_iou": med,
        "looked_right_rate_abs": float(np.mean(iou >= abs_iou_threshold)),
        "pointing_rate": float(np.mean(pointing >= 0.5)),
        "tables": tables,
        "high_iou_among_wrong_frac_abs": (tables["absolute"]["high_iou_wrong"] / n_wrong) if n_wrong else float("nan"),
        "mean_iou_lh_right": float(iou[right].mean()) if right.any() else float("nan"),
        "mean_iou_lh_wrong": float(iou[~right].mean()) if (~right).any() else float("nan"),
        "mean_vt_right": float(vt_right.mean()) if vt_right.size else float("nan"),
        "mean_vt_wrong": float(vt_wrong.mean()) if vt_wrong.size else float("nan"),
        "corr_correct_iou_lh": look_corr,
        "corr_correct_vt": use_corr,
        "mean_iou_gl": _mean(records_c3, "iou_gl"),
        "mean_vt_ratio": _mean(records_c3, "vt_ratio"),
    }

    # Correlation-driven verdict (does not depend on any IoU threshold).
    lc = look_corr if not np.isnan(look_corr) else 0.0
    uc = use_corr if not np.isnan(use_corr) else 0.0
    if lc >= 0.15:
        res["verdict"] = ("looking-failure (dominant): correctness rises with IoU_LH "
                          f"(corr={lc:+.2f}) → fixing where-to-look has headroom")
    elif uc >= 0.10 and lc <= 0.05:
        res["verdict"] = ("using-failure (dominant): IoU_LH does NOT predict correctness "
                          f"(corr={lc:+.2f}) but vt_ratio does (corr={uc:+.2f}) → output-level leverage")
    else:
        res["verdict"] = (f"mixed / inconclusive (corr(correct,IoU_LH)={lc:+.2f}, "
                          f"corr(correct,vt)={uc:+.2f}) — inspect tables + answer-span variant")
    return res


def looking_vs_using_by_subset(records_c3, *, iou_key="iou_lh_boxed", min_n=15) -> dict:
    """Per-task-type (Visual-CoT subset) looking-vs-using on the student.

    The G0 verdict can flip by task type (object/spatial vs OCR/doc), so we break
    the key correlations down per subset. Subsets with < ``min_n`` samples are
    reported but flagged (corr is noisy). Uses the boxed-span IoU by default.
    """
    if not records_c3:
        return {}
    by_sub: dict[str, list[dict]] = defaultdict(list)
    for r in records_c3:
        by_sub[r.get("subset", "?")].append(r)
    out = {}
    for sub, recs in sorted(by_sub.items()):
        a = analysis_2_looking_vs_using(recs, iou_key=iou_key)
        if not a:
            continue
        out[sub] = {
            "n": a["n"],
            "accuracy": a["accuracy"],
            "corr_correct_iou_lh": a["corr_correct_iou_lh"],
            "corr_correct_vt": a["corr_correct_vt"],
            "looked_right_rate_abs": a.get("looked_right_rate_abs"),
            "verdict_short": a["verdict"].split(":")[0],
            "low_n": a["n"] < min_n,
        }
    return out


# ------------------------------------------------- analysis 3: hint mechanism
def analysis_3_hint_mechanism(records_c1, records_c2) -> dict:
    if not records_c1 or not records_c2:
        return {}
    # Pair on (subset, sample_id): question_id repeats across subsets in saliency-r1-8k,
    # so keying on sample_id alone would mis-pair C1/C2 across subsets.
    by_id1 = {(r["subset"], r["sample_id"]): r for r in records_c1}
    by_id2 = {(r["subset"], r["sample_id"]): r for r in records_c2}
    common = sorted(set(by_id1) & set(by_id2))
    if not common:
        return {"n_paired": 0}

    def paired(key):
        a = np.array([by_id1[i][key] for i in common], dtype=np.float64)
        b = np.array([by_id2[i][key] for i in common], dtype=np.float64)
        m = ~(np.isnan(a) | np.isnan(b))
        return a[m], b[m]

    iou1, iou2 = paired("iou_lh")
    gl1, gl2 = paired("iou_gl")
    vt1, vt2 = paired("vt_ratio")
    acc1 = np.array([float(by_id1[i]["correct"]) for i in common])
    acc2 = np.array([float(by_id2[i]["correct"]) for i in common])

    d_iou = float((iou2 - iou1).mean()) if iou1.size else float("nan")
    d_gl = float((gl2 - gl1).mean()) if gl1.size else float("nan")
    d_vt = float((vt2 - vt1).mean()) if vt1.size else float("nan")
    d_acc = float(acc2.mean() - acc1.mean())

    res = {
        "n_paired": len(common),
        "delta_iou_lh": d_iou,
        "delta_iou_gl": d_gl,
        "delta_vt_ratio": d_vt,
        "delta_accuracy": d_acc,
        "acc_c1": float(acc1.mean()),
        "acc_c2": float(acc2.mean()),
        "mean_iou_lh_c1": float(iou1.mean()) if iou1.size else float("nan"),
        "mean_iou_lh_c2": float(iou2.mean()) if iou2.size else float("nan"),
    }
    # attentional if the hint moves attention toward GT AND meaningfully helps the
    # answer. d_acc must clear a noise floor (a +0.0004 delta is NOT "helps").
    moves_attn = (not np.isnan(d_iou)) and d_iou >= 0.05
    helps = d_acc >= 0.01
    if moves_attn and helps:
        res["verdict"] = "attentional: hint pulls attention toward GT and helps → 'attend like C2' is a real target"
    elif helps and not moves_attn:
        res["verdict"] = "non-attentional: answer improves with ~unchanged attention → hint acts at the output level"
    elif moves_attn and not helps:
        res["verdict"] = ("attention moved toward GT but accuracy flat → looking wasn't the bottleneck "
                          "(consistent with using-failure)")
    else:
        res["verdict"] = ("no effect: hint moved neither attention nor accuracy (Δacc≈0) → where-to-look "
                          "is not the lever on this dataset")
    # Δvt_ratio is confounded: the hint adds TEXT tokens to the prompt, which
    # mechanically lowers visual/(visual+textual). vt is only clean WITHIN a model
    # (correct vs wrong, analysis 2), not across C1 vs C2.
    res["note"] = ("Δvt_ratio is confounded by the hint adding prompt text tokens; do not read it as a "
                   "real attribution shift. Δaccuracy and ΔIoU_LH are the clean signals here.")
    return res


# ------------------------------------------- analysis 4: teacher-vs-student gap
def analysis_4_gap(records_c1, records_c3) -> dict:
    if not records_c1 or not records_c3:
        return {}
    res = {
        "teacher_mean_iou_lh": _mean(records_c1, "iou_lh"),
        "student_mean_iou_lh": _mean(records_c3, "iou_lh"),
        "teacher_mean_vt": _mean(records_c1, "vt_ratio"),
        "student_mean_vt": _mean(records_c3, "vt_ratio"),
        "teacher_acc": float(np.mean([r["correct"] for r in records_c1])),
        "student_acc": float(np.mean([r["correct"] for r in records_c3])),
    }
    res["gap_iou_lh"] = res["teacher_mean_iou_lh"] - res["student_mean_iou_lh"]
    res["gap_vt"] = res["teacher_mean_vt"] - res["student_mean_vt"]
    # which gap dominates (normalized by the teacher level)?
    loc = abs(res["gap_iou_lh"]) / max(1e-6, abs(res["teacher_mean_iou_lh"]))
    attr = abs(res["gap_vt"]) / max(1e-6, abs(res["teacher_mean_vt"]))
    res["dominant_gap"] = "localization (IoU_LH)" if loc > attr else "attribution (vt_ratio)"
    # Cross-model vt is confounded by CoT length (a more verbose teacher puts more
    # mass on its own generated tokens → lower visual/(visual+textual)). Don't read
    # student>teacher vt as the student being "more grounded".
    res["note"] = ("Cross-model vt_ratio is confounded by CoT length / verbosity; the clean vt signal is "
                   "WITHIN-model (correct vs wrong) in analysis 2, not this teacher-vs-student gap.")
    return res


# ----------------------------------------------------------------------- figures
def make_figures(run_dir, conds, a1, a2) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[g0.analyze] matplotlib unavailable; skipping figures.")
        return
    figs = os.path.join(run_dir, "figs")
    os.makedirs(figs, exist_ok=True)

    # 1) per-model mean_iou head heatmaps.
    for tag in ("teacher", "student"):
        path = os.path.join(run_dir, f"head_stats_{tag}.npz")
        if not os.path.exists(path):
            continue
        mean_iou = np.load(path)["mean_iou"]
        fig, ax = plt.subplots(figsize=(max(4, mean_iou.shape[1] * 0.3), max(4, mean_iou.shape[0] * 0.25)))
        im = ax.imshow(mean_iou, aspect="auto", cmap="viridis")
        ax.set_xlabel("head"); ax.set_ylabel("layer"); ax.set_title(f"{tag}: per-head mean IoU vs GT")
        fig.colorbar(im, ax=ax)
        fig.tight_layout(); fig.savefig(os.path.join(figs, f"head_iou_{tag}.png"), dpi=110); plt.close(fig)

    # 2) C3 scatter IoU_LH vs vt_ratio colored by correctness (the key plot).
    c3 = conds.get("c3", [])
    if c3:
        iou = _vals(c3, "iou_lh", drop_nan=False)
        vt = np.array([r["vt_ratio"] for r in c3], dtype=np.float64)
        correct = np.array([bool(r["correct"]) for r in c3])
        m = ~np.isnan(vt)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(iou[m & correct], vt[m & correct], c="green", label="correct", alpha=0.6)
        ax.scatter(iou[m & ~correct], vt[m & ~correct], c="red", label="wrong", alpha=0.6)
        if a2 and a2.get("tables"):
            ax.axvline(a2["tables"]["absolute"]["threshold"], color="gray", ls="--", lw=1)
        ax.set_xlabel("IoU_LH (looking)"); ax.set_ylabel("vt_ratio (using)")
        ax.set_title("C3 student: looking vs using"); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(figs, "c3_looking_vs_using.png"), dpi=110); plt.close(fig)

    # 3) condition bars: accuracy / IoU_LH / IoU_GL / vt_ratio.
    order = [c for c in ("c1", "c2", "c3") if c in conds]
    if order:
        metrics_ = {
            "accuracy": [float(np.mean([r["correct"] for r in conds[c]])) for c in order],
            "IoU_LH": [_mean(conds[c], "iou_lh") for c in order],
            "IoU_GL": [_mean(conds[c], "iou_gl") for c in order],
            "vt_ratio": [_mean(conds[c], "vt_ratio") for c in order],
        }
        fig, axes = plt.subplots(1, len(metrics_), figsize=(4 * len(metrics_), 4))
        for ax, (name, vals) in zip(axes, metrics_.items()):
            ax.bar(order, vals, color=["#4c72b0", "#dd8452", "#55a868"][: len(order)])
            ax.set_title(name)
        fig.tight_layout(); fig.savefig(os.path.join(figs, "condition_bars.png"), dpi=110); plt.close(fig)


# ------------------------------------------------------------------------ report
def _render_lvu_compact(L: list, a: dict, label: str) -> None:
    """Compact looking-vs-using block (headline corr + absolute 2x2 + verdict)."""
    if not a:
        return
    t = a["tables"]["absolute"]
    L.append(f"### looking-vs-using [{label}]  (n={a['n']}, acc={_fmt(a['accuracy'])})")
    L.append(f"- corr(correct, IoU)={_fmt(a['corr_correct_iou_lh'])}, "
             f"corr(correct, vt)={_fmt(a['corr_correct_vt'])}, "
             f"median IoU={_fmt(a.get('median_iou'))}, looked-right@{_fmt(t['threshold'])}={_fmt(a.get('looked_right_rate_abs'))}")
    L.append(f"- looked-right→correct {t['high_iou_correct']}/{t['high_iou_correct'] + t['high_iou_wrong']}, "
             f"looked-wrong→correct {t['low_iou_correct']}/{t['low_iou_correct'] + t['low_iou_wrong']}")
    L.append(f"- **verdict: {a['verdict']}**")
    L.append("")


def write_report(run_dir, analysis) -> None:
    a1, a2, a3, a4 = (analysis.get(k, {}) for k in ("head_usability", "looking_vs_using", "hint_mechanism", "gap"))
    L = ["# G0 grounding diagnostic — report", ""]
    L.append(f"_correctness source: **{analysis.get('correctness_source', 'rule')}** · "
             f"headline span: **{analysis.get('headline_span', 'first')}**_")
    L.append("")

    L.append("## Analysis 1 — head usability (can attention localize at all?)")
    for tag in ("teacher", "student"):
        e = a1.get(tag)
        if not e:
            continue
        L.append(f"- **{tag}** ({e['condition']}): assembled IoU_LH={_fmt(e['assembled_iou_lh_mean'])}, "
                 f"best single-head IoU={_fmt(e['best_single_iou_mean'])}, best per-head mean IoU="
                 f"{_fmt(e['best_head_mean_iou'])}, pointing={_fmt(e.get('assembled_pointing_mean'))}, "
                 f"energy={_fmt(e.get('assembled_energy_mean'))}, heads={e['selected_heads']} "
                 f"(layers {e['selected_head_layers']}) → **{e['verdict']}**")
    L.append("_pointing = argmax-patch-in-GT (chance ≈ GT-area fraction, small here); high pointing + low "
             "IoU = points right but diffuse; ~chance pointing = mislocalized._")
    L.append("")

    if a2:
        L.append(f"## Analysis 2 — student looking-vs-using (KEY) [{a2.get('iou_key', 'iou_lh')}]")
        L.append(f"n={a2['n']}, accuracy={_fmt(a2['accuracy'])}, median IoU_LH={_fmt(a2.get('median_iou'))}, "
                 f"looked-right rate (IoU≥{a2['tables']['absolute']['threshold']})={_fmt(a2.get('looked_right_rate_abs'))}, "
                 f"pointing rate={_fmt(a2.get('pointing_rate'))}")
        L.append("")
        L.append("**Headline (threshold-free): corr(correct, IoU_LH)="
                 f"{_fmt(a2['corr_correct_iou_lh'])}, corr(correct, vt_ratio)={_fmt(a2['corr_correct_vt'])}**")
        L.append(f"- mean IoU_LH: right={_fmt(a2['mean_iou_lh_right'])}, wrong={_fmt(a2['mean_iou_lh_wrong'])}")
        L.append(f"- mean vt_ratio: right={_fmt(a2['mean_vt_right'])}, wrong={_fmt(a2['mean_vt_wrong'])} "
                 f"(wrong<right ⇒ answer driven by text/prior)")
        L.append("")
        for name in ("absolute", "pointing", "relative_median"):
            t = a2["tables"][name]
            L.append(f"2x2 ({name}, looked-right @ {_fmt(t['threshold'])}):")
            L.append("")
            L.append("|             | correct | wrong |")
            L.append("|-------------|---------|-------|")
            L.append(f"| looked-right | {t['high_iou_correct']} | {t['high_iou_wrong']} ⚠ |")
            L.append(f"| looked-wrong | {t['low_iou_correct']} | {t['low_iou_wrong']} |")
            L.append("")
        L.append("_(median split is RELATIVE — it labels ~half 'high IoU' even when all IoU are low; "
                 "use the absolute / pointing tables for 'genuinely looked right'.)_")
        L.append(f"- **verdict: {a2['verdict']}**")
        L.append("")

    # Answer-span + boxed-span variants (boxed = the precise final answer; preferred).
    _render_lvu_compact(L, analysis.get("looking_vs_using_answer", {}), "answer-span (last-K)")
    _render_lvu_compact(L, analysis.get("looking_vs_using_boxed", {}), "boxed-span (\\boxed{}, PRIMARY)")

    by_sub = analysis.get("looking_vs_using_by_subset", {})
    if by_sub:
        L.append(f"## Looking-vs-using by task type ({analysis.get('headline_span','first')}-span)")
        L.append("")
        L.append("| subset | n | acc | corr(c,IoU) | corr(c,vt) | look-right | verdict |")
        L.append("|--------|---|-----|-------------|------------|-----------|---------|")
        for sub, e in by_sub.items():
            flag = " ⚠low-n" if e.get("low_n") else ""
            L.append(f"| {sub}{flag} | {e['n']} | {_fmt(e['accuracy'])} | {_fmt(e['corr_correct_iou_lh'])} | "
                     f"{_fmt(e['corr_correct_vt'])} | {_fmt(e.get('looked_right_rate_abs'))} | {e['verdict_short']} |")
        L.append("")
        L.append("_Object/spatial subsets (gqa/openimages/vsr/visual7w) vs OCR/doc (docvqa/textvqa/sroie) "
                 "often give different verdicts — read this table, not just the pooled number._")
        L.append("")

    if a3:
        L.append("## Analysis 3 — hint mechanism (C1 vs C2, paired)")
        L.append(f"- n_paired={a3['n_paired']}, Δaccuracy={_fmt(a3.get('delta_accuracy'))} "
                 f"(C1={_fmt(a3.get('acc_c1'))} → C2={_fmt(a3.get('acc_c2'))})")
        L.append(f"- ΔIoU_LH={_fmt(a3.get('delta_iou_lh'))}, ΔIoU_GL={_fmt(a3.get('delta_iou_gl'))}, "
                 f"Δvt_ratio={_fmt(a3.get('delta_vt_ratio'))}")
        L.append(f"- **verdict: {a3.get('verdict')}**")
        if a3.get("note"):
            L.append(f"- _note: {a3['note']}_")
        L.append("")

    if a4:
        L.append("## Analysis 4 — teacher-vs-student gap")
        L.append(f"- IoU_LH: teacher={_fmt(a4['teacher_mean_iou_lh'])} vs student={_fmt(a4['student_mean_iou_lh'])} "
                 f"(gap={_fmt(a4['gap_iou_lh'])})")
        L.append(f"- vt_ratio: teacher={_fmt(a4['teacher_mean_vt'])} vs student={_fmt(a4['student_mean_vt'])} "
                 f"(gap={_fmt(a4['gap_vt'])})")
        L.append(f"- accuracy: teacher={_fmt(a4['teacher_acc'])} vs student={_fmt(a4['student_acc'])}")
        L.append(f"- **dominant gap: {a4['dominant_gap']}**")
        if a4.get("note"):
            L.append(f"- _note: {a4['note']}_")
        L.append("")

    L.append("## Decision (G0 manual §7)")
    # Prefer the boxed-span verdict (precise final answer) for the decision.
    a2_primary = analysis.get("looking_vs_using_boxed") or analysis.get("looking_vs_using_answer") or a2
    L.append(_decision(a1, a2_primary, a3))
    with open(os.path.join(run_dir, "report.md"), "w") as f:
        f.write("\n".join(L) + "\n")


def _fmt(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def _decision(a1, a2, a3) -> str:
    head = a1.get("student", {}).get("verdict", "unknown")
    fail = a2.get("verdict", "unknown") if a2 else "unknown"
    hint = a3.get("verdict", "unknown") if a3 else "unknown"
    if "using" in fail:
        return ("- Student failure is **using** → looking right doesn't convert to right answers; "
                "an LH-box/attention intervention won't raise accuracy. Push the **output-level** "
                "hidden-hint OPD; keep LH only for label-free pseudo-boxes / diagnostics. "
                f"(2B head usability: {head}; hint mechanism: {hint}.)")
    if "looking" in fail:
        return ("- Student failure is **looking** → fixing where-to-look has headroom. If 2B heads are "
                f"usable ({head}), LH-region-as-hint / explicit grounding is well-motivated. "
                f"(hint mechanism: {hint}.)")
    return f"- Inconclusive (head:{head}; failure:{fail}; hint:{hint}). Gather more samples before deciding."


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze a G0 run.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--iou-threshold", type=float, default=None,
                    help="Relative (median) IoU_LH split for the 2x2 (default: median of C3).")
    ap.add_argument("--abs-iou-threshold", type=float, default=0.30,
                    help="ABSOLUTE 'looked right' IoU bar (the verdict uses correlations, not this).")
    ap.add_argument("--no-figs", action="store_true", help="Skip figures (faster mid-run preview).")
    ap.add_argument("--use-judge", action="store_true",
                    help="Overlay judgments.jsonl (baseline.g0.judge_g0) — use LLM correct_judge "
                         "instead of the rule grader for the correctness axis.")
    ap.add_argument("--span", default="boxed", choices=["boxed", "answer", "first"],
                    help="Which answer span drives the headline looking-vs-using + per-subset tables "
                         "(boxed=\\boxed{} primary; answer=last-K; first=first-gen-step). All variants "
                         "are still computed.")
    args = ap.parse_args()

    records = load_records(args.run_dir)
    if args.use_judge:
        apply_judge(args.run_dir, records)
    conds = by_condition(records)
    # Live progress line (handy when previewing a still-running sharded job).
    prog = " ".join(f"{c}={len(v)}" for c, v in sorted(conds.items()))
    print(f"[g0.analyze] {len(records)} records so far ({prog})")
    analysis = {
        "n_records": len(records),
        "conditions": {c: len(v) for c, v in conds.items()},
        "head_usability": analysis_1_head_usability(args.run_dir, conds),
        "looking_vs_using": analysis_2_looking_vs_using(
            conds.get("c3", []), iou_threshold=args.iou_threshold, abs_iou_threshold=args.abs_iou_threshold),
        "hint_mechanism": analysis_3_hint_mechanism(conds.get("c1", []), conds.get("c2", [])),
        "gap": analysis_4_gap(conds.get("c1", []), conds.get("c3", [])),
    }
    c3 = conds.get("c3", [])
    # If answer-span / boxed-span LH was recorded (newer runs), also run the KEY
    # analysis on each — the verdict prefers the boxed span (precise final answer).
    if any("iou_lh_answer" in r for r in c3):
        analysis["looking_vs_using_answer"] = analysis_2_looking_vs_using(
            c3, iou_threshold=args.iou_threshold,
            abs_iou_threshold=args.abs_iou_threshold, iou_key="iou_lh_answer")
    if any("iou_lh_boxed" in r for r in c3):
        analysis["looking_vs_using_boxed"] = analysis_2_looking_vs_using(
            c3, iou_threshold=args.iou_threshold,
            abs_iou_threshold=args.abs_iou_threshold, iou_key="iou_lh_boxed")
    # Per-task-type (Visual-CoT subset) breakdown of the headline correlations.
    span_iou_key = {"boxed": "iou_lh_boxed", "answer": "iou_lh_answer", "first": "iou_lh"}[args.span]
    if not any(span_iou_key in r for r in c3):
        span_iou_key = "iou_lh"
    analysis["looking_vs_using_by_subset"] = looking_vs_using_by_subset(c3, iou_key=span_iou_key)
    analysis["headline_span"] = args.span
    analysis["correctness_source"] = "llm_judge" if args.use_judge else "rule"
    with open(os.path.join(args.run_dir, "analysis.json"), "w") as f:
        json.dump(analysis, f, indent=2)
    if not args.no_figs:
        make_figures(args.run_dir, conds, analysis["head_usability"], analysis["looking_vs_using"])
    write_report(args.run_dir, analysis)
    print(json.dumps(analysis, indent=2))
    print(f"\n[g0.analyze] wrote analysis.json, report.md, figs/ → {args.run_dir}")


if __name__ == "__main__":
    main()
