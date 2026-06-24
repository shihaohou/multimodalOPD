"""Stage 0 analysis: Acc / Reliance / Delta_RG with paired bootstrap CIs + verdict.

Reads one or more ``per_sample.jsonl`` files (one per model from ``run_stage0.py``)
and computes, per subset and overall:

  Acc_full, Acc_mask_evidence, Acc_mask_random, Acc_crop@pad
  drop_evidence = Acc_full - Acc_mask_evidence
  drop_random   = Acc_full - Acc_mask_random
  Reliance      = drop_evidence - drop_random  ( = Acc_mask_random - Acc_mask_evidence )
  Delta_RG@pad  = Acc_crop@pad - Acc_full

Every metric gets a percentile bootstrap CI resampled over samples (paired: the
same resampled indices are used across conditions, and across teacher/student for
the Delta_RG difference). Encodes the go/no-go gate from the experiment plan:

  GO   if teacher Reliance is significantly > 0 (CI low > 0) -- its advantage is
       (at least partly) evidence-causal/visual.   Corroborated by teacher
       Delta_RG < student Delta_RG (teacher already focuses on the full image).
  STOP if teacher Reliance ~ 0 everywhere -- the advantage is not visual.

  uv run python baseline/probe/analyze_stage0.py \
      --model teacher=probe_outputs/stage0/MMR1-7B-RL \
      --model student=probe_outputs/stage0/MMR1-3B-SFT \
      --teacher teacher --student student --output probe_outputs/stage0/summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 0 analysis (Reliance / Delta_RG + bootstrap).")
    p.add_argument("--model", action="append", required=True, metavar="LABEL=PATH",
                   help="Repeatable. PATH = per_sample.jsonl or the model dir holding it.")
    p.add_argument("--teacher", default=None, help="Model label for the go/no-go gate.")
    p.add_argument("--student", default=None, help="Model label compared against the teacher.")
    p.add_argument("--primary-pad", type=float, default=0.1, help="Crop pad used for the Delta_RG gate.")
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--ci", type=float, default=0.95)
    p.add_argument("--min-effect", type=float, default=0.02, help="Min Reliance point to count as real.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=None, help="Write summary JSON here.")
    return p.parse_args()


# ----------------------------------------------------------------------- load
def _resolve_jsonl(path: str) -> Path:
    p = Path(path)
    if p.is_dir():
        p = p / "per_sample.jsonl"
    if not p.exists():
        raise SystemExit(f"per_sample.jsonl not found at {path}")
    return p


def load_model(path: str) -> dict[str, dict[str, dict[str, float]]]:
    """-> data[subset][sample_id] = {full, mask_evidence, mask_rand, crop_pad{p}}.

    Averages the N random-mask placements into one ``mask_rand`` value per sample.
    """
    rows: dict[str, dict[str, dict[str, list[int]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with _resolve_jsonl(path).open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("correct") is None:
                continue
            cond = r["condition"]
            key = cond if cond != "crop" else f"crop_{r['variant'].replace('pad', '')}"
            rows[r["subset"]][r["sample_id"]][key].append(int(r["correct"]))
    data: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for subset, by_id in rows.items():
        for sid, by_key in by_id.items():
            data[subset][sid] = {k: float(np.mean(v)) for k, v in by_key.items()}
    return data


def crop_pads(data: dict[str, dict[str, dict[str, float]]]) -> list[str]:
    keys: set[str] = set()
    for by_id in data.values():
        for rec in by_id.values():
            keys.update(k for k in rec if k.startswith("crop_"))
    return sorted(keys, key=lambda k: float(k.split("_", 1)[1]))


# ------------------------------------------------------------------ bootstrap
def _ci(samples: np.ndarray, ci: float) -> dict[str, float]:
    alpha = (1.0 - ci) / 2.0
    return {
        "point": float(samples.mean()),
        "lo": float(np.quantile(samples, alpha)),
        "hi": float(np.quantile(samples, 1.0 - alpha)),
    }


def _arrays(data_subset: dict[str, dict[str, float]], ids: list[str], pad_keys: list[str]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    base_keys = ["full", "mask_evidence", "mask_rand", *pad_keys]
    for k in base_keys:
        out[k] = np.asarray([data_subset[i].get(k, np.nan) for i in ids], dtype=float)
    return out


def analyze_model(data, *, bootstrap: int, ci: float, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    pad_keys = crop_pads(data)
    result: dict[str, Any] = {}
    # Per-subset, plus a pooled "overall".
    subsets = sorted(data.keys())
    pooled_ids: list[tuple[str, str]] = []
    for subset in subsets:
        ids = sorted(data[subset].keys())
        result[subset] = _metrics(_arrays(data[subset], ids, pad_keys), pad_keys, bootstrap, ci, rng, n=len(ids))
        pooled_ids.extend((subset, i) for i in ids)
    if len(subsets) > 1:
        merged = {f"{s}|{i}": data[s][i] for s in subsets for i in data[s]}
        ids = sorted(merged.keys())
        result["overall"] = _metrics(_arrays(merged, ids, pad_keys), pad_keys, bootstrap, ci, rng, n=len(ids))
    else:
        result["overall"] = result[subsets[0]]
    result["_pad_keys"] = pad_keys
    return result


def _metrics(arr, pad_keys, bootstrap, ci, rng, *, n) -> dict[str, Any]:
    idx = rng.integers(0, n, size=(bootstrap, n)) if n > 0 else np.zeros((bootstrap, 0), dtype=int)

    def boot(values: np.ndarray) -> np.ndarray:
        return np.nanmean(values[idx], axis=1)

    full_b = boot(arr["full"])
    ev_b = boot(arr["mask_evidence"])
    rand_b = boot(arr["mask_rand"])
    reliance_b = rand_b - ev_b
    out: dict[str, Any] = {
        "n": int(n),
        "acc_full": _ci(full_b, ci),
        "acc_mask_evidence": _ci(ev_b, ci),
        "acc_mask_random": _ci(rand_b, ci),
        "drop_evidence": float(np.nanmean(arr["full"]) - np.nanmean(arr["mask_evidence"])),
        "drop_random": float(np.nanmean(arr["full"]) - np.nanmean(arr["mask_rand"])),
        "reliance": _ci(reliance_b, ci),
        "delta_rg": {},
    }
    for k in pad_keys:
        out["delta_rg"][k.split("_", 1)[1]] = _ci(boot(arr[k]) - full_b, ci)
    return out


def paired_drg_diff(t_data, s_data, subset, pad_key, *, bootstrap, ci, seed) -> dict[str, Any] | None:
    """Bootstrap CI of (teacher Delta_RG - student Delta_RG) over COMMON samples."""
    if subset not in t_data or subset not in s_data:
        return None
    common = sorted(set(t_data[subset]) & set(s_data[subset]))
    if not common:
        return None
    rng = np.random.default_rng(seed + 99)
    tf = np.asarray([t_data[subset][i].get("full", np.nan) for i in common])
    tc = np.asarray([t_data[subset][i].get(pad_key, np.nan) for i in common])
    sf = np.asarray([s_data[subset][i].get("full", np.nan) for i in common])
    sc = np.asarray([s_data[subset][i].get(pad_key, np.nan) for i in common])
    n = len(common)
    idx = rng.integers(0, n, size=(bootstrap, n))
    diff = (np.nanmean(tc[idx], axis=1) - np.nanmean(tf[idx], axis=1)) - (
        np.nanmean(sc[idx], axis=1) - np.nanmean(sf[idx], axis=1)
    )
    res = _ci(diff, ci)
    res["n_common"] = n
    return res


# --------------------------------------------------------------------- report
def _fmt(m: dict[str, float]) -> str:
    return f"{m['point']:+.3f} [{m['lo']:+.3f}, {m['hi']:+.3f}]"


def _fmt_acc(m: dict[str, float]) -> str:
    return f"{m['point']:.3f} [{m['lo']:.3f}, {m['hi']:.3f}]"


def print_model(label: str, res: dict[str, Any]) -> None:
    print(f"\n================ {label} ================")
    for subset in [k for k in res if not k.startswith("_")]:
        m = res[subset]
        print(f"\n  [{subset}] n={m['n']}")
        print(f"    Acc_full          {_fmt_acc(m['acc_full'])}")
        print(f"    Acc_mask_evidence {_fmt_acc(m['acc_mask_evidence'])}   (drop {m['drop_evidence']:+.3f})")
        print(f"    Acc_mask_random   {_fmt_acc(m['acc_mask_random'])}   (drop {m['drop_random']:+.3f})")
        print(f"    Reliance          {_fmt(m['reliance'])}   <- evidence-causal signal")
        for pad, d in m["delta_rg"].items():
            print(f"    Delta_RG pad={pad:<4} {_fmt(d)}")


def decide(teacher_res, student_res, t_data, s_data, args) -> dict[str, Any]:
    # Resolve the primary-pad crop key robustly (match by float value, not string).
    pad_keys = teacher_res.get("_pad_keys", [])
    pad_key = next(
        (k for k in pad_keys if abs(float(k.split("_", 1)[1]) - args.primary_pad) < 1e-9), None
    )
    pad_str = pad_key.split("_", 1)[1] if pad_key else f"{args.primary_pad:g}"
    go = {"teacher": args.teacher, "student": args.student, "primary_pad": args.primary_pad,
          "per_subset": {}, "reasons": []}
    subsets = [k for k in teacher_res if not k.startswith("_")]
    any_reliance = False
    for subset in subsets:
        rel = teacher_res[subset]["reliance"]
        reliance_pass = bool(rel["lo"] > 0.0 and rel["point"] >= args.min_effect)
        any_reliance = any_reliance or reliance_pass
        entry: dict[str, Any] = {"teacher_reliance": rel, "reliance_pass": reliance_pass}
        if student_res is not None and subset in student_res:
            t_drg = teacher_res[subset]["delta_rg"].get(pad_str)
            s_drg = student_res[subset]["delta_rg"].get(pad_str)
            diff = (paired_drg_diff(t_data, s_data, subset, pad_key,
                                    bootstrap=args.bootstrap, ci=args.ci, seed=args.seed)
                    if subset != "overall" and pad_key else None)
            entry["delta_rg_teacher"] = t_drg
            entry["delta_rg_student"] = s_drg
            entry["delta_rg_diff"] = diff
            entry["delta_rg_teacher_smaller"] = bool(
                t_drg is not None and s_drg is not None and t_drg["point"] < s_drg["point"]
            )
        go["per_subset"][subset] = entry

    overall_rel = teacher_res["overall"]["reliance"]
    overall_pass = bool(overall_rel["lo"] > 0.0 and overall_rel["point"] >= args.min_effect)
    if overall_pass or any_reliance:
        go["verdict"] = "GO"
        go["reasons"].append("Teacher Reliance is significantly > 0 (evidence-causal advantage present).")
    elif overall_rel["hi"] <= args.min_effect:
        go["verdict"] = "STOP"
        go["reasons"].append("Teacher Reliance ~ 0 everywhere: advantage is not visual -- do not train.")
    else:
        go["verdict"] = "AMBIGUOUS"
        go["reasons"].append("Teacher Reliance CI straddles 0: inconclusive, widen N or inspect per subset.")
    return go


def print_decision(go: dict[str, Any]) -> None:
    print("\n################  GO / NO-GO  ################")
    print(f"teacher={go['teacher']}  student={go['student']}  primary crop pad={go['primary_pad']}")
    for subset, e in go["per_subset"].items():
        flag = "PASS" if e["reliance_pass"] else "fail"
        line = f"  [{subset}] teacher Reliance {_fmt(e['teacher_reliance'])}  -> {flag}"
        if e.get("delta_rg_diff") is not None:
            d = e["delta_rg_diff"]
            smaller = "teacher<student" if e.get("delta_rg_teacher_smaller") else "teacher>=student"
            line += f"  | Delta_RG diff(t-s) {_fmt(d)} ({smaller})"
        print(line)
    print(f"\n  VERDICT: {go['verdict']}")
    for reason in go["reasons"]:
        print(f"    - {reason}")
    if go["verdict"] == "GO":
        print("    Next: Stage 1 -- train one short vanilla token-KL OPD ckpt, then re-probe 1a/1b.")


def main() -> None:
    args = parse_args()
    models: dict[str, str] = {}
    for spec in args.model:
        if "=" not in spec:
            raise SystemExit(f"--model expects LABEL=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        models[label.strip()] = path.strip()

    raw = {label: load_model(path) for label, path in models.items()}
    analyzed = {label: analyze_model(d, bootstrap=args.bootstrap, ci=args.ci, seed=args.seed)
                for label, d in raw.items()}
    for label, res in analyzed.items():
        print_model(label, res)

    summary: dict[str, Any] = {
        "models": {label: {k: v for k, v in res.items() if not k.startswith("_")} for label, res in analyzed.items()},
        "config": {"bootstrap": args.bootstrap, "ci": args.ci, "min_effect": args.min_effect,
                   "primary_pad": args.primary_pad, "seed": args.seed},
    }
    if args.teacher and args.teacher in analyzed:
        student_res = analyzed.get(args.student) if args.student else None
        go = decide(analyzed[args.teacher], student_res,
                    raw[args.teacher], raw.get(args.student) if args.student else None, args)
        print_decision(go)
        summary["go_no_go"] = go

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
