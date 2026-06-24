"""Trajectory view: Acc / Reliance / Delta_RG vs OPD step, with the capability-line
residual that separates "OPD transferred vision" from "Reliance rose because Acc rose".

Reliance and Delta_RG both correlate with Acc across models (a stronger model relies
on evidence more), so an OPD checkpoint that simply gets more accurate moves on those
axes *for free*. Fit the reference models' Reliance~Acc line, then read each
checkpoint's residual = Reliance - predicted(Acc):

  resid < 0 (BELOW line): accuracy outran evidence-reliance -> DISSOCIATION
            (output/accuracy transferred, the visual grounding lagged).
  resid > 0 (ABOVE line): more Reliance than accuracy predicts -> OPD added visual
            grounding beyond raw capability.
  resid ~ 0 (ON line):    Reliance just tracks accuracy (capability proxy) -- this was
            checkpoint-65's case; the masking probe can't see past capability here.

  uv run python baseline/probe/analyze_trajectory.py \
      --ckpt c005=$P/c005 --ckpt c065=$P/c065 ... \
      --ref teacher=$P/Qwen3-VL-8B --ref before=$P/Qwen3-VL-2B \
      --ref Qwen2.5-VL-7B=$P/Qwen2.5-VL-7B --ref Qwen2.5-VL-3B=$P/Qwen2.5-VL-3B
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from baseline.probe.analyze_stage0 import _arrays, _ci, crop_pads, load_model


def _step(label: str) -> int:
    nums = re.findall(r"\d+", label)
    return int(nums[-1]) if nums else -1


def overall(data, pad: float, bootstrap: int, ci: float, seed: int) -> dict:
    """Pool all subsets and bootstrap overall Acc / Reliance / Delta_RG@pad."""
    rng = np.random.default_rng(seed)
    pad_keys = crop_pads(data)
    merged = {f"{s}|{i}": data[s][i] for s in data for i in data[s]}
    ids = sorted(merged.keys())
    n = len(ids)
    arr = _arrays(merged, ids, pad_keys)
    idx = rng.integers(0, n, size=(bootstrap, n))

    def boot(v):
        return np.nanmean(v[idx], axis=1)

    full_b = boot(arr["full"])
    rel_b = boot(arr["mask_rand"]) - boot(arr["mask_evidence"])
    pad_key = next((k for k in pad_keys if abs(float(k.split("_", 1)[1]) - pad) < 1e-9), None)
    drg_b = (boot(arr[pad_key]) - full_b) if pad_key else np.zeros_like(full_b)
    return {"n": n, "acc": _ci(full_b, ci), "reliance": _ci(rel_b, ci), "delta_rg": _ci(drg_b, ci)}


def parse_models(specs) -> dict[str, str]:
    out: dict[str, str] = {}
    for s in specs or []:
        if "=" not in s:
            raise SystemExit(f"expected LABEL=PATH, got {s!r}")
        k, v = s.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="OPD trajectory + capability-line residual.")
    p.add_argument("--ckpt", action="append", required=True, metavar="LABEL=PATH")
    p.add_argument("--ref", action="append", default=[], metavar="LABEL=PATH",
                   help="Reference models to fit the Reliance~Acc capability line (>=2).")
    p.add_argument("--primary-pad", type=float, default=0.1)
    p.add_argument("--bootstrap", type=int, default=4000)
    p.add_argument("--ci", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    ckpts = parse_models(args.ckpt)
    refs = parse_models(args.ref)
    ck = {k: overall(load_model(v), args.primary_pad, args.bootstrap, args.ci, args.seed) for k, v in ckpts.items()}
    rf = {k: overall(load_model(v), args.primary_pad, args.bootstrap, args.ci, args.seed) for k, v in refs.items()}

    fit = None
    if len(rf) >= 2:
        xs = np.array([m["acc"]["point"] for m in rf.values()])
        ys = np.array([m["reliance"]["point"] for m in rf.values()])
        slope, intercept = (float(c) for c in np.polyfit(xs, ys, 1))
        fit = {"slope": slope, "intercept": intercept}

    def resid(m):
        return None if not fit else m["reliance"]["point"] - (fit["slope"] * m["acc"]["point"] + fit["intercept"])

    print("\n=== reference capability line (Reliance ~ Acc) ===")
    if fit:
        print(f"  Reliance ≈ {fit['slope']:.3f}*Acc {fit['intercept']:+.3f}   (fit on {len(rf)} models)")
        for k, m in sorted(rf.items(), key=lambda kv: kv[1]["acc"]["point"]):
            print(f"    {k:18s} Acc {m['acc']['point']:.3f}  Reliance {m['reliance']['point']:.3f}  resid {resid(m):+.3f}")
    else:
        print("  (pass >=2 --ref models to fit the line; trajectory only)")

    print("\n=== OPD trajectory (sorted by step) ===")
    print(f"  {'step':>5} {'n':>5}  {'Acc_full':>16} {'Reliance':>20} {'Delta_RG@p':>20} {'resid':>7}")
    rows = []
    for label, m in sorted(ck.items(), key=lambda kv: _step(kv[0])):
        r = resid(m)

        def fa(c):
            return f"{c['point']:.3f}[{c['lo']:.2f},{c['hi']:.2f}]"

        def fs(c):
            return f"{c['point']:+.3f}[{c['lo']:+.2f},{c['hi']:+.2f}]"

        rstr = "" if r is None else f"{r:+.3f}"
        print(f"  {_step(label):>5} {m['n']:>5}  {fa(m['acc']):>16} {fs(m['reliance']):>20} {fs(m['delta_rg']):>20} {rstr:>7}")
        rows.append({"label": label, "step": _step(label), "n": m["n"], "acc": m["acc"],
                     "reliance": m["reliance"], "delta_rg": m["delta_rg"], "reliance_residual": r})

    if fit:
        print("\n  resid < 0 (below line): accuracy outran evidence-reliance -> DISSOCIATION (output transferred, vision lagged).")
        print("  resid > 0 (above line): more Reliance than accuracy predicts -> OPD added visual grounding.")
        print("  resid ~ 0 (on line):    Reliance just tracks accuracy (capability proxy).")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(
            {"fit": fit, "trajectory": rows,
             "refs": {k: {kk: m[kk] for kk in ("n", "acc", "reliance", "delta_rg")} for k, m in rf.items()}},
            indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
