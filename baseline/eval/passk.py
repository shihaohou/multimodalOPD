"""Unbiased pass@k / avg@k over k samples, computed from the per-attempt judge
verdicts that ``run_opd_eval`` / ``run_vqa_eval`` already write.

Methodology (Chen et al. 2021, the Codex pass@k estimator): generate N >= max(k)
samples ONCE at temperature>0, count c correct per question, then estimate

    pass@k = 1 - C(N-c, k) / C(N, k)          (P[at least one of k draws correct])
    avg@k  = c / N                            (mean per-sample accuracy; k-independent)

so pass@8 and pass@16 come from the SAME 16 samples — no re-generation per k. This
is the sampling-robustness view used in reasoning / RL evals; it is **not** how
lmms-eval scores VLM benchmarks (those are greedy single-sample → that single
number is our ``acc@1`` / greedy pass, the one comparable to lmms-eval).
"""

from __future__ import annotations

import json
from math import comb
from pathlib import Path
from typing import Any


def pass_at_k(n: int, c: int, k: int) -> float | None:
    """Unbiased P(>=1 correct in k draws) from n samples with c correct.

    Returns None when k > n (can't estimate pass@k with fewer than k samples).
    ``comb(n - c, k)`` is 0 once (n - c) < k, so an all-or-some-correct question
    collapses to the expected 1.0 / 0.0.
    """
    if n <= 0 or k <= 0 or k > n:
        return None
    return 1.0 - comb(n - c, k) / comb(n, k)


def _n_correct(judgment: dict[str, Any]) -> tuple[int, int]:
    verdicts = judgment.get("judge_attempt_verdicts") or []
    n = len(verdicts)
    c = sum(1 for v in verdicts if str(v).strip().lower() == "correct")
    return n, c


def multi_k(judgments: list[dict[str, Any]], ks: list[int]) -> dict[str, Any]:
    """Aggregate pass@k (mean over questions) + avg (mean per-sample accuracy)."""
    pairs = [_n_correct(j) for j in judgments]
    pairs = [(n, c) for n, c in pairs if n > 0]
    if not pairs:
        return {"questions": 0, "n_samples": 0, "avg": None, "pass_at_k": {}}
    n_samples = min(n for n, _ in pairs)
    avg = sum(c / n for n, c in pairs) / len(pairs)
    pass_map: dict[int, float] = {}
    for k in ks:
        vals = [v for v in (pass_at_k(n, c, k) for n, c in pairs) if v is not None]
        if vals:
            pass_map[k] = sum(vals) / len(vals)
    return {
        "questions": len(pairs),
        "n_samples": n_samples,
        "avg": avg,
        "pass_at_k": pass_map,
    }


def multi_k_from_file(path: str | Path, ks: list[int]) -> dict[str, Any]:
    """Read a judgments JSONL (with ``judge_attempt_verdicts``) and aggregate."""
    records: list[dict[str, Any]] = []
    file_path = Path(path)
    if file_path.exists():
        with file_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return multi_k(records, ks)
