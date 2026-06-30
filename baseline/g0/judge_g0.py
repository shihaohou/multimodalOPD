"""LLM-judge correctness for a G0 run — the trustworthy ``correct`` signal.

The rule grader (``baseline.eval.grading.attempt_correct``) under-scores free-form
answers (paraphrases, units, "the bird is a cardinal" vs "cardinal"), which biases
the G0 looking-vs-using verdict: the correctness axis is exactly what we correlate
IoU / vt_ratio against. This pass re-grades each record's answer with an
OpenAI-compatible LLM judge (same prompts + dummy-key/no-think handling as
``验证/compare_bbox_prompt.py`` and ``baseline.eval.run_opd_eval``).

It reads a run dir's ``records*.jsonl`` and writes a **sidecar** ``judgments.jsonl``
— one line per ``(model, condition, subset, sample_id)`` with ``correct_judge`` —
deliberately NOT named ``records*.jsonl`` so it does not get re-globbed as records.
``analyze_g0 --use-judge`` overlays it (preferring ``correct_judge`` over the rule
``correct``).

Run (self-hosted Qwen judge, no key needed):

    uv run python -m baseline.g0.judge_g0 --run-dir eval_outputs/g0/run1 \
        --judge-api-url http://localhost:8000/v1 --judge-model Qwen3-... --judge-no-think
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from baseline.g0.analyze_g0 import load_records


def _rec_key(r: dict) -> tuple:
    return (r.get("model", ""), r.get("condition", ""), r.get("subset", ""), str(r.get("sample_id", "")))


def _pred_of(r: dict) -> str:
    """Best available model answer for judging: the boxed answer, else completion."""
    pred = (r.get("pred_boxed") or "").strip()
    if pred:
        return pred
    return (r.get("completion") or "").strip()


def judge_records(records: list[dict], args) -> dict[tuple, dict]:
    """Return ``{rec_key: {"correct_judge": bool, "verdict": str}}``.

    Dedupes identical ``(question, solution, pred)`` triples so repeated answers
    cost one judge call. Records missing ``question``/``solution`` (older runs) are
    skipped (left to the rule grader downstream).
    """
    from openai import OpenAI

    from vigos.eval_utils import build_judge_messages, parse_judge_output

    key = os.environ.get(args.judge_key_env) or os.environ.get("OPENAI_API_KEY") or "EMPTY"
    client = OpenAI(base_url=args.judge_api_url, api_key=key)
    extra_body = {"chat_template_kwargs": {"enable_thinking": False}} if args.judge_no_think else None

    # Build the unique work list (question, solution, pred).
    judged_recs = [r for r in records if r.get("question") and r.get("solution")]
    n_missing = len(records) - len(judged_recs)
    uniq: dict[tuple, dict] = {}
    for r in judged_recs:
        triple = (r["question"], r["solution"], _pred_of(r))
        uniq.setdefault(triple, {"messages": build_judge_messages(_pred_of(r), r["solution"], r["question"])})

    triples = list(uniq.keys())

    def judge_one(triple):
        try:
            resp = client.chat.completions.create(
                model=args.judge_model, messages=uniq[triple]["messages"], temperature=0.0,
                max_tokens=args.judge_max_tokens, timeout=args.judge_timeout, extra_body=extra_body,
            )
            verdict = parse_judge_output(resp.choices[0].message.content).get("verdict", "incorrect")
        except Exception as exc:  # noqa: BLE001
            return triple, ("error", f"{type(exc).__name__}: {exc}")
        return triple, (verdict, None)

    mode = "no-think" if extra_body else "thinking-on"
    print(f"[judge_g0] {len(judged_recs)} records, {len(triples)} unique answers to judge "
          f"(model={args.judge_model}, {mode}); {n_missing} records lack question/solution (skipped).",
          flush=True)
    verdicts: dict[tuple, str] = {}
    n_err = 0
    with ThreadPoolExecutor(max_workers=max(1, args.judge_workers)) as pool:
        for triple, (verdict, err) in pool.map(judge_one, triples):
            if err:
                n_err += 1
                if n_err <= 5:
                    print(f"[judge_g0] {err}", flush=True)
            verdicts[triple] = verdict
    if n_err:
        print(f"[judge_g0] {n_err}/{len(triples)} judge calls errored (counted incorrect).", flush=True)

    out: dict[tuple, dict] = {}
    for r in judged_recs:
        triple = (r["question"], r["solution"], _pred_of(r))
        v = verdicts.get(triple, "incorrect")
        out[_rec_key(r)] = {"correct_judge": v == "correct", "verdict": v}
    return out


def write_judgments(run_dir: str, records: list[dict], judged: dict[tuple, dict]) -> str:
    path = os.path.join(run_dir, "judgments.jsonl")
    with open(path, "w") as f:
        for r in records:
            jk = _rec_key(r)
            if jk not in judged:
                continue
            f.write(json.dumps({
                "model": r.get("model"), "condition": r.get("condition"),
                "subset": r.get("subset"), "sample_id": str(r.get("sample_id")),
                "correct_rule": bool(r.get("correct", False)),
                "correct_judge": judged[jk]["correct_judge"],
                "judge_verdict": judged[jk]["verdict"],
            }) + "\n")
    return path


def _acc_summary(records: list[dict], judged: dict[tuple, dict]) -> None:
    """Print rule-vs-judge accuracy per condition (and overall) for a quick read."""
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_cond[r.get("condition", "?")].append(r)
    print("\ncondition  n     acc(rule)  acc(judge)   Δ")
    print("-" * 48)
    for c, recs in sorted(by_cond.items()):
        matched = [(r, judged[_rec_key(r)]) for r in recs if _rec_key(r) in judged]
        if not matched:
            continue
        n = len(matched)
        rule = sum(int(r.get("correct", False)) for r, _ in matched) / n
        jud = sum(int(j["correct_judge"]) for _, j in matched) / n
        print(f"{c:<10} {n:<5} {rule:>8.3f}  {jud:>9.3f}   {jud - rule:+.3f}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="LLM-judge a G0 run's answers → judgments.jsonl")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--judge-api-url", default=os.environ.get("JUDGE_API_URL", "http://localhost:8000/v1"),
                    help="OpenAI-compatible base URL (self-hosted vLLM needs no real key).")
    ap.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", "Qwen3-30B-A3B"))
    ap.add_argument("--judge-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--judge-no-think", action="store_true",
                    help="Disable Qwen3 <think> (chat_template_kwargs.enable_thinking=False) so the "
                         "JSON verdict isn't truncated by a reasoning block.")
    ap.add_argument("--judge-max-tokens", type=int, default=2048)
    ap.add_argument("--judge-timeout", type=int, default=180)
    ap.add_argument("--judge-workers", type=int, default=16)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.run_dir)
    if not records:
        raise SystemExit(f"[judge_g0] no records*.jsonl in {args.run_dir}")
    judged = judge_records(records, args)
    path = write_judgments(args.run_dir, records, judged)
    _acc_summary(records, judged)
    print(f"\n[judge_g0] wrote {len(judged)} judgments → {path}")
    print(f"[judge_g0] now run: uv run python -m baseline.g0.analyze_g0 --run-dir {args.run_dir} --use-judge")


if __name__ == "__main__":
    main()
