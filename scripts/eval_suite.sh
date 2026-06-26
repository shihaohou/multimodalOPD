#!/usr/bin/env bash
set -euo pipefail

# ONE command -> the full OPD benchmark suite, merged into one table.
#
# The suite splits by grading philosophy (you do NOT judge benchmarks that have an
# official deterministic metric):
#   judged group (LLM judge)        : MathVista, MathVerse, MathVision, MMMU,
#                                     MMMU-Pro (x2 sub-scores), MMStar, HallusionBench
#                                     -> scripts/eval_opd.sh
#   deterministic group (no judge)  : POPE (3 categories + avg), ChartQA, VQAv2
#                                     -> scripts/eval_vqa.sh
# Then baseline/eval/aggregate_suite.py merges both summaries into one table, with
# POPE split by category (+avg) and MMMU-Pro split by sub-score (+avg).
#
# Required: MODEL_PATH, plus an LLM judge for the judged group (your 32B served
# OpenAI-compatible), unless you rule-grade (GRADER=rule) or SKIP_JUDGE=true:
#   export JUDGE_API_URL=http://<your-32b-host>:<port>/v1 \
#          JUDGE_MODEL=<served-name> OPENAI_API_KEY=<key>
# Usage:
#   CUDA_VISIBLE_DEVICES=0 MODEL_PATH=runs/<run> bash scripts/eval_suite.sh
#   # quick read: cap VQAv2 (its val set is ~214k) and smoke the rest:
#   MODEL_PATH=<model> VQAV2_LIMIT=2000 bash scripts/eval_suite.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT_DIR"

: "${MODEL_PATH:?Set MODEL_PATH to a model dir or HuggingFace model id.}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-eval_outputs/suite_${RUN_ID}}"
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"

# Greedy Acc@1 by default (the canonical single-number setting for these tables);
# set PASS_K>1 + GEN_TEMPERATURE>0 for a pass@k/avg@k robustness read instead.
PASS_K="${PASS_K:-1}"
GEN_TEMPERATURE="${GEN_TEMPERATURE:-0.0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

# Judged group (LLM judge). MMMU-Pro = the two sub-score datasets.
JUDGED_DATASETS="${JUDGED_DATASETS:-zli12321/mathvista,zli12321/mathverse,zli12321/mathvision,zli12321/MMMU,zli12321/mmmu_pro_10options,zli12321/mmmu-pro-vision,zli12321/mmstar,zli12321/hallusionbench}"
GRADER="${GRADER:-llm}"
SKIP_JUDGE="${SKIP_JUDGE:-false}"
JUDGE_KEY_ENV="${JUDGE_KEY_ENV:-DEEPSEEK_API_KEY}"

# Deterministic group. VQAv2 val is huge; cap it (POPE/ChartQA stay full).
# Point POPE_REPO / CHARTQA_REPO / VQAV2_REPO (+ *_SPLIT) at local dirs for an
# offline box; SKIP_DET=true runs the judged group only (POPE/ChartQA/VQAv2 -> '-').
DET_BENCHMARKS="${DET_BENCHMARKS:-pope,chartqa,vqav2}"
VQAV2_LIMIT="${VQAV2_LIMIT:-}"
SKIP_DET="${SKIP_DET:-false}"

# Optional: ALSO report pass@k / avg@k for the judged (math/MCQ) group. Adds a
# second, sampled generation pass (PASS_K=SAMPLED_K, temperature>0) over the judged
# datasets; pass@8 and pass@16 are estimated from the SAME N samples (unbiased
# estimator), so it costs ~N x the greedy generation, not N x per k. OFF by default.
# Not applied to POPE/ChartQA/VQAv2 — pass@k on yes/no / short-answer is not a
# standard metric there (their official greedy metric is the headline).
MULTI_K="${MULTI_K:-false}"
SAMPLED_K="${SAMPLED_K:-16}"               # N samples generated = max usable k
PASSK_KS="${PASSK_KS:-1,8,16}"            # which k to report pass@k for
SAMPLED_TEMPERATURE="${SAMPLED_TEMPERATURE:-1.0}"
SAMPLED_TOP_P="${SAMPLED_TOP_P:-0.9}"

# Fail early with a helpful message if the judged group has no judge configured.
if [[ "$GRADER" == "llm" && "$SKIP_JUDGE" != "true" ]]; then
  if [[ -z "${!JUDGE_KEY_ENV:-}" && -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: the judged group needs an LLM judge API key in \$$JUDGE_KEY_ENV (or \$OPENAI_API_KEY)." >&2
    echo "  Point the harness at your 32B judge, e.g.:" >&2
    echo "    export JUDGE_API_URL=http://<host>:<port>/v1 JUDGE_MODEL=<name> ${JUDGE_KEY_ENV}=<key>" >&2
    echo "  Or rule-grade (no API): GRADER=rule  (MCQ/yes-no fine; free-form math needs the judge)." >&2
    exit 1
  fi
fi

echo "== judged group (grader=$GRADER, greedy Acc@1) -> $OUTPUT_ROOT/judged =="
OUTPUT_DIR="$OUTPUT_ROOT/judged" MODEL_NAME="$MODEL_NAME" \
  EVAL_DATASETS="$JUDGED_DATASETS" EVAL_BENCHMARKS="" \
  PASS_K="$PASS_K" GEN_TEMPERATURE="$GEN_TEMPERATURE" \
  TENSOR_PARALLEL_SIZE="$TENSOR_PARALLEL_SIZE" \
  GRADER="$GRADER" SKIP_JUDGE="$SKIP_JUDGE" JUDGE_KEY_ENV="$JUDGE_KEY_ENV" \
  bash scripts/eval_opd.sh

if [[ "$SKIP_DET" != "true" ]]; then
  echo "== deterministic group (no judge, greedy official metric) -> $OUTPUT_ROOT/vqa =="
  OUTPUT_DIR="$OUTPUT_ROOT/vqa" MODEL_NAME="$MODEL_NAME" \
    BENCHMARKS="$DET_BENCHMARKS" \
    PASS_K="$PASS_K" GEN_TEMPERATURE="$GEN_TEMPERATURE" \
    TENSOR_PARALLEL_SIZE="$TENSOR_PARALLEL_SIZE" \
    VQAV2_LIMIT="$VQAV2_LIMIT" \
    bash scripts/eval_vqa.sh
else
  echo "== deterministic group SKIPPED (SKIP_DET=true) =="
fi

if [[ "$MULTI_K" == "true" ]]; then
  echo "== sampled pass for pass@k/avg@k (judged group, N=$SAMPLED_K temp=$SAMPLED_TEMPERATURE) -> $OUTPUT_ROOT/judged_sampled =="
  OUTPUT_DIR="$OUTPUT_ROOT/judged_sampled" MODEL_NAME="$MODEL_NAME" \
    EVAL_DATASETS="$JUDGED_DATASETS" EVAL_BENCHMARKS="" \
    PASS_K="$SAMPLED_K" GEN_TEMPERATURE="$SAMPLED_TEMPERATURE" GEN_TOP_P="$SAMPLED_TOP_P" \
    TENSOR_PARALLEL_SIZE="$TENSOR_PARALLEL_SIZE" \
    GRADER="$GRADER" SKIP_JUDGE="$SKIP_JUDGE" JUDGE_KEY_ENV="$JUDGE_KEY_ENV" \
    bash scripts/eval_opd.sh
fi

echo "== aggregate -> $OUTPUT_ROOT/suite_summary.json =="
AGG=(
  uv run python baseline/eval/aggregate_suite.py
  --judged-summary "$OUTPUT_ROOT/judged/summary.json"
  --ks "$PASSK_KS"
  --model-name "$MODEL_NAME"
  --output "$OUTPUT_ROOT/suite_summary.json"
)
if [[ "$SKIP_DET" != "true" ]]; then
  AGG+=(--vqa-summary "$OUTPUT_ROOT/vqa/summary.json")
fi
if [[ "$MULTI_K" == "true" ]]; then
  AGG+=(--sampled-summary "$OUTPUT_ROOT/judged_sampled/summary.json")
fi
"${AGG[@]}"
