#!/usr/bin/env bash
set -euo pipefail

# Phase B of the DECOUPLED eval: judge a suite's already-generated judged-group
# responses with NO GPU, in ONE controlled --judge-workers pool. This is the fix for
# "all the judges pile onto the judge at once": generation fans out over all GPUs
# (Phase A, with SKIP_JUDGE=true), then judging runs here as a single process per
# model, so the judge only ever sees JUDGE_WORKERS concurrent requests — no 7x stack.
#
# Prereq (Phase A): generate with SKIP_JUDGE=true so $OUTPUT_ROOT/judged/responses/*.jsonl
# exist. Needs the SAME JUDGED_DATASETS, plus the judge env.
# Usage (one model):
#   OUTPUT_ROOT=eval_outputs/suite_<tag> MODEL_NAME=<tag> JUDGED_DATASETS=... \
#   JUDGE_API_URL=http://127.0.0.1:8000/v1 JUDGE_MODEL=judge OPENAI_API_KEY=dummy \
#   JUDGE_WORKERS=48 bash scripts/judge_suite.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT_DIR"

: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the suite dir (e.g. eval_outputs/suite_<tag>)}"
: "${JUDGED_DATASETS:?Set JUDGED_DATASETS (same value used for generation)}"

export TRANSFORMERS_NO_TF=1
export TOKENIZERS_PARALLELISM=false

MODEL_NAME="${MODEL_NAME:-$(basename "$OUTPUT_ROOT")}"
GRADER="${GRADER:-llm}"
JUDGE_MODEL="${JUDGE_MODEL:-judge}"
JUDGE_API_URL="${JUDGE_API_URL:-http://127.0.0.1:8000/v1}"
JUDGE_KEY_ENV="${JUDGE_KEY_ENV:-OPENAI_API_KEY}"
JUDGE_WORKERS="${JUDGE_WORKERS:-48}"        # single controlled pool: judge sees exactly this many
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-512}"
JUDGE_TIMEOUT="${JUDGE_TIMEOUT:-120}"
JUDGE_RETRIES="${JUDGE_RETRIES:-2}"
JUDGE_EXTRA_BODY="${JUDGE_EXTRA_BODY:-}"
PASSK_KS="${PASSK_KS:-1,8,16}"

uv run python baseline/eval/run_opd_eval.py --judge-only \
  --model-path "${MODEL_PATH:-judge-only}" --model-name "$MODEL_NAME" \
  --output-dir "$OUTPUT_ROOT/judged" --datasets "$JUDGED_DATASETS" --benchmarks "" \
  --grader "$GRADER" \
  --judge-model "$JUDGE_MODEL" --judge-api-url "$JUDGE_API_URL" --judge-key-env "$JUDGE_KEY_ENV" \
  --judge-workers "$JUDGE_WORKERS" --judge-max-tokens "$JUDGE_MAX_TOKENS" \
  --judge-timeout "$JUDGE_TIMEOUT" --judge-retries "$JUDGE_RETRIES" \
  --judge-extra-body "$JUDGE_EXTRA_BODY"

# Re-aggregate judged + (if present) deterministic into the final table.
AGG=(
  uv run python baseline/eval/aggregate_suite.py
  --judged-summary "$OUTPUT_ROOT/judged/summary.json"
  --ks "$PASSK_KS"
  --model-name "$MODEL_NAME"
  --output "$OUTPUT_ROOT/suite_summary.json"
)
if [[ -f "$OUTPUT_ROOT/vqa/summary.json" ]]; then
  AGG+=(--vqa-summary "$OUTPUT_ROOT/vqa/summary.json")
fi
"${AGG[@]}"
