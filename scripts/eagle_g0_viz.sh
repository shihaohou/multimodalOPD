#!/usr/bin/env bash
set -euo pipefail

# Post-hoc EAGLE-G0 map renderer.
#
# It reuses an existing eval_outputs/eagle_g0/<model>/ run, selects a few records
# from records*.jsonl, and reruns only those samples with visualization enabled.
# No judge or full evaluation is rerun here.
#
# Examples:
#   RUN_DIRS=eval_outputs/eagle_g0/qwen3vl-8b bash scripts/eagle_g0_viz.sh
#   SELECTS=wrong,correct SPAN_MODES=answer,sentence PER_SUBSET=1 GPU=0 bash scripts/eagle_g0_viz.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT_DIR"

export TRANSFORMERS_NO_TF=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -n "${PYRUN:-}" ]]; then read -r -a PY <<< "$PYRUN";
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then PY=(python);
else PY=(uv run python); fi

[[ -n "${GPU:-}" ]] && export CUDA_VISIBLE_DEVICES="$GPU"

OUTPUT_BASE="${OUTPUT_BASE:-eval_outputs/eagle_g0}"
SELECTION_RUN_DIR="${SELECTION_RUN_DIR:-}"
CASE_MANIFEST="${CASE_MANIFEST:-}"
SELECT="${SELECT:-wrong}"
SELECTS="${SELECTS:-$SELECT}"
SPAN_MODES="${SPAN_MODES:-answer}"
RANK_CONDITION="${RANK_CONDITION:-plain}"
CONDITIONS="${CONDITIONS:-plain,hint}"
PER_SUBSET="${PER_SUBSET:-3}"
MAX_CASES="${MAX_CASES:-}"
SUBSETS="${SUBSETS:-}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
EAGLE_BATCH_SIZE="${EAGLE_BATCH_SIZE:-128}"
EAGLE_IMAGE_SIZE="${EAGLE_IMAGE_SIZE:-}"
EAGLE_TOKEN_MODE="${EAGLE_TOKEN_MODE:-span}"
EAGLE_TOKEN_LIMIT="${EAGLE_TOKEN_LIMIT:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
ATTN="${ATTN:-sdpa}"
WITH_SALR1="${WITH_SALR1:-0}"
NO_USE_JUDGE="${NO_USE_JUDGE:-0}"
SAVE_EAGLE_ARTIFACTS="${SAVE_EAGLE_ARTIFACTS:-1}"

run_dirs=()
if [[ -n "${RUN_DIRS:-}" ]]; then
  IFS=';' read -r -a run_dirs <<< "$RUN_DIRS"
elif [[ -d "$OUTPUT_BASE" ]]; then
  while IFS= read -r cfg; do
    run_dirs+=("${cfg%/config.json}")
  done < <(find "$OUTPUT_BASE" -mindepth 2 -maxdepth 2 -name config.json -print | sort)
fi

if [[ "${#run_dirs[@]}" -eq 0 ]]; then
  echo "[eagle_viz] no run dirs found. Set RUN_DIRS=eval_outputs/eagle_g0/<model>." >&2
  exit 1
fi

for d in "${run_dirs[@]}"; do
  [[ -z "$d" ]] && continue
  if [[ ! -f "$d/config.json" ]]; then
    echo "[eagle_viz] skip $d: missing config.json" >&2
    continue
  fi

  args=(
    --run-dir "$d"
    --selects "$SELECTS"
    --span-modes "$SPAN_MODES"
    --rank-condition "$RANK_CONDITION"
    --conditions "$CONDITIONS"
    --per-subset "$PER_SUBSET"
    --eagle-batch-size "$EAGLE_BATCH_SIZE"
    --eagle-token-mode "$EAGLE_TOKEN_MODE"
    --eagle-token-limit "$EAGLE_TOKEN_LIMIT"
  )
  [[ -n "$MAX_CASES" ]] && args+=(--max-cases "$MAX_CASES")
  [[ -n "$SUBSETS" ]] && args+=(--subsets "$SUBSETS")
  [[ -n "$SELECTION_RUN_DIR" ]] && args+=(--selection-run-dir "$SELECTION_RUN_DIR")
  [[ -n "$CASE_MANIFEST" ]] && args+=(--case-manifest "$CASE_MANIFEST")
  [[ -n "$OUTPUT_SUBDIR" ]] && args+=(--output-subdir "$OUTPUT_SUBDIR")
  [[ -n "$OUTPUT_ROOT" ]] && args+=(--output-root "$OUTPUT_ROOT")
  [[ -n "$EAGLE_IMAGE_SIZE" ]] && args+=(--eagle-image-size "$EAGLE_IMAGE_SIZE")
  [[ -n "$MAX_NEW_TOKENS" ]] && args+=(--max-new-tokens "$MAX_NEW_TOKENS")
  [[ -n "$ATTN" ]] && args+=(--attn "$ATTN")
  [[ "$WITH_SALR1" == "1" || "$WITH_SALR1" == "true" ]] && args+=(--with-salr1)
  [[ "$NO_USE_JUDGE" == "1" || "$NO_USE_JUDGE" == "true" ]] && args+=(--no-use-judge)
  [[ "$SAVE_EAGLE_ARTIFACTS" == "1" || "$SAVE_EAGLE_ARTIFACTS" == "true" ]] && args+=(--save-eagle-artifacts)

  echo "[eagle_viz] rendering $d selects=$SELECTS span_modes=$SPAN_MODES token_mode=$EAGLE_TOKEN_MODE per_subset=$PER_SUBSET conditions=$CONDITIONS"
  "${PY[@]}" -m baseline.g0.viz_eagle_g0 "${args[@]}"
done
