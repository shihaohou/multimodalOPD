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
#   SELECT=low_iou_eagle PER_SUBSET=5 GPU=0 bash scripts/eagle_g0_viz.sh

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
SELECT="${SELECT:-wrong}"
RANK_CONDITION="${RANK_CONDITION:-plain}"
CONDITIONS="${CONDITIONS:-plain,hint}"
PER_SUBSET="${PER_SUBSET:-3}"
MAX_CASES="${MAX_CASES:-}"
SUBSETS="${SUBSETS:-}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-}"
EAGLE_BATCH_SIZE="${EAGLE_BATCH_SIZE:-128}"
EAGLE_IMAGE_SIZE="${EAGLE_IMAGE_SIZE:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
ATTN="${ATTN:-}"
WITH_SALR1="${WITH_SALR1:-0}"
NO_USE_JUDGE="${NO_USE_JUDGE:-0}"

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
    --select "$SELECT"
    --rank-condition "$RANK_CONDITION"
    --conditions "$CONDITIONS"
    --per-subset "$PER_SUBSET"
    --eagle-batch-size "$EAGLE_BATCH_SIZE"
  )
  [[ -n "$MAX_CASES" ]] && args+=(--max-cases "$MAX_CASES")
  [[ -n "$SUBSETS" ]] && args+=(--subsets "$SUBSETS")
  [[ -n "$OUTPUT_SUBDIR" ]] && args+=(--output-subdir "$OUTPUT_SUBDIR")
  [[ -n "$EAGLE_IMAGE_SIZE" ]] && args+=(--eagle-image-size "$EAGLE_IMAGE_SIZE")
  [[ -n "$MAX_NEW_TOKENS" ]] && args+=(--max-new-tokens "$MAX_NEW_TOKENS")
  [[ -n "$ATTN" ]] && args+=(--attn "$ATTN")
  [[ "$WITH_SALR1" == "1" || "$WITH_SALR1" == "true" ]] && args+=(--with-salr1)
  [[ "$NO_USE_JUDGE" == "1" || "$NO_USE_JUDGE" == "true" ]] && args+=(--no-use-judge)

  echo "[eagle_viz] rendering $d select=$SELECT per_subset=$PER_SUBSET conditions=$CONDITIONS"
  "${PY[@]}" -m baseline.g0.viz_eagle_g0 "${args[@]}"
done
