#!/usr/bin/env bash
set -euo pipefail

# Parallel post-hoc EAGLE-G0 map renderer.
#
# Default task split is model x selector x span mode. With two models, two
# selectors, and answer/sentence spans this yields eight independent GPU tasks.
#
# Example:
#   GPUS=0,1,2,3,4,5,6,7 SELECTS=wrong,correct SPAN_MODES=answer,sentence \
#     PER_SUBSET=1 EAGLE_BATCH_SIZE=128 bash scripts/eagle_g0_viz_multi.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT_DIR"

OUTPUT_BASE="${OUTPUT_BASE:-eval_outputs/eagle_g0}"
GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
SELECTS="${SELECTS:-wrong,correct}"
SPAN_MODES="${SPAN_MODES:-answer,sentence}"
PER_SUBSET="${PER_SUBSET:-1}"
CONDITIONS="${CONDITIONS:-plain,hint}"
EAGLE_BATCH_SIZE="${EAGLE_BATCH_SIZE:-128}"
EAGLE_TOKEN_MODE="${EAGLE_TOKEN_MODE:-span}"
EAGLE_TOKEN_LIMIT="${EAGLE_TOKEN_LIMIT:-16}"
SAVE_EAGLE_ARTIFACTS="${SAVE_EAGLE_ARTIFACTS:-1}"
SPLIT_SPAN_MODES="${SPLIT_SPAN_MODES:-1}"
CASE_MANIFEST="${CASE_MANIFEST:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"

run_dirs=()
if [[ -n "${RUN_DIRS:-}" ]]; then
  IFS=';' read -r -a run_dirs <<< "$RUN_DIRS"
elif [[ -d "$OUTPUT_BASE" ]]; then
  while IFS= read -r cfg; do
    run_dirs+=("${cfg%/config.json}")
  done < <(find "$OUTPUT_BASE" -mindepth 2 -maxdepth 2 -name config.json -print | sort)
fi

if [[ "${#run_dirs[@]}" -eq 0 ]]; then
  echo "[eagle_viz_multi] no run dirs found. Set RUN_DIRS=eval_outputs/eagle_g0/<model>[;...]." >&2
  exit 1
fi
SELECTION_RUN_DIR="${SELECTION_RUN_DIR:-${run_dirs[0]}}"

IFS=',' read -r -a gpu_arr <<< "$GPUS"
IFS=',' read -r -a selector_arr <<< "$SELECTS"
IFS=',' read -r -a span_arr <<< "$SPAN_MODES"
if [[ "${#gpu_arr[@]}" -eq 0 ]]; then
  echo "[eagle_viz_multi] GPUS is empty." >&2
  exit 1
fi

tasks=()
for d in "${run_dirs[@]}"; do
  [[ -z "$d" ]] && continue
  for selector in "${selector_arr[@]}"; do
    selector="${selector//[[:space:]]/}"
    [[ -z "$selector" ]] && continue
    if [[ "$SPLIT_SPAN_MODES" == "1" || "$SPLIT_SPAN_MODES" == "true" ]]; then
      for span_mode in "${span_arr[@]}"; do
        span_mode="${span_mode//[[:space:]]/}"
        [[ -z "$span_mode" ]] && continue
        tasks+=("$d|$selector|$span_mode")
      done
    else
      tasks+=("$d|$selector|$SPAN_MODES")
    fi
  done
done

if [[ "${#tasks[@]}" -eq 0 ]]; then
  echo "[eagle_viz_multi] no tasks." >&2
  exit 1
fi

echo "[eagle_viz_multi] run_dirs=${#run_dirs[@]} tasks=${#tasks[@]} gpus=$GPUS selectors=$SELECTS span_modes=$SPAN_MODES token_mode=$EAGLE_TOKEN_MODE per_subset=$PER_SUBSET selection_run_dir=$SELECTION_RUN_DIR"

worker() {
  local gpu="$1"
  local worker_idx="$2"
  local ngpu="$3"
  local fail=0
  for ((i = worker_idx; i < ${#tasks[@]}; i += ngpu)); do
    local task="${tasks[$i]}"
    local d="${task%%|*}"
    local rest="${task#*|}"
    local selector="${rest%%|*}"
    local span_modes="${rest#*|}"
    local span_tag="${span_modes//,/_}"
    local log="$d/viz_${selector}_${span_tag}_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p "$d"
    echo "[eagle_viz_multi] GPU $gpu -> $d selector=$selector spans=$span_modes log=$log"
    if ! GPU="$gpu" RUN_DIRS="$d" SELECTS="$selector" SPAN_MODES="$span_modes" \
      PER_SUBSET="$PER_SUBSET" CONDITIONS="$CONDITIONS" EAGLE_BATCH_SIZE="$EAGLE_BATCH_SIZE" \
      EAGLE_TOKEN_MODE="$EAGLE_TOKEN_MODE" EAGLE_TOKEN_LIMIT="$EAGLE_TOKEN_LIMIT" \
      SELECTION_RUN_DIR="$SELECTION_RUN_DIR" \
      CASE_MANIFEST="$CASE_MANIFEST" OUTPUT_ROOT="$OUTPUT_ROOT" \
      SAVE_EAGLE_ARTIFACTS="$SAVE_EAGLE_ARTIFACTS" \
      bash scripts/eagle_g0_viz.sh > "$log" 2>&1; then
      echo "[eagle_viz_multi] FAILED GPU $gpu -> $d selector=$selector (see $log)" >&2
      fail=$((fail + 1))
    fi
  done
  return "$fail"
}

pids=()
for idx in "${!gpu_arr[@]}"; do
  worker "${gpu_arr[$idx]}" "$idx" "${#gpu_arr[@]}" &
  pids+=("$!")
done

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=$((fail + 1))
done

if [[ "$fail" -gt 0 ]]; then
  echo "[eagle_viz_multi] $fail worker(s) had failed task(s)." >&2
  exit 1
fi

echo "[eagle_viz_multi] done. PNGs are under eval_outputs/eagle_g0/<model>/viz_{wrong,correct}_{answer,sentence}/"
