#!/usr/bin/env bash
set -euo pipefail

# Parallel post-hoc EAGLE-G0 map renderer.
#
# Default task split is model x selector: each process loads one model, renders
# both span modes, and workers are distributed across GPUS.
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

IFS=',' read -r -a gpu_arr <<< "$GPUS"
IFS=',' read -r -a selector_arr <<< "$SELECTS"
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
    tasks+=("$d|$selector")
  done
done

if [[ "${#tasks[@]}" -eq 0 ]]; then
  echo "[eagle_viz_multi] no tasks." >&2
  exit 1
fi

echo "[eagle_viz_multi] run_dirs=${#run_dirs[@]} tasks=${#tasks[@]} gpus=$GPUS selectors=$SELECTS span_modes=$SPAN_MODES per_subset=$PER_SUBSET"

worker() {
  local gpu="$1"
  local worker_idx="$2"
  local ngpu="$3"
  local fail=0
  for ((i = worker_idx; i < ${#tasks[@]}; i += ngpu)); do
    local task="${tasks[$i]}"
    local d="${task%%|*}"
    local selector="${task#*|}"
    local log="$d/viz_${selector}_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p "$d"
    echo "[eagle_viz_multi] GPU $gpu -> $d selector=$selector log=$log"
    if ! GPU="$gpu" RUN_DIRS="$d" SELECTS="$selector" SPAN_MODES="$SPAN_MODES" \
      PER_SUBSET="$PER_SUBSET" CONDITIONS="$CONDITIONS" EAGLE_BATCH_SIZE="$EAGLE_BATCH_SIZE" \
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
