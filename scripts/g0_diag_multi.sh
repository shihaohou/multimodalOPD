#!/usr/bin/env bash
set -uo pipefail

# G0 diagnostic across MULTIPLE teachers, FULL dataset, ALL GPUs.
# For each teacher it launches NUM_SHARDS data-parallel shards (one process per
# GPU, each decoding only its 1/N of the images), waits, then runs the analysis
# once over the merged shard records. Teachers run sequentially, each using all
# GPUs. The student (C3) is re-run per teacher (so each run dir is self-contained
# and C1/C2-vs-C3 is comparable within it).
#
# Required: STUDENT_MODEL, TEACHER_MODELS (comma-separated paths).
#
#   export D=/home/web_server/antispam/project/houshihao/datasets
#   export M=/home/web_server/antispam/project/houshihao/models
#   STUDENT_MODEL=$M/Qwen3-VL-2B-Instruct \
#   TEACHER_MODELS=$M/Qwen3-VL-8B-Instruct,$M/CapCurriculum-8B \
#   bash scripts/g0_diag_multi.sh
# → eval_outputs/g0/{Qwen3-VL-8B-Instruct,CapCurriculum-8B}/ each with report.md

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT_DIR"

: "${STUDENT_MODEL:?Set STUDENT_MODEL (the 2B student).}"
: "${TEACHER_MODELS:?Set TEACHER_MODELS=path1,path2 (comma-separated 8B teachers).}"

NUM_SHARDS="${NUM_SHARDS:-8}"                                   # one process per GPU
GPUS="${GPUS:-$(seq -s, 0 $((NUM_SHARDS - 1)))}"               # e.g. 0,1,2,3,4,5,6,7
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
IFS=',' read -r -a TEACHER_ARR <<< "$TEACHER_MODELS"

# Full dataset by default: all subsets, no per-subset cap.
if [[ -z "${DATASET:-}" ]]; then
  if [[ -n "${D:-}" && -d "$D/saliency-r1-8k" ]]; then DATASET="$D/saliency-r1-8k"; else DATASET="peterant330/saliency-r1-8k"; fi
fi
export DATASET
export SUBSETS="${SUBSETS:-}"     # "" = all subsets
export LIMIT="${LIMIT:-0}"        # 0 = no per-subset cap (full 8k)
OUTPUT_BASE="${OUTPUT_BASE:-eval_outputs/g0}"

echo "[g0_multi] student=$STUDENT_MODEL"
echo "[g0_multi] teachers=${TEACHER_ARR[*]}"
echo "[g0_multi] shards=$NUM_SHARDS gpus=$GPUS dataset=$DATASET subsets='${SUBSETS}' limit=$LIMIT"

for teacher in "${TEACHER_ARR[@]}"; do
  [[ -z "$teacher" ]] && continue
  name="$(basename "$teacher")"
  outdir="$OUTPUT_BASE/$name"
  mkdir -p "$outdir"
  echo "============================================================"
  echo "[g0_multi] teacher=$name → $outdir  (launching $NUM_SHARDS shards)"
  pids=()
  for ((i = 0; i < NUM_SHARDS; i++)); do
    gpu="${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}"
    CUDA_VISIBLE_DEVICES="$gpu" SHARD_INDEX="$i" NUM_SHARDS="$NUM_SHARDS" SKIP_ANALYZE=1 \
      STUDENT_MODEL="$STUDENT_MODEL" TEACHER_MODEL="$teacher" \
      RUN_NAME="$name" OUTPUT_DIR="$outdir" \
      bash scripts/g0_diag.sh > "$outdir/shard${i}.log" 2>&1 &
    pids+=($!)
    echo "[g0_multi]   shard $i → GPU $gpu (pid ${pids[-1]}, log $outdir/shard${i}.log)"
  done
  fail=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then fail=$((fail + 1)); fi
  done
  [[ $fail -gt 0 ]] && echo "[g0_multi] WARNING: $fail/$NUM_SHARDS shards exited non-zero (see logs); analyzing what completed."
  echo "[g0_multi] teacher=$name shards done; analyzing ..."
  uv run python -m baseline.g0.analyze_g0 --run-dir "$outdir"
  echo "[g0_multi] teacher=$name done → $outdir/report.md"
done

echo "============================================================"
echo "[g0_multi] all teachers done. Compare report.md across: $OUTPUT_BASE/*/"
