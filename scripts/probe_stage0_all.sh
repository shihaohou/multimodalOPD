#!/usr/bin/env bash
set -euo pipefail

# Run the Stage 0 evidence-reliance probe over a MATRIX of models, one GPU per
# model, in parallel. Then aggregate with baseline/probe/analyze_stage0.py.
#
# Requires (#models) <= (#GPUS): each GPU runs at most one model at a time. To run
# more models than GPUs, split into waves (call this twice with different MODELS).
#
#   bash scripts/probe_stage0_all.sh
#   # custom matrix / GPUs:
#   GPUS="0 1 2 3" MODELS="A=$M/A B=$M/B" bash scripts/probe_stage0_all.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT_DIR"

M="${M:-/home/web_server/antispam/project/houshihao/models}"
D="${D:-/home/web_server/antispam/project/houshihao/datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-probe_outputs/stage0}"
# Clean tier: specific-token answers, small boxes, floor ~ 0 (see baseline/probe/README.md).
SUBSETS="${SUBSETS:-textvqa,textcap,docvqa,infographicsvqa,gqa,openimages}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"

# name=path entries (whitespace-separated). Override MODELS to change the matrix.
MODELS="${MODELS:-
MMR1-7B-RL=$M/MMR1-7B-RL
Saliency-R1-7B=$M/Saliency-R1-7B
Qwen2.5-VL-7B=$M/Qwen2.5-VL-7B-Instruct
Qwen3-VL-8B=$M/Qwen3-VL-8B-Instruct
MMR1-3B-SFT=$M/MMR1-3B-SFT
Qwen2.5-VL-3B=$M/Qwen2.5-VL-3B-Instruct
Qwen3-VL-2B=$M/Qwen3-VL-2B-Instruct
}"

read -r -a gpu_arr <<< "$GPUS"
names=(); paths=()
for entry in $MODELS; do
  [[ -z "$entry" || "$entry" != *=* ]] && continue
  names+=("${entry%%=*}"); paths+=("${entry#*=}")
done
if (( ${#names[@]} > ${#gpu_arr[@]} )); then
  echo "ERROR: ${#names[@]} models but only ${#gpu_arr[@]} GPUs ($GPUS)." >&2
  echo "       Set GPUS to more devices, or split MODELS into waves." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR/logs"
pids=()
for i in "${!names[@]}"; do
  name="${names[$i]}"; path="${paths[$i]}"; gpu="${gpu_arr[$i]}"
  log="$OUTPUT_DIR/logs/${name}.log"
  echo "[all] launching $name on GPU $gpu -> $log"
  CUDA_VISIBLE_DEVICES="$gpu" DATASET="$D/saliency-r1-8k" SUBSETS="$SUBSETS" \
    OUTPUT_DIR="$OUTPUT_DIR" MODEL_PATH="$path" MODEL_NAME="$name" \
    bash scripts/probe_stage0.sh > "$log" 2>&1 &
  pids+=($!)
done

echo "[all] launched ${#pids[@]} runs; waiting (tail -f $OUTPUT_DIR/logs/<name>.log to watch)..."
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=$(( fail + 1 )); done

echo "[all] done; $fail run(s) failed. Acc_full recap (sanity-check none are ~0):"
grep -h "Acc_full =" "$OUTPUT_DIR"/logs/*.log 2>/dev/null || echo "  (no Acc_full lines -- check logs)"
echo "[all] next: aggregate with baseline/probe/analyze_stage0.py (see README)."
