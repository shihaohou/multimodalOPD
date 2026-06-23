#!/usr/bin/env bash
set -euo pipefail

# Start the OPD teacher scoring server (vLLM offline engine -> top-k prompt
# logprobs). Run on GPUs SEPARATE from training, then point training at it with
# TEACHER_SOURCE=vllm_server TEACHER_SERVER_URL=http://<host>:<port>.
#
# Required: TEACHER_MODEL.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT_DIR"

: "${TEACHER_MODEL:?Set TEACHER_MODEL to the teacher checkpoint to serve.}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TRANSFORMERS_NO_TF=1
export TOKENIZERS_PARALLELISM=false

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8200}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
LIMIT_IMAGES="${LIMIT_IMAGES:-16}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
DTYPE="${DTYPE:-auto}"
SEED="${SEED:-0}"

CMD=(
  uv run python baseline/serve_teacher.py
  --model "$TEACHER_MODEL"
  --host "$HOST"
  --port "$PORT"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --limit-images "$LIMIT_IMAGES"
  --max-num-seqs "$MAX_NUM_SEQS"
  --dtype "$DTYPE"
  --seed "$SEED"
)
if [[ -n "$MAX_MODEL_LEN" ]]; then
  CMD+=(--max-model-len "$MAX_MODEL_LEN")
fi

"${CMD[@]}"
