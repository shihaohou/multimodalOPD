#!/usr/bin/env bash
set -euo pipefail

# Fast lmms-eval-aligned benchmark evaluation.
#
# This keeps the OPD project's vLLM batching path, but uses lmms-eval task
# definitions for benchmark data loading, split selection, benchmark prompt text,
# process_results, and metric aggregation.
#
# Required:
#   MODEL_PATH=/path/to/checkpoint
#
# Common knobs:
#   DATASETS="mathvista mathverse mathvision MMMU MMMU-Pro MMStar HallusionBench POPE ChartQA vstar HRBench4K HRBench8K MME-RealWorld-Lite"
#   PROMPT_MODE=lmms  # lmms | opd
#   LMMS_EVAL_DIR=/Users/houshihao/project/code/lmms-eval-main
#   LMMS_MODEL_NAME=qwen3_vl  # selects model-specific prompt branches in lmms-eval YAML

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT_DIR"

: "${MODEL_PATH:?Set MODEL_PATH to a model dir or HuggingFace model id.}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"
export TRANSFORMERS_NO_TF=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:Palette images with Transparency:UserWarning}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-eval_outputs/lmms_aligned_${RUN_ID}}"
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"

DATASETS="${DATASETS:-standard}"
LMMS_PHASE="${LMMS_PHASE:-${PHASE:-all}}"  # all | generate | judge
PROMPT_MODE="${PROMPT_MODE:-lmms}"  # lmms | opd
LMMS_EVAL_DIR="${LMMS_EVAL_DIR:-/Users/houshihao/project/code/lmms-eval-main}"
LMMS_MODEL_NAME="${LMMS_MODEL_NAME:-qwen3_vl}"
OPD_PROMPT_STYLE="${OPD_PROMPT_STYLE:-think}"
OPD_PROMPT_SUFFIX="${OPD_PROMPT_SUFFIX:-}"

LIMIT="${LIMIT:-}"
BATCH_SIZE="${BATCH_SIZE:-0}"
MAX_TOKENS="${MAX_TOKENS:-}"
GEN_TEMPERATURE="${GEN_TEMPERATURE:-}"
GEN_TOP_P="${GEN_TOP_P:-}"
GEN_TOP_K="${GEN_TOP_K:-}"
GEN_SEED="${GEN_SEED:-42}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-}"
VLLM_LIMIT_IMAGES="${VLLM_LIMIT_IMAGES:-16}"
DTYPE="${DTYPE:-auto}"
TOKENIZER_MODE="${TOKENIZER_MODE:-auto}"

CMD=(
  uv run python baseline/eval/run_lmms_aligned_eval.py
  --model-path "$MODEL_PATH"
  --model-name "$MODEL_NAME"
  --output-dir "$OUTPUT_DIR"
  --benchmarks "$DATASETS"
  --prompt-mode "$PROMPT_MODE"
  --lmms-eval-dir "$LMMS_EVAL_DIR"
  --lmms-model-name "$LMMS_MODEL_NAME"
  --opd-prompt-style "$OPD_PROMPT_STYLE"
  --opd-prompt-suffix "$OPD_PROMPT_SUFFIX"
  --batch-size "$BATCH_SIZE"
  --seed "$GEN_SEED"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"
  --limit-images "$VLLM_LIMIT_IMAGES"
  --dtype "$DTYPE"
  --tokenizer-mode "$TOKENIZER_MODE"
)

[[ -n "$LIMIT" ]] && CMD+=(--limit "$LIMIT")
[[ -n "$MAX_TOKENS" ]] && CMD+=(--max-tokens "$MAX_TOKENS")
[[ -n "$GEN_TEMPERATURE" ]] && CMD+=(--temperature "$GEN_TEMPERATURE")
[[ -n "$GEN_TOP_P" ]] && CMD+=(--top-p "$GEN_TOP_P")
[[ -n "$GEN_TOP_K" ]] && CMD+=(--top-k "$GEN_TOP_K")
[[ -n "$VLLM_MAX_MODEL_LEN" ]] && CMD+=(--max-model-len "$VLLM_MAX_MODEL_LEN")

case "$LMMS_PHASE" in
  all) ;;
  generate) CMD+=(--skip-score) ;;
  judge) CMD+=(--score-only) ;;
  *) echo "ERROR: LMMS_PHASE must be all|generate|judge (got '$LMMS_PHASE')." >&2; exit 1 ;;
esac

"${CMD[@]}"
