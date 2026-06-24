#!/usr/bin/env bash
set -euo pipefail

# Stage 0 evidence-reliance probe for ONE model (the "命门" go/no-go experiment).
# No training. Run once per model (teacher / student-before / candidates), then
# aggregate with baseline/probe/analyze_stage0.py.
#
# Required: MODEL_PATH (a local dir or HF id).
# Default grader is rule-based (no API). For DocVQA free-form you may prefer
# GRADER=llm with DEEPSEEK_API_KEY set.
#
# Single-GPU by default (a 7B/8B teacher fits one H800-80G under vLLM); run several
# models concurrently by pinning a different CUDA_VISIBLE_DEVICES per shell.
#
#   MODEL_PATH=$M/MMR1-7B-RL  MODEL_NAME=MMR1-7B-RL  bash scripts/probe_stage0.sh
#   MODEL_PATH=$M/MMR1-3B-SFT MODEL_NAME=MMR1-3B-SFT bash scripts/probe_stage0.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT_DIR"

: "${MODEL_PATH:?Set MODEL_PATH to a model dir or HuggingFace model id.}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TRANSFORMERS_NO_TF=1
export TOKENIZERS_PARALLELISM=false
# Offline by default (assets are pre-downloaded on the box); unset to allow fetch.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"
OUTPUT_DIR="${OUTPUT_DIR:-probe_outputs/stage0}"

DATASET="${DATASET:-peterant330/saliency-r1-8k}"
SPLIT="${SPLIT:-train}"
LIMIT="${LIMIT:-200}"            # per-subset cap
SUBSETS="${SUBSETS:-}"           # e.g. textvqa,textcap,docvqa (default all)
MAX_BBOX_AREA="${MAX_BBOX_AREA:-0.5}"   # drop near-whole-image boxes (random control needs room)
MIN_BBOX_AREA="${MIN_BBOX_AREA:-}"

MASK_FILL="${MASK_FILL:-gray}"   # gray|black|mean|blur
N_RAND="${N_RAND:-3}"
CROP_PADS="${CROP_PADS:-0,0.1,0.2}"
MASK_SEED="${MASK_SEED:-1234}"
SANITY_DUMP="${SANITY_DUMP:-8}"
NUM_SHARDS="${NUM_SHARDS:-1}"        # data-parallel: split samples across N GPUs
SHARD_INDEX="${SHARD_INDEX:-0}"

PROMPT_SUFFIX="${PROMPT_SUFFIX:-}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"          # override the system prompt (else OPD default)
NO_SYSTEM_PROMPT="${NO_SYSTEM_PROMPT:-}"    # set true/1 to send no system prompt (native format)
TEMPERATURE="${TEMPERATURE:-0.0}"   # 0 => greedy (clean, reproducible probe)
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"
PASS_K="${PASS_K:-1}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
GEN_SEED="${GEN_SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
# If a big model still OOMs after the caps below, drop util to 0.80.
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"   # probe needs <<this; caps KV reservation (Qwen3-VL ctx is huge)
VLLM_LIMIT_IMAGES="${VLLM_LIMIT_IMAGES:-1}"          # exactly 1 image/prompt -> less vision-profiling memory
DTYPE="${DTYPE:-auto}"

GRADER="${GRADER:-rule}"            # rule (no API) | llm
JUDGE_MODEL="${JUDGE_MODEL:-deepseek-v4-flash}"
JUDGE_API_URL="${JUDGE_API_URL:-https://api.deepseek.com}"
JUDGE_KEY_ENV="${JUDGE_KEY_ENV:-DEEPSEEK_API_KEY}"
JUDGE_WORKERS="${JUDGE_WORKERS:-32}"

CMD=(
  uv run python baseline/probe/run_stage0.py
  --model-path "$MODEL_PATH"
  --model-name "$MODEL_NAME"
  --output-dir "$OUTPUT_DIR"
  --dataset "$DATASET"
  --split "$SPLIT"
  --limit "$LIMIT"
  --max-bbox-area "$MAX_BBOX_AREA"
  --num-shards "$NUM_SHARDS"
  --shard-index "$SHARD_INDEX"
  --mask-fill "$MASK_FILL"
  --n-rand "$N_RAND"
  --crop-pads "$CROP_PADS"
  --mask-seed "$MASK_SEED"
  --sanity-dump "$SANITY_DUMP"
  --prompt-suffix "$PROMPT_SUFFIX"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --top-k "$TOP_K"
  --pass-k "$PASS_K"
  --max-tokens "$MAX_TOKENS"
  --seed "$GEN_SEED"
  --batch-size "$BATCH_SIZE"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"
  --limit-images "$VLLM_LIMIT_IMAGES"
  --dtype "$DTYPE"
  --grader "$GRADER"
  --judge-model "$JUDGE_MODEL"
  --judge-api-url "$JUDGE_API_URL"
  --judge-key-env "$JUDGE_KEY_ENV"
  --judge-workers "$JUDGE_WORKERS"
)
if [[ -n "$SUBSETS" ]]; then CMD+=(--subsets "$SUBSETS"); fi
if [[ -n "$SYSTEM_PROMPT" ]]; then CMD+=(--system-prompt "$SYSTEM_PROMPT"); fi
if [[ "$NO_SYSTEM_PROMPT" == "true" || "$NO_SYSTEM_PROMPT" == "1" ]]; then CMD+=(--no-system-prompt); fi
if [[ -n "$MIN_BBOX_AREA" ]]; then CMD+=(--min-bbox-area "$MIN_BBOX_AREA"); fi
if [[ -n "$VLLM_MAX_MODEL_LEN" ]]; then CMD+=(--max-model-len "$VLLM_MAX_MODEL_LEN"); fi

echo "[probe_stage0] model=$MODEL_NAME  gpu=$CUDA_VISIBLE_DEVICES  out=$OUTPUT_DIR/$MODEL_NAME"
"${CMD[@]}"
