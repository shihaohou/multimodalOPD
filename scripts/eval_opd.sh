#!/usr/bin/env bash
set -euo pipefail

# General multi-benchmark evaluation for an OPD-trained (or any) VLM checkpoint.
# Uses the dataset's own prompt (no ViGOS scaffolding). Full-FT OPD writes a full
# checkpoint, so MODEL_PATH points straight at the run dir (no LoRA merge needed).
#
# Required: MODEL_PATH.  Judge needs DEEPSEEK_API_KEY (or set SKIP_JUDGE=true).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT_DIR"

: "${MODEL_PATH:?Set MODEL_PATH to a model dir or HuggingFace model id.}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TRANSFORMERS_NO_TF=1
export TOKENIZERS_PARALLELISM=false
# vLLM v1 forks its EngineCore; force spawn so a CUDA-initialized parent can't break it.
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-eval_outputs/opd_${RUN_ID}}"
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"
EVAL_DATASETS="${EVAL_DATASETS:-zli12321/mm-vet,zli12321/mmmu_pro_10options,zli12321/mmmu-pro-vision,zli12321/MMMU,zli12321/MMSI,zli12321/mathverse,zli12321/mathvista,zli12321/realWorldQA}"
EVAL_BENCHMARKS="${EVAL_BENCHMARKS:-}"
DEFAULT_SPLIT="${DEFAULT_SPLIT:-test}"
LIMIT="${LIMIT:-}"
# Format instruction lives in the unified system prompt (baseline/eval/opd_eval_prompt
# OPD_SYSTEM_PROMPT); the user turn is just the question, so no suffix by default.
PROMPT_SUFFIX="${PROMPT_SUFFIX:-}"
PASS_K="${PASS_K:-5}"
BATCH_SIZE="${BATCH_SIZE:-0}"   # 0 = feed all prompts to vLLM at once (fastest)
MAX_TOKENS="${MAX_TOKENS:-4096}"
GEN_TEMPERATURE="${GEN_TEMPERATURE:-1.0}"
GEN_TOP_P="${GEN_TOP_P:-0.9}"
GEN_TOP_K="${GEN_TOP_K:-50}"
GEN_SEED="${GEN_SEED:-42}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-}"
VLLM_LIMIT_IMAGES="${VLLM_LIMIT_IMAGES:-16}"
DTYPE="${DTYPE:-auto}"
# Grading: llm = LLM judge (default, same as ViGOS); rule = mathruler + option/exact
# match (no API, deterministic/reproducible).
GRADER="${GRADER:-llm}"
SKIP_JUDGE="${SKIP_JUDGE:-false}"
JUDGE_MODEL="${JUDGE_MODEL:-deepseek-v4-flash}"
JUDGE_API_URL="${JUDGE_API_URL:-https://api.deepseek.com}"
JUDGE_KEY_ENV="${JUDGE_KEY_ENV:-DEEPSEEK_API_KEY}"
JUDGE_WORKERS="${JUDGE_WORKERS:-64}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-4096}"
JUDGE_TIMEOUT="${JUDGE_TIMEOUT:-120}"
JUDGE_RETRIES="${JUDGE_RETRIES:-2}"
# Optional JSON merged into each judge request (OpenAI extra_body). For a Qwen3
# thinking model served by vLLM, disable thinking so the judge returns its JSON
# verdict (not buried in reasoning):
#   JUDGE_EXTRA_BODY='{"chat_template_kwargs": {"enable_thinking": false}}'
JUDGE_EXTRA_BODY="${JUDGE_EXTRA_BODY:-}"

CMD=(
  uv run python baseline/eval/run_opd_eval.py
  --model-path "$MODEL_PATH"
  --model-name "$MODEL_NAME"
  --output-dir "$OUTPUT_DIR"
  --datasets "$EVAL_DATASETS"
  --benchmarks "$EVAL_BENCHMARKS"
  --default-split "$DEFAULT_SPLIT"
  --grader "$GRADER"
  --prompt-suffix "$PROMPT_SUFFIX"
  --pass-k "$PASS_K"
  --batch-size "$BATCH_SIZE"
  --max-tokens "$MAX_TOKENS"
  --temperature "$GEN_TEMPERATURE"
  --top-p "$GEN_TOP_P"
  --top-k "$GEN_TOP_K"
  --seed "$GEN_SEED"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"
  --limit-images "$VLLM_LIMIT_IMAGES"
  --dtype "$DTYPE"
  --judge-model "$JUDGE_MODEL"
  --judge-api-url "$JUDGE_API_URL"
  --judge-key-env "$JUDGE_KEY_ENV"
  --judge-workers "$JUDGE_WORKERS"
  --judge-max-tokens "$JUDGE_MAX_TOKENS"
  --judge-timeout "$JUDGE_TIMEOUT"
  --judge-retries "$JUDGE_RETRIES"
  --judge-extra-body "$JUDGE_EXTRA_BODY"
)

if [[ -n "$LIMIT" ]]; then
  CMD+=(--limit "$LIMIT")
fi
if [[ -n "$VLLM_MAX_MODEL_LEN" ]]; then
  CMD+=(--max-model-len "$VLLM_MAX_MODEL_LEN")
fi
if [[ "$SKIP_JUDGE" == "true" ]]; then
  CMD+=(--skip-judge)
fi

"${CMD[@]}"
