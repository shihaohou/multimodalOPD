#!/usr/bin/env bash
set -euo pipefail

# POPE / ChartQA / VQAv2 short-answer evaluation for an OPD-trained (or any) VLM.
#
# Three classic single-image benchmarks, each scored by its OWN canonical official
# metric, deterministically and with NO LLM judge (no API key needed):
#   POPE    -> F1 (+ accuracy/precision/recall/yes-ratio), per category
#   ChartQA -> relaxed accuracy (5% numeric tolerance), human vs augmented
#   VQAv2   -> official VQA soft accuracy (min(1, agreement/3) over 10 answers)
# Run under the unified OPD system prompt; the vLLM engine is loaded once and all
# requested benchmarks are evaluated in turn. Full-FT OPD writes a full checkpoint,
# so MODEL_PATH points straight at the run dir (no LoRA merge needed).
#
# Required: MODEL_PATH.  Usage: MODEL_PATH=runs/<run> bash scripts/eval_vqa.sh
# Sources default to the canonical lmms-lab HF datasets (auto-downloaded/cached);
# point *_REPO at a local snapshot dir for offline boxes. NOTE: VQAv2 validation is
# ~214k questions -> set LIMIT for a quick read, or BENCHMARKS=pope,chartqa to skip.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT_DIR"

: "${MODEL_PATH:?Set MODEL_PATH to a model dir or HuggingFace model id.}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TRANSFORMERS_NO_TF=1
export TOKENIZERS_PARALLELISM=false

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-eval_outputs/vqa_${RUN_ID}}"
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"

BENCHMARKS="${BENCHMARKS:-pope,chartqa,vqav2}"
# Sources: HF dataset id OR a local snapshot dir (see the download commands in the
# README). Pre-fetch with e.g. `hf download lmms-lab/POPE --repo-type dataset
# --local-dir <dir>` then set POPE_REPO=<dir>.
POPE_REPO="${POPE_REPO:-lmms-lab/POPE}"
POPE_SPLIT="${POPE_SPLIT:-test}"
POPE_CATEGORY="${POPE_CATEGORY:-}"        # random|popular|adversarial ('' = all)
CHARTQA_REPO="${CHARTQA_REPO:-lmms-lab/ChartQA}"
CHARTQA_SPLIT="${CHARTQA_SPLIT:-test}"
VQAV2_REPO="${VQAV2_REPO:-lmms-lab/VQAv2}"
VQAV2_SPLIT="${VQAV2_SPLIT:-validation}"
LIMIT="${LIMIT:-}"
# VQAv2-only cap (its val set is ~214k vs a few-k for POPE/ChartQA); overrides
# LIMIT for VQAv2 only, so you can keep POPE/ChartQA full while sampling VQAv2.
VQAV2_LIMIT="${VQAV2_LIMIT:-}"
# Answer-format suffix appended to each question (the system prompt already forces
# \boxed{}). Unset -> the script's per-benchmark default (yes/no for POPE, "single
# word or phrase" for ChartQA/VQAv2). Set PROMPT_SUFFIX="" to force the bare
# question. Sentinel keeps the bash side free of embedded-newline quoting.
PROMPT_SUFFIX="${PROMPT_SUFFIX-__DEFAULT__}"

# Greedy single-sample by default (the canonical setting for all three). If you set
# PASS_K>1 for a robustness read, also raise GEN_TEMPERATURE (>0).
PASS_K="${PASS_K:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
GEN_TEMPERATURE="${GEN_TEMPERATURE:-0.0}"
GEN_TOP_P="${GEN_TOP_P:-1.0}"
GEN_TOP_K="${GEN_TOP_K:-0}"
GEN_SEED="${GEN_SEED:-42}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-}"
VLLM_LIMIT_IMAGES="${VLLM_LIMIT_IMAGES:-4}"
DTYPE="${DTYPE:-auto}"

CMD=(
  uv run python baseline/eval/run_vqa_eval.py
  --model-path "$MODEL_PATH"
  --model-name "$MODEL_NAME"
  --output-dir "$OUTPUT_DIR"
  --benchmarks "$BENCHMARKS"
  --pope-repo "$POPE_REPO"
  --pope-split "$POPE_SPLIT"
  --pope-category "$POPE_CATEGORY"
  --chartqa-repo "$CHARTQA_REPO"
  --chartqa-split "$CHARTQA_SPLIT"
  --vqav2-repo "$VQAV2_REPO"
  --vqav2-split "$VQAV2_SPLIT"
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
)

if [[ "$PROMPT_SUFFIX" != "__DEFAULT__" ]]; then
  CMD+=(--prompt-suffix "$PROMPT_SUFFIX")
fi
if [[ -n "$LIMIT" ]]; then
  CMD+=(--limit "$LIMIT")
fi
if [[ -n "$VQAV2_LIMIT" ]]; then
  CMD+=(--vqav2-limit "$VQAV2_LIMIT")
fi
if [[ -n "$VLLM_MAX_MODEL_LEN" ]]; then
  CMD+=(--max-model-len "$VLLM_MAX_MODEL_LEN")
fi

"${CMD[@]}"
