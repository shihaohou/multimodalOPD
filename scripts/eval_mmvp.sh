#!/usr/bin/env bash
set -euo pipefail

# MMVP pair-metric evaluation for an OPD-trained (or any) VLM checkpoint.
#
# MMVP = 150 CLIP-blind image pairs -> 300 binary MCQs. Headline metric is PAIR
# accuracy (both questions in a pair correct), which cannot be gamed by a language
# prior -> a clean probe for whether unfreezing the ViT during OPD improved or
# degraded general visual perception. Deterministic MCQ grading, NO LLM judge
# (no API key needed). Full-FT OPD writes a full checkpoint, so MODEL_PATH points
# straight at the run dir (no LoRA merge needed).
#
# Required: MODEL_PATH.  Usage: MODEL_PATH=runs/<run> bash scripts/eval_mmvp.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT_DIR"

: "${MODEL_PATH:?Set MODEL_PATH to a model dir or HuggingFace model id.}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TRANSFORMERS_NO_TF=1
export TOKENIZERS_PARALLELISM=false
# vLLM v1 forks its EngineCore; force spawn so a CUDA-initialized parent can't break it.
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
# Silence PIL's harmless palette-transparency advisory (it floods logs on chart/PNG
# datasets); .convert("RGB") handles those images fine. Propagates to vLLM's subprocess.
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:Palette images with Transparency:UserWarning}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-eval_outputs/mmvp_${RUN_ID}}"
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"

# HF dataset id, OR a local snapshot dir for offline boxes (HF_HUB_OFFLINE=1):
#   hf download MMVP/MMVP --repo-type dataset --local-dir <dir>; MMVP_REPO=<dir>
MMVP_REPO="${MMVP_REPO:-MMVP/MMVP}"
IMAGE_DIR="${IMAGE_DIR:-}"
PAIR_SIZE="${PAIR_SIZE:-2}"
LIMIT="${LIMIT:-}"
# Answer-format instruction appended to each MCQ (the system prompt already forces
# \boxed{}). Unset -> the script's built-in default (MMVP_PROMPT_SUFFIX). Set
# PROMPT_SUFFIX="" for the bare question only. Sentinel keeps the bash side free of
# embedded-newline quoting.
PROMPT_SUFFIX="${PROMPT_SUFFIX-__DEFAULT__}"
# System-prompt style: think (default, <think> tags) | freecot (no tags) | reason |
# none, or a raw string. Match how the checkpoint was trained (OPD_PROMPT_STYLE).
OPD_PROMPT_STYLE="${OPD_PROMPT_STYLE:-think}"

# Greedy single-sample by default (the canonical MMVP setting). If you set
# PASS_K>1 for a robustness read, also raise GEN_TEMPERATURE (>0) or every sample
# is identical.
PASS_K="${PASS_K:-1}"
BATCH_SIZE="${BATCH_SIZE:-0}"   # 0 = feed all prompts to vLLM at once (fastest)
MAX_TOKENS="${MAX_TOKENS:-2048}"
GEN_TEMPERATURE="${GEN_TEMPERATURE:-0.0}"
GEN_TOP_P="${GEN_TOP_P:-1.0}"
GEN_TOP_K="${GEN_TOP_K:-0}"
GEN_SEED="${GEN_SEED:-42}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-}"
VLLM_LIMIT_IMAGES="${VLLM_LIMIT_IMAGES:-2}"
DTYPE="${DTYPE:-auto}"
TOKENIZER_MODE="${TOKENIZER_MODE:-auto}"   # auto = fast tokenizer (quicker preprocessing); slow = fallback

CMD=(
  uv run python baseline/eval/run_mmvp_eval.py
  --model-path "$MODEL_PATH"
  --model-name "$MODEL_NAME"
  --output-dir "$OUTPUT_DIR"
  --mmvp-repo "$MMVP_REPO"
  --pair-size "$PAIR_SIZE"
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
  --tokenizer-mode "$TOKENIZER_MODE"
  --system-prompt "$OPD_PROMPT_STYLE"
)

if [[ "$PROMPT_SUFFIX" != "__DEFAULT__" ]]; then
  CMD+=(--prompt-suffix "$PROMPT_SUFFIX")
fi
if [[ -n "$IMAGE_DIR" ]]; then
  CMD+=(--image-dir "$IMAGE_DIR")
fi
if [[ -n "$LIMIT" ]]; then
  CMD+=(--limit "$LIMIT")
fi
if [[ -n "$VLLM_MAX_MODEL_LEN" ]]; then
  CMD+=(--max-model-len "$VLLM_MAX_MODEL_LEN")
fi

"${CMD[@]}"
