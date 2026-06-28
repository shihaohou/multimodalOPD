#!/usr/bin/env bash
set -euo pipefail

# V*Bench (V*) visual-search MCQ evaluation for an OPD-trained (or any) VLM.
#
# V*Bench = 191 high-resolution images -> 191 multiple-choice questions in two
# categories (direct_attributes 115, relative_position 76). The discriminative
# detail is a tiny fraction of a very large image, so a model that does not
# actually *search* the image cannot answer from a global glance -> a clean probe
# of fine-grained visual perception (the property OPD's ViT-unfreezing targets).
# Deterministic MCQ grading, NO LLM judge (no API key needed). Full-FT OPD writes
# a full checkpoint, so MODEL_PATH points straight at the run dir (no LoRA merge).
#
# Required: MODEL_PATH.  Usage: MODEL_PATH=runs/<run> bash scripts/eval_vstar.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT_DIR"

: "${MODEL_PATH:?Set MODEL_PATH to a model dir or HuggingFace model id.}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TRANSFORMERS_NO_TF=1
export TOKENIZERS_PARALLELISM=false
# vLLM v1 forks its EngineCore; force spawn so a CUDA-initialized parent can't break it.
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
# Silence PIL's harmless palette-transparency advisory; .convert("RGB") handles those.
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:Palette images with Transparency:UserWarning}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-eval_outputs/vstar_${RUN_ID}}"
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"

# HF dataset id, OR a local snapshot dir for offline boxes (HF_HUB_OFFLINE=1):
#   hf download craigwu/vstar_bench --repo-type dataset --local-dir <dir>; VSTAR_REPO=<dir>
# Default: the shared-disk snapshot ($D/VStarBench, same layout/root as eval_opd_multi.sh)
# when present, else fall back to the HF id (auto-download) so this still works on any box.
D="${D:-/home/web_server/antispam/project/houshihao/datasets}"
_VSTAR_DEFAULT="$D/VStarBench"; [[ -d "$_VSTAR_DEFAULT" ]] || _VSTAR_DEFAULT="craigwu/vstar_bench"
VSTAR_REPO="${VSTAR_REPO:-$_VSTAR_DEFAULT}"
QUESTIONS_FILE="${QUESTIONS_FILE:-test_questions.jsonl}"
# Comma-separated category filter (default: all). e.g. CATEGORIES=direct_attributes
CATEGORIES="${CATEGORIES:-}"
LIMIT="${LIMIT:-}"
# Answer-format instruction appended to each MCQ (the system prompt already forces
# \boxed{}). Unset -> the script's built-in default (VSTAR_PROMPT_SUFFIX). Set
# PROMPT_SUFFIX="" for the bare question only. Sentinel keeps the bash side free of
# embedded-newline quoting.
PROMPT_SUFFIX="${PROMPT_SUFFIX-__DEFAULT__}"
# System-prompt style: think (default, <think> tags) | freecot (no tags) | reason |
# none, or a raw string. Match how the checkpoint was trained (OPD_PROMPT_STYLE).
OPD_PROMPT_STYLE="${OPD_PROMPT_STYLE:-think}"

# Greedy single-sample by default (the canonical V*Bench setting). If you set
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
# V*Bench is high-resolution single-image; 1 image per prompt is enough.
VLLM_LIMIT_IMAGES="${VLLM_LIMIT_IMAGES:-1}"
DTYPE="${DTYPE:-auto}"
TOKENIZER_MODE="${TOKENIZER_MODE:-auto}"   # auto = fast tokenizer; slow = fallback

CMD=(
  uv run python baseline/eval/run_vstar_eval.py
  --model-path "$MODEL_PATH"
  --model-name "$MODEL_NAME"
  --output-dir "$OUTPUT_DIR"
  --vstar-repo "$VSTAR_REPO"
  --questions-file "$QUESTIONS_FILE"
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
if [[ -n "$CATEGORIES" ]]; then
  CMD+=(--categories "$CATEGORIES")
fi
if [[ -n "$LIMIT" ]]; then
  CMD+=(--limit "$LIMIT")
fi
if [[ -n "$VLLM_MAX_MODEL_LEN" ]]; then
  CMD+=(--max-model-len "$VLLM_MAX_MODEL_LEN")
fi

"${CMD[@]}"
