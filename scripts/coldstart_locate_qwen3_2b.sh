#!/usr/bin/env bash
set -euo pipefail

# Locate-Once cold-start (Option beta): teach the student the locate format BEFORE RL.
# Qwen3-VL-2B-Instruct ignores the locate prompt zero-shot (no <think>/<box>), so the
# RL term never fires (box_coverage=0). This runs two phases:
#   Phase 1 (1 GPU, vLLM):  baseline.locate.coldstart_build — self-distill traces:
#     sample the student's reasoning, keep answer-correct ones, inject the GT box +
#     <think>/<box> scaffold -> a small SFT dataset (save_to_disk).
#   Phase 2 (N GPU, SFT):   baseline.locate.coldstart_sft — supervised fine-tune the
#     student on those traces (vanilla CE; prompt masked). Output ckpt -> the RL run.
#
# Then run RL+OPD from the cold-started ckpt:
#   MODEL_NAME_OR_PATH=$SFT_OUTPUT_DIR ... bash scripts/train_opd_locate_qwen3_2b.sh
#
# Required:
#   DATASET_NAME (bbox-carrying; Visual-CoT / saliency-r1-8k), ANSWER_FIELD.
# Example:
#   export M=/path/to/models D=/path/to/datasets
#   MODEL_NAME_OR_PATH=$M/Qwen3-VL-2B-Instruct \
#   DATASET_NAME=$D/Visual-CoT ANSWER_FIELD=answer \
#   bash scripts/coldstart_locate_qwen3_2b.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT_DIR"

export TRANSFORMERS_NO_TF=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost,0.0.0.0"
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost,0.0.0.0"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-VL-2B-Instruct}"
GEN_MODEL="${GEN_MODEL:-$MODEL_NAME_OR_PATH}"   # generator for traces (default = student)
DATASET_NAME="${DATASET_NAME:-peterant330/saliency-r1-8k}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
ANSWER_FIELD="${ANSWER_FIELD:-solution}"
BBOX_FIELD="${BBOX_FIELD:-bbox}"

TRACES_DIR="${TRACES_DIR:-runs/coldstart_locate_traces}"
MODEL_TAG="$(basename "${MODEL_NAME_OR_PATH%/}")"
SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-runs/${MODEL_TAG}_locate_coldstart}"

# --- Phase 1: build traces (1 GPU vLLM) -------------------------------------
SKIP_BUILD="${SKIP_BUILD:-false}"
COLDSTART_GEN_GPU="${COLDSTART_GEN_GPU:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-4000}"
NUM_SAMPLES="${NUM_SAMPLES:-4}"
GEN_TEMPERATURE="${GEN_TEMPERATURE:-0.8}"
GEN_MAX_TOKENS="${GEN_MAX_TOKENS:-1024}"
MAX_REASONING_CHARS="${MAX_REASONING_CHARS:-1500}"
KEEP_INCORRECT="${KEEP_INCORRECT:-false}"
# inject: generate reasoning, bolt <box>[GT]</box> onto the head (GEN_HINT=true grounds
#   the reasoning first). natural: the teacher (set GEN_MODEL=<teacher>) is shown the GT box
#   and writes the WHOLE locate trace itself (box woven in), used verbatim — matches the
#   teacher's own pattern (recommended; Rethinking-OPD: OPD needs compatible patterns).
TRACE_MODE="${TRACE_MODE:-inject}"
GEN_HINT="${GEN_HINT:-false}"
GEN_GPU_MEM_UTIL="${GEN_GPU_MEM_UTIL:-0.9}"
GEN_MAX_MODEL_LEN="${GEN_MAX_MODEL_LEN:-}"

if [[ "$SKIP_BUILD" != "true" ]]; then
  KEEP_INCORRECT_ARG=()
  [[ "$KEEP_INCORRECT" == "true" ]] && KEEP_INCORRECT_ARG=(--keep_incorrect)
  GEN_HINT_ARG=()
  [[ "$GEN_HINT" == "true" ]] && GEN_HINT_ARG=(--gen_hint)
  GEN_MAX_MODEL_LEN_ARG=()
  [[ -n "$GEN_MAX_MODEL_LEN" ]] && GEN_MAX_MODEL_LEN_ARG=(--max_model_len "$GEN_MAX_MODEL_LEN")
  echo "[coldstart] Phase 1: building traces -> $TRACES_DIR (GPU $COLDSTART_GEN_GPU)"
  CUDA_VISIBLE_DEVICES="$COLDSTART_GEN_GPU" uv run python -m baseline.locate.coldstart_build \
    --model_path "$MODEL_NAME_OR_PATH" \
    --gen_model "$GEN_MODEL" \
    --dataset_name "$DATASET_NAME" \
    --dataset_split "$DATASET_SPLIT" \
    --answer_field "$ANSWER_FIELD" \
    --bbox_field "$BBOX_FIELD" \
    --output_dir "$TRACES_DIR" \
    --trace_mode "$TRACE_MODE" \
    --max_samples "$MAX_SAMPLES" \
    --num_samples "$NUM_SAMPLES" \
    --temperature "$GEN_TEMPERATURE" \
    --max_tokens "$GEN_MAX_TOKENS" \
    --max_reasoning_chars "$MAX_REASONING_CHARS" \
    --gpu_memory_utilization "$GEN_GPU_MEM_UTIL" \
    "${GEN_MAX_MODEL_LEN_ARG[@]}" \
    "${GEN_HINT_ARG[@]}" \
    "${KEEP_INCORRECT_ARG[@]}"
else
  echo "[coldstart] Phase 1 skipped (SKIP_BUILD=true); reusing $TRACES_DIR"
fi

# --- Phase 2: SFT (N GPU, DeepSpeed) ----------------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate_zero2_gpu_8.yaml}"
FINETUNING_MODE="${FINETUNING_MODE:-full}"
FREEZE_VISION_TOWER="${FREEZE_VISION_TOWER:-false}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
# Must EQUAL the yaml's gradient_accumulation_steps (ga=8 in accelerate_zero2_gpu_8.yaml).
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-1024}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
SAVE_STEPS="${SAVE_STEPS:-200}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
REPORT_TO="${REPORT_TO:-none}"

GRADIENT_CHECKPOINTING_ARGS=()
[[ "$GRADIENT_CHECKPOINTING" == "true" ]] && GRADIENT_CHECKPOINTING_ARGS=(--gradient_checkpointing)

echo "[coldstart] Phase 2: SFT $MODEL_NAME_OR_PATH on $TRACES_DIR -> $SFT_OUTPUT_DIR"
uv run accelerate launch \
  --config_file "$ACCELERATE_CONFIG" \
  --num_processes "$NUM_PROCESSES" \
  --main_process_port "${MAIN_PROCESS_PORT:-13381}" \
  baseline/locate/coldstart_sft.py \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --dataset_dir "$TRACES_DIR" \
  --finetuning_mode "$FINETUNING_MODE" \
  --freeze_vision_tower "$FREEZE_VISION_TOWER" \
  --output_dir "$SFT_OUTPUT_DIR" \
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --learning_rate "$LEARNING_RATE" \
  --weight_decay "$WEIGHT_DECAY" \
  --lr_scheduler_type "$LR_SCHEDULER_TYPE" \
  --warmup_ratio "$WARMUP_RATIO" \
  --max_grad_norm "$MAX_GRAD_NORM" \
  --bf16 \
  --max_prompt_length "$MAX_PROMPT_LENGTH" \
  --max_target_length "$MAX_TARGET_LENGTH" \
  "${GRADIENT_CHECKPOINTING_ARGS[@]}" \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit "$SAVE_TOTAL_LIMIT" \
  --logging_steps "$LOGGING_STEPS" \
  --report_to "$REPORT_TO"

echo "[coldstart] DONE. Cold-started student: $SFT_OUTPUT_DIR"
echo "[coldstart] Next: MODEL_NAME_OR_PATH=$SFT_OUTPUT_DIR TEACHER_MODEL=... DATASET_NAME=$DATASET_NAME ANSWER_FIELD=$ANSWER_FIELD bash scripts/train_opd_locate_qwen3_2b.sh"
