#!/usr/bin/env bash
set -euo pipefail

# OPD + differentiable evidence-alignment training for a Qwen2.5-VL-3B student
# against a frozen, stronger same-family teacher. Adds the saliency
# evidence-alignment loss on top of vanilla OPD (see baseline/evidence/).
#
# Required:
#   DATASET_NAME   HuggingFace dataset id (problem/images/answer).
#   TEACHER_MODEL  Frozen teacher checkpoint (local_hf only — evidence needs a
#                  full local teacher forward).
#
# IMPORTANT: run the Step-1 sanity check FIRST to confirm the engine backward
# works and to read peak memory / the teacher-student grid match:
#   uv run python -m baseline.evidence.sanity_check \
#       --student_model "$MODEL_NAME_OR_PATH" --teacher_model "$TEACHER_MODEL"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT_DIR"

: "${DATASET_NAME:?Set DATASET_NAME to the HuggingFace training dataset id.}"
: "${TEACHER_MODEL:?Set TEACHER_MODEL (local_hf teacher checkpoint).}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TRANSFORMERS_NO_TF=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_PROJECT="${WANDB_PROJECT:-MultimodalOPD}"
export WANDB_MODE="${WANDB_MODE:-online}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost,0.0.0.0"
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost,0.0.0.0"

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate_zero2_gpu_8.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/opd_evidence_qwen25_3b_${RUN_ID}}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}"
TEACHER_TORCH_DTYPE="${TEACHER_TORCH_DTYPE:-bfloat16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
TEACHER_ATTN_IMPLEMENTATION="${TEACHER_ATTN_IMPLEMENTATION:-sdpa}"
FINETUNING_MODE="${FINETUNING_MODE:-full}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
FILTER_TINY_IMAGES="${FILTER_TINY_IMAGES:-false}"
MIN_IMAGE_SIZE="${MIN_IMAGE_SIZE:-3}"
MAX_STEPS="${MAX_STEPS:-}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-constant}"
WARMUP_RATIO="${WARMUP_RATIO:-0.0}"
FREEZE_VISION_TOWER="${FREEZE_VISION_TOWER:-false}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-16384}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-2048}"
ANSWER_FIELD="${ANSWER_FIELD:-answer}"
OPD_PROMPT_SUFFIX="${OPD_PROMPT_SUFFIX:-}"
GENERATION_TEMPERATURE="${GENERATION_TEMPERATURE:-1.0}"
GENERATION_TOP_P="${GENERATION_TOP_P:-1.0}"
GENERATION_TOP_K="${GENERATION_TOP_K:-0}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-1.0}"
LAMBDA_OPD="${LAMBDA_OPD:-1.0}"
OPD_LOSS_MODE="${OPD_LOSS_MODE:-full_kl}"
OPD_KL_DIRECTION="${OPD_KL_DIRECTION:-reverse}"
OPD_TOP_K="${OPD_TOP_K:-32}"
TOKEN_LOSS_CLIP="${TOKEN_LOSS_CLIP:-0.0}"

# --- evidence-alignment knobs ------------------------------------------------
LAMBDA_EVIDENCE="${LAMBDA_EVIDENCE:-1.0}"
# Rows of each micro-batch the eager evidence forward runs on. Keep small —
# eager output_attentions over thousands of visual tokens is the OOM point.
EVIDENCE_MAX_SAMPLES="${EVIDENCE_MAX_SAMPLES:-1}"
# Comma list of decoder layers to sum saliency over (empty = all).
EVIDENCE_LAYERS="${EVIDENCE_LAYERS:-}"
EVIDENCE_TOP_RATIO="${EVIDENCE_TOP_RATIO:-0.2}"
EVIDENCE_MIN_TOKENS="${EVIDENCE_MIN_TOKENS:-1}"
EVIDENCE_MAX_TOKENS="${EVIDENCE_MAX_TOKENS:-8}"
EVIDENCE_SIGNED="${EVIDENCE_SIGNED:-true}"
EVIDENCE_KL_DIRECTION="${EVIDENCE_KL_DIRECTION:-forward}"
EVIDENCE_GATE_TEMP="${EVIDENCE_GATE_TEMP:-1.0}"
EVIDENCE_GATE_H0="${EVIDENCE_GATE_H0:-0.9}"
EVIDENCE_GATE_TAU="${EVIDENCE_GATE_TAU:-0.1}"
EVIDENCE_KL_THRESHOLD="${EVIDENCE_KL_THRESHOLD:-0.0}"
EVIDENCE_MASS_THRESHOLD="${EVIDENCE_MASS_THRESHOLD:-0.0}"

USE_VLLM="${USE_VLLM:-true}"
VLLM_MODE="${VLLM_MODE:-colocate}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.25}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
VLLM_SYNC_FREQUENCY="${VLLM_SYNC_FREQUENCY:-1}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH))}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-$((PER_DEVICE_TRAIN_BATCH_SIZE * VLLM_TENSOR_PARALLEL_SIZE))}"
COMPLETION_LOG_STEPS="${COMPLETION_LOG_STEPS:-0}"
COMPLETION_LOG_MAX_SAMPLES="${COMPLETION_LOG_MAX_SAMPLES:-16}"
# Gradient checkpointing can swallow output_attentions on some stacks (the
# trainer then logs a warning and skips evidence). If loss_ev never appears,
# set GRADIENT_CHECKPOINTING=false (costs memory) or lower the batch.
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
SAVE_STEPS="${SAVE_STEPS:-5}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-100}"
SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-true}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"
REPORT_TO="${REPORT_TO:-wandb}"

GRADIENT_CHECKPOINTING_ARGS=()
if [[ "$GRADIENT_CHECKPOINTING" == "true" ]]; then
  GRADIENT_CHECKPOINTING_ARGS=(--gradient_checkpointing)
fi

LIMIT_ARGS=()
if [[ -n "$MAX_TRAIN_SAMPLES" ]]; then
  LIMIT_ARGS+=(--max_train_samples "$MAX_TRAIN_SAMPLES")
fi
if [[ -n "$MAX_STEPS" ]]; then
  LIMIT_ARGS+=(--max_steps "$MAX_STEPS")
fi

EVIDENCE_LAYERS_ARGS=()
if [[ -n "$EVIDENCE_LAYERS" ]]; then
  EVIDENCE_LAYERS_ARGS+=(--evidence_layers "$EVIDENCE_LAYERS")
fi

RUN_CONFIG="${RUN_CONFIG:-opd_ev_qwen25_3b_lev${LAMBDA_EVIDENCE}_top${EVIDENCE_TOP_RATIO}_np${NUM_PROCESSES}}"

uv run accelerate launch \
  --config_file "$ACCELERATE_CONFIG" \
  --num_processes "$NUM_PROCESSES" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --main_process_port "${MAIN_PROCESS_PORT:-13379}" \
  baseline/train_opd_evidence.py \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --finetuning_mode "$FINETUNING_MODE" \
  --attn_implementation "$ATTN_IMPLEMENTATION" \
  --teacher_source local_hf \
  --teacher_model_name_or_path "$TEACHER_MODEL" \
  --teacher_torch_dtype "$TEACHER_TORCH_DTYPE" \
  --teacher_attn_implementation "$TEACHER_ATTN_IMPLEMENTATION" \
  --dataset_name "$DATASET_NAME" \
  --dataset_split "$DATASET_SPLIT" \
  --filter_tiny_images "$FILTER_TINY_IMAGES" \
  --min_image_size "$MIN_IMAGE_SIZE" \
  --answer_field "$ANSWER_FIELD" \
  --opd_prompt_suffix "$OPD_PROMPT_SUFFIX" \
  --output_dir "$OUTPUT_DIR" \
  --run_name "opd_ev_qwen25_3b_${RUN_ID}" \
  --run_config "$RUN_CONFIG" \
  "${LIMIT_ARGS[@]}" \
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --learning_rate "$LEARNING_RATE" \
  --weight_decay "$WEIGHT_DECAY" \
  --lr_scheduler_type "$LR_SCHEDULER_TYPE" \
  --warmup_ratio "$WARMUP_RATIO" \
  --max_grad_norm 0.1 \
  --bf16 \
  --freeze_vision_tower "$FREEZE_VISION_TOWER" \
  "${GRADIENT_CHECKPOINTING_ARGS[@]}" \
  --max_prompt_length "$MAX_PROMPT_LENGTH" \
  --max_completion_length "$MAX_COMPLETION_LENGTH" \
  --generation_temperature "$GENERATION_TEMPERATURE" \
  --generation_top_p "$GENERATION_TOP_P" \
  --generation_top_k "$GENERATION_TOP_K" \
  --distill_temperature "$DISTILL_TEMPERATURE" \
  --lambda_opd "$LAMBDA_OPD" \
  --opd_loss_mode "$OPD_LOSS_MODE" \
  --opd_kl_direction "$OPD_KL_DIRECTION" \
  --opd_top_k "$OPD_TOP_K" \
  --token_loss_clip "$TOKEN_LOSS_CLIP" \
  --lambda_evidence "$LAMBDA_EVIDENCE" \
  --evidence_max_samples "$EVIDENCE_MAX_SAMPLES" \
  "${EVIDENCE_LAYERS_ARGS[@]}" \
  --evidence_top_ratio "$EVIDENCE_TOP_RATIO" \
  --evidence_min_tokens "$EVIDENCE_MIN_TOKENS" \
  --evidence_max_tokens "$EVIDENCE_MAX_TOKENS" \
  --evidence_signed "$EVIDENCE_SIGNED" \
  --evidence_kl_direction "$EVIDENCE_KL_DIRECTION" \
  --evidence_gate_temp "$EVIDENCE_GATE_TEMP" \
  --evidence_gate_h0 "$EVIDENCE_GATE_H0" \
  --evidence_gate_tau "$EVIDENCE_GATE_TAU" \
  --evidence_kl_threshold "$EVIDENCE_KL_THRESHOLD" \
  --evidence_mass_threshold "$EVIDENCE_MASS_THRESHOLD" \
  --use_vllm "$USE_VLLM" \
  --vllm_mode "$VLLM_MODE" \
  --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
  --vllm_tensor_parallel_size "$VLLM_TENSOR_PARALLEL_SIZE" \
  --vllm_sync_frequency "$VLLM_SYNC_FREQUENCY" \
  --vllm_max_model_len "$VLLM_MAX_MODEL_LEN" \
  --vllm_max_num_seqs "$VLLM_MAX_NUM_SEQS" \
  --completion_log_steps "$COMPLETION_LOG_STEPS" \
  --completion_log_max_samples "$COMPLETION_LOG_MAX_SAMPLES" \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit "$SAVE_TOTAL_LIMIT" \
  --save_only_model "$SAVE_ONLY_MODEL" \
  --logging_steps "$LOGGING_STEPS" \
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
  --report_to "$REPORT_TO" \
  --lora_r "$LORA_R" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_dropout "$LORA_DROPOUT" \
  --lora_target_modules "$LORA_TARGET_MODULES"
