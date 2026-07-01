#!/usr/bin/env bash
set -euo pipefail

# Evidence Anchor OPD: vanilla on-policy KL plus an anchor hidden-state alignment
# term. Student prompt: image + question + <EVID>. Teacher prompt: image +
# question + hidden GT-box hint + <EVID>. The teacher scores the SAME student
# rollout; the anchor loss aligns the prompt anchor hidden states.
#
# This is intentionally modeled after scripts/train_opd_hint_qwen3_2b.sh because
# the method's teacher branch is the same hidden-hint prompt.
#
# Required:
#   DATASET_NAME   HF id or local path with problem/image/answer and a bbox column.
#   TEACHER_MODEL  Frozen stronger same-family VLM teacher (local HF only).
#
# Example matching the current Qwen3 line:
#   cd /home/web_server/antispam/project/houshihao/multimodalOPD && git pull
#   export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NUM_PROCESSES=8 WANDB_MODE=online HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
#   export M=/home/web_server/antispam/project/houshihao/models D=/home/web_server/antispam/project/houshihao/datasets
#   unset MAX_STEPS MAX_TRAIN_SAMPLES OUTPUT_DIR
#   PER_DEVICE_TRAIN_BATCH_SIZE=8 GRADIENT_ACCUMULATION_STEPS=8 FREEZE_VISION_TOWER=false \
#   MODEL_NAME_OR_PATH=$M/Qwen3-VL-2B-Instruct TEACHER_MODEL=$M/CapCurriculum-8B \
#   DATASET_NAME=$D/Visual-CoT ANSWER_FIELD=answer \
#   RUN_CONFIG=opd_anchor_qwen3_CapCurriculum-8B-to-2B_Visual-CoT \
#   bash scripts/train_opd_anchor.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT_DIR"

: "${DATASET_NAME:?Set DATASET_NAME to the training dataset id/path.}"
: "${TEACHER_MODEL:?Set TEACHER_MODEL to a frozen stronger same-family VLM checkpoint.}"

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
OUTPUT_DIR="${OUTPUT_DIR:-}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-VL-2B-Instruct}"
TEACHER_TORCH_DTYPE="${TEACHER_TORCH_DTYPE:-bfloat16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
TEACHER_ATTN_IMPLEMENTATION="${TEACHER_ATTN_IMPLEMENTATION:-flash_attention_2}"
FINETUNING_MODE="${FINETUNING_MODE:-full}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
FILTER_TINY_IMAGES="${FILTER_TINY_IMAGES:-false}"
MIN_IMAGE_SIZE="${MIN_IMAGE_SIZE:-28}"
MAX_STEPS="${MAX_STEPS:-}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-constant}"
WARMUP_RATIO="${WARMUP_RATIO:-0.0}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
FREEZE_VISION_TOWER="${FREEZE_VISION_TOWER:-false}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-16384}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-2048}"
ANSWER_FIELD="${ANSWER_FIELD:-answer}"
OPD_PROMPT_SUFFIX="${OPD_PROMPT_SUFFIX:-}"
OPD_PROMPT_STYLE="${OPD_PROMPT_STYLE:-think}"

# --- hidden-hint teacher knobs ------------------------------------------------
TEACHER_PRIVILEGE_MODE="${TEACHER_PRIVILEGE_MODE:-hint}"  # hint | crop
BBOX_FIELD="${BBOX_FIELD:-bbox}"
FILTER_NO_BBOX="${FILTER_NO_BBOX:-true}"
HINT_COORD_DECIMALS="${HINT_COORD_DECIMALS:-2}"
HINT_TEMPLATE="${HINT_TEMPLATE:-}"
CROP_PADDING="${CROP_PADDING:-0.0}"

# --- anchor knobs -------------------------------------------------------------
LAMBDA_ANCHOR="${LAMBDA_ANCHOR:-1.0}"
ANCHOR_TOKEN="${ANCHOR_TOKEN:-<EVID>}"
ANCHOR_NUM_TOKENS="${ANCHOR_NUM_TOKENS:-1}"
ANCHOR_INDEXED_TOKENS="${ANCHOR_INDEXED_TOKENS:-true}"
ANCHOR_ANSWER_CUE="${ANCHOR_ANSWER_CUE:-Now answer the question.}"
ANCHOR_PROJECTION_DIM="${ANCHOR_PROJECTION_DIM:-1024}"
ANCHOR_PROJECTOR_BIAS="${ANCHOR_PROJECTOR_BIAS:-false}"
ANCHOR_TRAIN_TEACHER_PROJECTOR="${ANCHOR_TRAIN_TEACHER_PROJECTOR:-false}"

# --- OPD knobs ----------------------------------------------------------------
GENERATION_TEMPERATURE="${GENERATION_TEMPERATURE:-1.0}"
GENERATION_TOP_P="${GENERATION_TOP_P:-1.0}"
GENERATION_TOP_K="${GENERATION_TOP_K:-0}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-1.0}"
LAMBDA_OPD="${LAMBDA_OPD:-1.0}"
OPD_LOSS_MODE="${OPD_LOSS_MODE:-topk_kl}"
OPD_KL_DIRECTION="${OPD_KL_DIRECTION:-reverse}"
OPD_TOP_K="${OPD_TOP_K:-100}"
TOKEN_LOSS_CLIP="${TOKEN_LOSS_CLIP:-0.0}"
PRESENCE_PENALTY="${PRESENCE_PENALTY:-0.0}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
MIN_P="${MIN_P:-0.0}"

USE_VLLM="${USE_VLLM:-true}"
VLLM_MODE="${VLLM_MODE:-colocate}"
# Anchor needs output_hidden_states on the student/teacher scoring forwards, so
# keep a little more headroom than vanilla OPD's default pool.
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.2}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
VLLM_SYNC_FREQUENCY="${VLLM_SYNC_FREQUENCY:-1}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH))}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-$((PER_DEVICE_TRAIN_BATCH_SIZE * VLLM_TENSOR_PARALLEL_SIZE))}"
VLLM_DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-false}"
COMPLETION_LOG_STEPS="${COMPLETION_LOG_STEPS:-5}"
COMPLETION_LOG_MAX_SAMPLES="${COMPLETION_LOG_MAX_SAMPLES:-16}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
SAVE_STEPS="${SAVE_STEPS:-5}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-100}"
SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-true}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-4}"
DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-true}"
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"
REPORT_TO="${REPORT_TO:-wandb}"
ACCELERATE_CMD="${ACCELERATE_CMD:-uv run accelerate}"
read -r -a ACCELERATE <<< "$ACCELERATE_CMD"

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

HINT_TEMPLATE_ARGS=()
if [[ -n "$HINT_TEMPLATE" ]]; then
  HINT_TEMPLATE_ARGS=(--hint_template "$HINT_TEMPLATE")
fi

DATASET_TAG="$(basename "${DATASET_NAME%/}")"
DATASET_TAG="${DATASET_TAG//[^A-Za-z0-9._-]/_}"
RUN_CONFIG="${RUN_CONFIG:-opd_anchor_${DATASET_TAG}}_${RUN_ID}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/${RUN_CONFIG}}"

echo "[opd-anchor] student=$MODEL_NAME_OR_PATH"
echo "[opd-anchor] teacher=$TEACHER_MODEL  (hidden-hint, frozen)"
echo "[opd-anchor] dataset=$DATASET_NAME  bbox_field=$BBOX_FIELD  lambda_anchor=$LAMBDA_ANCHOR  anchor_token=$ANCHOR_TOKEN x$ANCHOR_NUM_TOKENS"

"${ACCELERATE[@]}" launch \
  --config_file "$ACCELERATE_CONFIG" \
  --num_processes "$NUM_PROCESSES" \
  --main_process_port "${MAIN_PROCESS_PORT:-13393}" \
  baseline/train_opd_anchor.py \
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
  --teacher_privilege_mode "$TEACHER_PRIVILEGE_MODE" \
  --bbox_field "$BBOX_FIELD" \
  --filter_no_bbox "$FILTER_NO_BBOX" \
  --hint_coord_decimals "$HINT_COORD_DECIMALS" \
  --crop_padding "$CROP_PADDING" \
  "${HINT_TEMPLATE_ARGS[@]}" \
  --lambda_anchor "$LAMBDA_ANCHOR" \
  --anchor_token "$ANCHOR_TOKEN" \
  --num_anchor_tokens "$ANCHOR_NUM_TOKENS" \
  --anchor_indexed_tokens "$ANCHOR_INDEXED_TOKENS" \
  --anchor_answer_cue "$ANCHOR_ANSWER_CUE" \
  --anchor_projection_dim "$ANCHOR_PROJECTION_DIM" \
  --anchor_projector_bias "$ANCHOR_PROJECTOR_BIAS" \
  --anchor_train_teacher_projector "$ANCHOR_TRAIN_TEACHER_PROJECTOR" \
  --opd_prompt_suffix "$OPD_PROMPT_SUFFIX" \
  --opd_system_prompt "$OPD_PROMPT_STYLE" \
  --output_dir "$OUTPUT_DIR" \
  --run_name "opd_anchor_${RUN_ID}" \
  --run_config "$RUN_CONFIG" \
  "${LIMIT_ARGS[@]}" \
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --learning_rate "$LEARNING_RATE" \
  --weight_decay "$WEIGHT_DECAY" \
  --lr_scheduler_type "$LR_SCHEDULER_TYPE" \
  --warmup_ratio "$WARMUP_RATIO" \
  --max_grad_norm "$MAX_GRAD_NORM" \
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
  --presence_penalty "$PRESENCE_PENALTY" \
  --repetition_penalty "$REPETITION_PENALTY" \
  --min_p "$MIN_P" \
  --use_vllm "$USE_VLLM" \
  --vllm_mode "$VLLM_MODE" \
  --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
  --vllm_tensor_parallel_size "$VLLM_TENSOR_PARALLEL_SIZE" \
  --vllm_sync_frequency "$VLLM_SYNC_FREQUENCY" \
  --vllm_max_model_len "$VLLM_MAX_MODEL_LEN" \
  --vllm_max_num_seqs "$VLLM_MAX_NUM_SEQS" \
  --vllm_disable_custom_all_reduce "$VLLM_DISABLE_CUSTOM_ALL_REDUCE" \
  --completion_log_steps "$COMPLETION_LOG_STEPS" \
  --completion_log_max_samples "$COMPLETION_LOG_MAX_SAMPLES" \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit "$SAVE_TOTAL_LIMIT" \
  --save_only_model "$SAVE_ONLY_MODEL" \
  --logging_steps "$LOGGING_STEPS" \
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
  --dataloader_prefetch_factor "$DATALOADER_PREFETCH_FACTOR" \
  --dataloader_persistent_workers "$DATALOADER_PERSISTENT_WORKERS" \
  --report_to "$REPORT_TO" \
  --lora_r "$LORA_R" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_dropout "$LORA_DROPOUT" \
  --lora_target_modules "$LORA_TARGET_MODULES"
