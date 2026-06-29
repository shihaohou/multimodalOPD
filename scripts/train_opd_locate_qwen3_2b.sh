#!/usr/bin/env bash
set -euo pipefail

# Locate-Once Grounding (LOG): OPD distillation + an explicit, student-generated
# evidence box trained by RL. The student rolls out from a *locate-once* prompt (opens
# <think> with one <box>[x1,y1,x2,y2]</box>, no crop). The frozen teacher shares that
# SAME structured prompt (LOCATE_TEACHER_MODE=shared, default) so student and teacher
# stay in one distribution — a plain/box-forbidding teacher makes reverse-KL erase the
# <box> (box_present -> 0 ~step 35). OPD reverse-KL covers answer/reasoning with the box
# span (and its decision token) masked; GRPO reinforces the box coords with an IoU reward
# gated by answer correctness. Question (G2): does the explicit box beat the spine on V*Bench?
#
# Required:
#   TEACHER_MODEL  Path/id of the frozen, stronger, SAME-FAMILY VLM teacher.
# Strongly recommended (defaults target the bbox-carrying saliency-r1-8k):
#   DATASET_NAME   A dataset with an evidence-box column (default below).
#   ANSWER_FIELD   The answer column (saliency-r1-8k uses 'solution', not 'answer').
#
# Example (Qwen3-VL line; the v2 doc's models are MMR1-3B-SFT student / MMR1-7B-RL
# teacher — set MODEL_NAME_OR_PATH / TEACHER_MODEL to use them):
#   export M=/path/to/models D=/path/to/datasets
#   PER_DEVICE_TRAIN_BATCH_SIZE=1 GROUP_SIZE=8 GRADIENT_ACCUMULATION_STEPS=8 \
#   MODEL_NAME_OR_PATH=$M/Qwen3-VL-2B-Instruct TEACHER_MODEL=$M/Vero-Qwen3I-8B \
#   DATASET_NAME=$D/saliency-r1-8k ANSWER_FIELD=solution \
#   bash scripts/train_opd_locate_qwen3_2b.sh
#
# BATCHING: per_device_train_batch_size counts PROMPTS; the collator expands each into
# GROUP_SIZE rollouts (the GRPO group). Effective rollouts/step = per_device * GROUP_SIZE
# * grad_accum * world. Keep per_device small (1-2) and let GROUP_SIZE batch the rollout
# (vLLM samples each replica independently -> diverse group; needs temperature > 0).
#
# A/B vs the spine: this is B2. Compare against B1 = scripts/train_opd_hint_qwen3_2b.sh
# (hidden-hint, box-free) on the SAME data/teacher/schedule. B2 > B1 on V*Bench => the
# explicit box earns its keep; B2 ~= B1 => ship B1 (simpler, already verified).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT_DIR"

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
DATASET_NAME="${DATASET_NAME:-peterant330/saliency-r1-8k}"
TEACHER_TORCH_DTYPE="${TEACHER_TORCH_DTYPE:-bfloat16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
TEACHER_ATTN_IMPLEMENTATION="${TEACHER_ATTN_IMPLEMENTATION:-flash_attention_2}"
FINETUNING_MODE="${FINETUNING_MODE:-full}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
# Off by default: the collator's _safe_rgb_image already pads degenerate/tiny images,
# so the pre-filter is redundant; set true to drop sub-MIN_IMAGE_SIZE images instead.
FILTER_TINY_IMAGES="${FILTER_TINY_IMAGES:-false}"
MIN_IMAGE_SIZE="${MIN_IMAGE_SIZE:-28}"
MAX_STEPS="${MAX_STEPS:-}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
# per_device counts PROMPTS; GROUP_SIZE rollouts per prompt do the rollout batching.
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GROUP_SIZE="${GROUP_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-constant}"
WARMUP_RATIO="${WARMUP_RATIO:-0.0}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
FREEZE_VISION_TOWER="${FREEZE_VISION_TOWER:-false}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-16384}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-2048}"
ANSWER_FIELD="${ANSWER_FIELD:-solution}"
OPD_PROMPT_SUFFIX="${OPD_PROMPT_SUFFIX:-}"
# OPD teacher channel (FIX for the box-death collapse). The student emits a <box>, so the
# teacher must share that distribution or reverse-KL kills the format:
#   shared (default) = teacher uses the SAME structured locate prompt, no coordinate hint;
#   crop             = structured teacher + image cropped to the GT box (privilege, no digits);
#   plain_hint       = OLD plain-think + no-verbalize hint (reproduces the collapse — ablation only).
LOCATE_TEACHER_MODE="${LOCATE_TEACHER_MODE:-shared}"
# Also drop the box-emission decision token (before "<box>") from OPD so RL/SFT own localization.
MASK_BOX_TRANSITION="${MASK_BOX_TRANSITION:-true}"
# Teacher base system-prompt style — only used by LOCATE_TEACHER_MODE=plain_hint. think | freecot | reason | none.
OPD_PROMPT_STYLE="${OPD_PROMPT_STYLE:-think}"
# --- Grounding / box knobs ---------------------------------------------------
BBOX_FIELD="${BBOX_FIELD:-bbox}"
FILTER_NO_BBOX="${FILTER_NO_BBOX:-true}"
HINT_COORD_DECIMALS="${HINT_COORD_DECIMALS:-2}"
HINT_TEMPLATE="${HINT_TEMPLATE:-}"
LOCATE_SYSTEM_PROMPT="${LOCATE_SYSTEM_PROMPT:-}"
# --- Loss weights / RL -------------------------------------------------------
LAMBDA_OPD="${LAMBDA_OPD:-1.0}"
LAMBDA_RL="${LAMBDA_RL:-0.5}"
RL_REWARD="${RL_REWARD:-gated_iou}"          # gated_iou | iou
# Warmup (gated_iou only): reward += RL_UNGATED_WEIGHT * IoU even when answer wrong. 0=off.
RL_UNGATED_WEIGHT="${RL_UNGATED_WEIGHT:-0.0}"
RL_NORMALIZE_ADV="${RL_NORMALIZE_ADV:-true}" # true=(r-mean)/std | false=r-mean (Dr.GRPO)
KL_POSITION_GATE="${KL_POSITION_GATE:-false}"
# --- OPD distillation --------------------------------------------------------
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-1.0}"
OPD_LOSS_MODE="${OPD_LOSS_MODE:-topk_kl}"          # topk_kl | full_kl
OPD_KL_DIRECTION="${OPD_KL_DIRECTION:-reverse}"    # reverse | forward | jsd
OPD_TOP_K="${OPD_TOP_K:-100}"
TOKEN_LOSS_CLIP="${TOKEN_LOSS_CLIP:-0.0}"
# Rollout sampling: temperature > 0 so the GROUP_SIZE replicas diverge (GRPO needs it).
GENERATION_TEMPERATURE="${GENERATION_TEMPERATURE:-1.0}"
GENERATION_TOP_P="${GENERATION_TOP_P:-1.0}"
GENERATION_TOP_K="${GENERATION_TOP_K:-0}"
PRESENCE_PENALTY="${PRESENCE_PENALTY:-0.0}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
MIN_P="${MIN_P:-0.0}"
USE_VLLM="${USE_VLLM:-true}"
VLLM_MODE="${VLLM_MODE:-colocate}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.25}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
VLLM_SYNC_FREQUENCY="${VLLM_SYNC_FREQUENCY:-1}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH))}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-$((PER_DEVICE_TRAIN_BATCH_SIZE * GROUP_SIZE * VLLM_TENSOR_PARALLEL_SIZE))}"
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

OPTIONAL_PROMPT_ARGS=()
if [[ -n "$HINT_TEMPLATE" ]]; then
  OPTIONAL_PROMPT_ARGS+=(--hint_template "$HINT_TEMPLATE")
fi
if [[ -n "$LOCATE_SYSTEM_PROMPT" ]]; then
  OPTIONAL_PROMPT_ARGS+=(--locate_system_prompt "$LOCATE_SYSTEM_PROMPT")
fi

DATASET_TAG="$(basename "${DATASET_NAME%/}")"
DATASET_TAG="${DATASET_TAG//[^A-Za-z0-9._-]/_}"
RUN_CONFIG="${RUN_CONFIG:-opd_locate_${DATASET_TAG}}_${RUN_ID}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/${RUN_CONFIG}}"

uv run accelerate launch \
  --config_file "$ACCELERATE_CONFIG" \
  --num_processes "$NUM_PROCESSES" \
  --main_process_port "${MAIN_PROCESS_PORT:-13380}" \
  baseline/train_opd_locate.py \
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
  --locate_teacher_mode "$LOCATE_TEACHER_MODE" \
  --mask_box_transition "$MASK_BOX_TRANSITION" \
  --bbox_field "$BBOX_FIELD" \
  --filter_no_bbox "$FILTER_NO_BBOX" \
  --hint_coord_decimals "$HINT_COORD_DECIMALS" \
  "${OPTIONAL_PROMPT_ARGS[@]}" \
  --opd_prompt_suffix "$OPD_PROMPT_SUFFIX" \
  --opd_system_prompt "$OPD_PROMPT_STYLE" \
  --group_size "$GROUP_SIZE" \
  --lambda_opd "$LAMBDA_OPD" \
  --lambda_rl "$LAMBDA_RL" \
  --rl_reward "$RL_REWARD" \
  --rl_ungated_weight "$RL_UNGATED_WEIGHT" \
  --rl_normalize_adv "$RL_NORMALIZE_ADV" \
  --kl_position_gate "$KL_POSITION_GATE" \
  --output_dir "$OUTPUT_DIR" \
  --run_name "opd_locate_qwen3_2b_${RUN_ID}" \
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
