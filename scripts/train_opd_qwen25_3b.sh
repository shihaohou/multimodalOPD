#!/usr/bin/env bash
set -euo pipefail

# Vanilla multimodal On-Policy Distillation (OPD) for a Qwen2.5-VL-3B student
# against a separate, frozen, stronger same-family VLM teacher.
#
# Required:
#   DATASET_NAME   HuggingFace dataset id (must provide problem/images/answer).
#   TEACHER_MODEL  Path/id of the frozen teacher checkpoint (base or RL-tuned).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT_DIR"

: "${DATASET_NAME:?Set DATASET_NAME to the HuggingFace training dataset id.}"
# Teacher source: local_hf (frozen replica per GPU) | vllm_server (separate
# server returning top-k logprobs; start it with scripts/serve_teacher_vllm.sh).
TEACHER_SOURCE="${TEACHER_SOURCE:-local_hf}"
TEACHER_SERVER_URL="${TEACHER_SERVER_URL:-http://127.0.0.1:8200}"
if [[ "$TEACHER_SOURCE" == "local_hf" ]]; then
  : "${TEACHER_MODEL:?Set TEACHER_MODEL (local_hf), or use TEACHER_SOURCE=vllm_server.}"
fi
TEACHER_MODEL="${TEACHER_MODEL:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TRANSFORMERS_NO_TF=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_PROJECT="${WANDB_PROJECT:-MultimodalOPD}"
# WandB online by default (needs network + WANDB_API_KEY); set offline to disable.
export WANDB_MODE="${WANDB_MODE:-online}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost,0.0.0.0"
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost,0.0.0.0"

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate_zero2_gpu_8.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/opd_qwen25_3b_${RUN_ID}}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}"
TEACHER_TORCH_DTYPE="${TEACHER_TORCH_DTYPE:-bfloat16}"
# Attention impl for the HF training forward (student + local_hf teacher). Default
# sdpa so training runs without flash-attn installed; set flash_attention_2 once
# it is built. (eval + vllm_server teacher use vLLM and never need flash-attn.)
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
TEACHER_ATTN_IMPLEMENTATION="${TEACHER_ATTN_IMPLEMENTATION:-sdpa}"
# Full-parameter training by default (like Vision-OPD); set to "lora" for a
# cheap memory-constrained run.
FINETUNING_MODE="${FINETUNING_MODE:-full}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
# Drop degenerate tiny images: a 1-3px side trips the Qwen image processor's
# channel-axis inference -> "mean must have 1 elements" crash. On by default;
# 28px = one Qwen merged patch. The collator also pads such images as a backstop.
FILTER_TINY_IMAGES="${FILTER_TINY_IMAGES:-true}"
MIN_IMAGE_SIZE="${MIN_IMAGE_SIZE:-28}"
MAX_STEPS="${MAX_STEPS:-}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
# Full FT. Paper (Vision-OPD/VGS Table 4) global batch 512 = per_device 4 x
# grad_accum 16 x 8 GPU. per_device 4 is the stable max on the 140GB cards (~3.5h).
# per_device 8 is faster (~2h) BUT OOMs the training forward on heavy multi-image
# batches and lowering vLLM util does NOT help: vLLM is small here (~129G is the
# per_device-8 training activation itself, not the KV cache). Only fewer sequences
# per micro-step fixes it. Rescale grad_accum to keep batch 512 if you change GPUs.
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
# Paper (Table 4): AdamW, lr 1e-6, weight_decay 1e-2, constant schedule, no warmup.
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-constant}"
WARMUP_RATIO="${WARMUP_RATIO:-0.0}"
# Freeze the vision tower under full FT. Off by default: at a real multi-GPU
# effective batch (e.g. 8 GPU x grad_accum 4 = 32) the ViT grad spikes average
# out and full FT incl. ViT is stable (matches Vision-OPD). Turn on as a fallback
# for small-batch / single-GPU runs where the ViT bf16 grad can overflow -> NaN.
FREEZE_VISION_TOWER="${FREEZE_VISION_TOWER:-false}"
# Paper (Table 4): max input prompt 16384, max response 2048.
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-16384}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-2048}"
ANSWER_FIELD="${ANSWER_FIELD:-answer}"
# Format instruction lives in the unified system prompt (baseline/opd_data_collator
# OPD_SYSTEM_PROMPT); the user turn is just the question, so no suffix by default.
OPD_PROMPT_SUFFIX="${OPD_PROMPT_SUFFIX:-}"
# Paper (Table 4) rollout: temperature 1.0, top_p 1.0, no top-k (0 -> vLLM -1).
GENERATION_TEMPERATURE="${GENERATION_TEMPERATURE:-1.0}"
GENERATION_TOP_P="${GENERATION_TOP_P:-1.0}"
GENERATION_TOP_K="${GENERATION_TOP_K:-0}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-1.0}"
LAMBDA_OPD="${LAMBDA_OPD:-1.0}"
# Distillation loss. Default = exact reverse KL (canonical OPD); the local teacher
# has full logits so full-vocab is free. Use topk_kl + forward for the vllm_server
# teacher (it returns only the teacher's top-k logprobs).
OPD_LOSS_MODE="${OPD_LOSS_MODE:-full_kl}"          # full_kl | topk_kl
OPD_KL_DIRECTION="${OPD_KL_DIRECTION:-reverse}"    # reverse | forward | jsd
OPD_TOP_K="${OPD_TOP_K:-32}"
TOKEN_LOSS_CLIP="${TOKEN_LOSS_CLIP:-0.0}"
PRESENCE_PENALTY="${PRESENCE_PENALTY:-0.0}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
MIN_P="${MIN_P:-0.0}"
USE_VLLM="${USE_VLLM:-true}"
VLLM_MODE="${VLLM_MODE:-colocate}"
# Colocate shares the pool with the frozen teacher replica + the training forward.
# vLLM's own footprint is small here (2B weights + modest KV), so tuning this does
# NOT fix per_device-8 OOMs (the training activation is the consumer). 0.25 is fine
# at per_device 4.
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.25}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
VLLM_SYNC_FREQUENCY="${VLLM_SYNC_FREQUENCY:-1}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH))}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-$((PER_DEVICE_TRAIN_BATCH_SIZE * VLLM_TENSOR_PARALLEL_SIZE))}"
VLLM_DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-false}"
VLLM_SERVER_BASE_URL="${VLLM_SERVER_BASE_URL:-}"
VLLM_SERVER_HOST="${VLLM_SERVER_HOST:-127.0.0.1}"
VLLM_SERVER_PORT="${VLLM_SERVER_PORT:-8000}"
VLLM_SERVER_TIMEOUT="${VLLM_SERVER_TIMEOUT:-600}"
VLLM_SERVER_GROUP_PORT="${VLLM_SERVER_GROUP_PORT:-51216}"
VLLM_SERVER_REQUEST_BATCH_SIZE="${VLLM_SERVER_REQUEST_BATCH_SIZE:-}"
COMPLETION_LOG_STEPS="${COMPLETION_LOG_STEPS:-0}"
COMPLETION_LOG_MAX_SAMPLES="${COMPLETION_LOG_MAX_SAMPLES:-16}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
# Paper batch 512 -> ~92 steps/epoch, so save often for an acc curve. Raise
# SAVE_TOTAL_LIMIT (or override per run) to keep all checkpoints for the curve.
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

VLLM_SERVER_ARGS=()
if [[ -n "$VLLM_SERVER_BASE_URL" ]]; then
  VLLM_SERVER_ARGS+=(--vllm_server_base_url "$VLLM_SERVER_BASE_URL")
fi
if [[ -n "$VLLM_SERVER_REQUEST_BATCH_SIZE" ]]; then
  VLLM_SERVER_ARGS+=(--vllm_server_request_batch_size "$VLLM_SERVER_REQUEST_BATCH_SIZE")
fi

RUN_CONFIG="${RUN_CONFIG:-opd_qwen25_3b_gen${MAX_COMPLETION_LENGTH}_mb${PER_DEVICE_TRAIN_BATCH_SIZE}_ga${GRADIENT_ACCUMULATION_STEPS}_np${NUM_PROCESSES}}"

uv run accelerate launch \
  --config_file "$ACCELERATE_CONFIG" \
  --num_processes "$NUM_PROCESSES" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --main_process_port "${MAIN_PROCESS_PORT:-13378}" \
  baseline/train_opd.py \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --finetuning_mode "$FINETUNING_MODE" \
  --attn_implementation "$ATTN_IMPLEMENTATION" \
  --teacher_source "$TEACHER_SOURCE" \
  --teacher_server_url "$TEACHER_SERVER_URL" \
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
  --run_name "opd_qwen25_3b_${RUN_ID}" \
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
  --vllm_server_host "$VLLM_SERVER_HOST" \
  --vllm_server_port "$VLLM_SERVER_PORT" \
  --vllm_server_timeout "$VLLM_SERVER_TIMEOUT" \
  --vllm_server_group_port "$VLLM_SERVER_GROUP_PORT" \
  "${VLLM_SERVER_ARGS[@]}" \
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
