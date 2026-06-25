#!/usr/bin/env bash
set -euo pipefail

# OPD + differentiable TAM visual-evidence alignment.
# Preset: Qwen3-VL-8B (frozen teacher) -> Qwen3-VL-2B (student), ViT TRAINED (full).
#
# Adds the Token Activation Map alignment loss on top of vanilla OPD (see
# baseline/tam/). Unlike the evidence/saliency path, TAM needs only
# output_hidden_states (NO attention weights) -> runs under SDPA/Flash, no eager,
# much cheaper. The student vision tower is trained (NOT frozen) so this matches
# the vanilla full-FT OPD baseline knob-for-knob; the only added ingredient is the
# TAM term (lambda_tam). Set LAMBDA_TAM=0 to recover vanilla OPD exactly.
#
# Required:
#   DATASET_NAME   HF id or local path to the training set (Vision-SR1-47K).
# Optional:
#   M              Models dir for local OFFLINE checkpoints (recommended on the
#                  cluster). If set, student/teacher default to
#                  $M/Qwen3-VL-2B-Instruct and $M/Qwen3-VL-8B-Instruct.
#   MODEL_NAME_OR_PATH / TEACHER_MODEL   Override the pair explicitly (ignores $M).
#
# Validate the engine FIRST (grad/grid/no-attention/memory, ~1 GPU):
#   uv run python -m baseline.tam.sanity_check \
#       --student_model "$MODEL_NAME_OR_PATH" --teacher_model "$TEACHER_MODEL"
#
# Smoke test: prepend  MAX_STEPS=5 SAVE_STEPS=5 REPORT_TO=none WANDB_MODE=offline.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT_DIR"

# Resolve the dataset from $D (datasets dir) when DATASET_NAME isn't passed.
D="${D:-}"
if [[ -n "$D" ]]; then
  DATASET_NAME="${DATASET_NAME:-${D%/}/Vision-SR1-47K}"
fi
: "${DATASET_NAME:?Set DATASET_NAME (HF id / local path) or D=<datasets dir>.}"

# Resolve the student/teacher checkpoints. Prefer a local models dir ($M). The
# default local teacher is Vero (GRPO-trained from Qwen3-VL-8B, our standard
# teacher) — set TEACHER_MODEL to use a stock-8B / different teacher.
M="${M:-}"
if [[ -n "$M" ]]; then
  MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${M%/}/Qwen3-VL-2B-Instruct}"
  TEACHER_MODEL="${TEACHER_MODEL:-${M%/}/Vero-Qwen3I-8B}"
else
  MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-VL-2B-Instruct}"
  TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
fi

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
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH}"
TEACHER_MODEL="${TEACHER_MODEL}"
TEACHER_TORCH_DTYPE="${TEACHER_TORCH_DTYPE:-bfloat16}"
# Default to flash_attention_2 (faster + lower memory for the long-sequence VLM
# forwards). If flash-attn is NOT built in the env the model load errors loudly —
# then set ATTN_IMPLEMENTATION=sdpa TEACHER_ATTN_IMPLEMENTATION=sdpa to fall back.
# TAM is flash-compatible (reads output_hidden_states, not attention weights).
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
TEACHER_ATTN_IMPLEMENTATION="${TEACHER_ATTN_IMPLEMENTATION:-flash_attention_2}"
FINETUNING_MODE="${FINETUNING_MODE:-full}"
# Train the vision tower (matches the vanilla full-FT OPD baseline for a clean
# OPD-vs-OPD+TAM comparison; the migration doc's MVP "freeze ViT" is overridden
# here on purpose).
FREEZE_VISION_TOWER="${FREEZE_VISION_TOWER:-false}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
# Drop degenerate tiny images: a 1-3px side trips the Qwen image processor's
# channel-axis inference -> "mean must have 1 elements" crash. On by default;
# 28px = one Qwen merged patch. The collator also pads such images as a backstop.
FILTER_TINY_IMAGES="${FILTER_TINY_IMAGES:-true}"
MIN_IMAGE_SIZE="${MIN_IMAGE_SIZE:-28}"
MAX_STEPS="${MAX_STEPS:-}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
# Default eff_batch = pd(4)*ga(16)*world(8) = 512 (matches the OPD baseline). TAM
# adds two output_hidden_states forwards, so pd only affects memory, not the
# gradient: if OOM, PER_DEVICE_TRAIN_BATCH_SIZE=2 GRADIENT_ACCUMULATION_STEPS=32
# (still 512); if memory is plentiful, pd=8 ga=8. eff_batch = pd*ga*world.
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-constant}"
WARMUP_RATIO="${WARMUP_RATIO:-0.0}"
# Gradient-norm clip. 1.0 = the OPD field standard (Uni-OPD/miles --clip-grad=1.0,
# OPD-main LlamaFactory max_grad_norm=1.0 and verl actor.grad_clip=1.0); the ViGOS
# repo's old hardcoded 0.1 was 10x tighter and (with lr=1e-6) throttled updates.
# Keep this identical between the OPD baseline (LAMBDA_TAM=0) and OPD+TAM so the
# comparison stays clean.
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-16384}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-2048}"
# Cap image resolution -> caps #visual tokens -> caps the per-layer hidden-state
# retention of output_hidden_states (the TAM memory lever; far milder than the
# evidence eager-attention S^2 wall). Empty = processor default.
MAX_PIXELS="${MAX_PIXELS:-}"
MIN_PIXELS="${MIN_PIXELS:-}"
ANSWER_FIELD="${ANSWER_FIELD:-answer}"
OPD_PROMPT_SUFFIX="${OPD_PROMPT_SUFFIX:-}"
GENERATION_TEMPERATURE="${GENERATION_TEMPERATURE:-1.0}"
GENERATION_TOP_P="${GENERATION_TOP_P:-1.0}"
GENERATION_TOP_K="${GENERATION_TOP_K:-0}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-1.0}"
LAMBDA_OPD="${LAMBDA_OPD:-1.0}"
# Default top-k reverse KL (top-100): OPD-ecosystem standard, ~99% mass, and avoids
# the full-vocab exp/diff OOM at mb8. OPD_LOSS_MODE=full_kl for exact full-vocab KL.
OPD_LOSS_MODE="${OPD_LOSS_MODE:-topk_kl}"
OPD_KL_DIRECTION="${OPD_KL_DIRECTION:-reverse}"
OPD_TOP_K="${OPD_TOP_K:-100}"
TOKEN_LOSS_CLIP="${TOKEN_LOSS_CLIP:-0.0}"

# --- TAM visual-evidence alignment knobs (migration doc §2 / MVP) -------------
LAMBDA_TAM="${LAMBDA_TAM:-1.0}"
# completion = align on all rollout tokens, gate selects visual ones (default).
TAM_ALIGN_SPAN="${TAM_ALIGN_SPAN:-completion}"
TAM_USE_ECI="${TAM_USE_ECI:-true}"
TAM_DETACH_LM_HEAD="${TAM_DETACH_LM_HEAD:-true}"
TAM_DIVERGENCE="${TAM_DIVERGENCE:-cosine}"      # cosine | js | l1
TAM_BLUR="${TAM_BLUR:-true}"
TAM_BLUR_KERNEL="${TAM_BLUR_KERNEL:-3}"
TAM_BLUR_SIGMA="${TAM_BLUR_SIGMA:-1.0}"
TAM_GATE_TEMP="${TAM_GATE_TEMP:-1.0}"
TAM_GATE_H0="${TAM_GATE_H0:-0.9}"
TAM_GATE_TAU="${TAM_GATE_TAU:-0.1}"
TAM_MASS_THRESHOLD="${TAM_MASS_THRESHOLD:-0.0}"
TAM_MAX_TOKENS="${TAM_MAX_TOKENS:-0}"

USE_VLLM="${USE_VLLM:-true}"
VLLM_MODE="${VLLM_MODE:-colocate}"
# Slightly below vanilla OPD (0.25): the two output_hidden_states forwards want
# headroom. The 2B-student rollout only needs a small vLLM pool.
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.2}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
VLLM_SYNC_FREQUENCY="${VLLM_SYNC_FREQUENCY:-1}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH))}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-$((PER_DEVICE_TRAIN_BATCH_SIZE * VLLM_TENSOR_PARALLEL_SIZE))}"
COMPLETION_LOG_STEPS="${COMPLETION_LOG_STEPS:-0}"
COMPLETION_LOG_MAX_SAMPLES="${COMPLETION_LOG_MAX_SAMPLES:-16}"
# On by default: TAM is GC-compatible (it uses output_hidden_states, which the
# forward returns regardless of gradient checkpointing — no attention capture to
# be swallowed, unlike the evidence path).
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
SAVE_STEPS="${SAVE_STEPS:-200}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-100}"
SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-true}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"
REPORT_TO="${REPORT_TO:-wandb}"

echo "[opd-tam-qwen3] student=$MODEL_NAME_OR_PATH"
echo "[opd-tam-qwen3] teacher=$TEACHER_MODEL  (frozen)"
echo "[opd-tam-qwen3] freeze_vision_tower=$FREEZE_VISION_TOWER  lambda_tam=$LAMBDA_TAM  divergence=$TAM_DIVERGENCE"

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

PIXEL_ARGS=()
if [[ -n "$MAX_PIXELS" ]]; then
  PIXEL_ARGS+=(--max_pixels "$MAX_PIXELS")
fi
if [[ -n "$MIN_PIXELS" ]]; then
  PIXEL_ARGS+=(--min_pixels "$MIN_PIXELS")
fi

# Auto-name encodes lambda_tam + ViT mode (fullft vs freezevit) so OPD (ltam0) /
# OPD+TAM / full-FT / frozen-ViT runs are distinguishable without passing RUN_CONFIG.
VIT_TAG=$([[ "$FREEZE_VISION_TOWER" == "true" ]] && echo freezevit || echo fullft)
RUN_CONFIG="${RUN_CONFIG:-opd_tam_qwen3_8b_to_2b_ltam${LAMBDA_TAM}_${VIT_TAG}}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/${RUN_CONFIG}}"

uv run accelerate launch \
  --config_file "$ACCELERATE_CONFIG" \
  --num_processes "$NUM_PROCESSES" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --main_process_port "${MAIN_PROCESS_PORT:-13391}" \
  baseline/train_opd_tam.py \
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
  --run_name "opd_tam_qwen3_8b_to_2b_${RUN_ID}" \
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
  "${PIXEL_ARGS[@]}" \
  --generation_temperature "$GENERATION_TEMPERATURE" \
  --generation_top_p "$GENERATION_TOP_P" \
  --generation_top_k "$GENERATION_TOP_K" \
  --distill_temperature "$DISTILL_TEMPERATURE" \
  --lambda_opd "$LAMBDA_OPD" \
  --opd_loss_mode "$OPD_LOSS_MODE" \
  --opd_kl_direction "$OPD_KL_DIRECTION" \
  --opd_top_k "$OPD_TOP_K" \
  --token_loss_clip "$TOKEN_LOSS_CLIP" \
  --lambda_tam "$LAMBDA_TAM" \
  --tam_align_span "$TAM_ALIGN_SPAN" \
  --tam_use_eci "$TAM_USE_ECI" \
  --tam_detach_lm_head "$TAM_DETACH_LM_HEAD" \
  --tam_divergence "$TAM_DIVERGENCE" \
  --tam_blur "$TAM_BLUR" \
  --tam_blur_kernel "$TAM_BLUR_KERNEL" \
  --tam_blur_sigma "$TAM_BLUR_SIGMA" \
  --tam_gate_temp "$TAM_GATE_TEMP" \
  --tam_gate_h0 "$TAM_GATE_H0" \
  --tam_gate_tau "$TAM_GATE_TAU" \
  --tam_mass_threshold "$TAM_MASS_THRESHOLD" \
  --tam_max_tokens "$TAM_MAX_TOKENS" \
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
