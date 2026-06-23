#!/usr/bin/env bash
set -euo pipefail

# GRPO-train a Qwen3-VL teacher on Vision-SR1-47K with ms-swift, producing a
# stronger checkpoint to use as the OPD teacher (teacher_model_name_or_path).
#
# Run INSIDE the ms-swift venv (separate from the OPD env):
#   source /root/shihao_project/swift-env/.venv/bin/activate
# First build the dataset with prepare_vision_sr1.py.
#
# Rollout backend:
#   USE_VLLM=false (default) -> transformers rollout (slow but safe; Qwen3-VL is
#                               supported by transformers 4.57.1)
#   USE_VLLM=true            -> vLLM colocate rollout (fast; needs vLLM installed
#                               AND vLLM support for Qwen3-VL — verify first)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

MODEL="${MODEL:-/home/web_server/antispam/project/houshihao/models/Qwen3-VL-8B-Instruct}"
DATA="${DATA:-/home/web_server/antispam/project/houshihao/datasets/vision_sr1_swift/train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/web_server/antispam/project/houshihao/runs/teacher_grpo_qwen3vl8b}"

TUNER_TYPE="${TUNER_TYPE:-lora}"        # lora (cheap; merge after) | full (paper-faithful, needs zero3)
USE_VLLM="${USE_VLLM:-false}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1024}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
MAX_PIXELS="${MAX_PIXELS:-1003520}"
TEMPERATURE="${TEMPERATURE:-1.0}"
SAVE_STEPS="${SAVE_STEPS:-200}"
DEEPSPEED="${DEEPSPEED:-zero2}"         # use zero3 for TUNER_TYPE=full on 8B
MAX_STEPS="${MAX_STEPS:-}"              # set small (e.g. 4) for a smoke
SYSTEM_PROMPT="${SYSTEM_PROMPT:-Solve the problem step by step and put your final answer within \\boxed{}.}"

EXTRA_ARGS=()
if [[ "$USE_VLLM" == "true" ]]; then
  EXTRA_ARGS+=(--use_vllm true --vllm_mode colocate
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.5}"
    --vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE:-1}"
    --sleep_level 1)
fi
if [[ -n "$MAX_STEPS" ]]; then
  EXTRA_ARGS+=(--max_steps "$MAX_STEPS")
fi

MAX_PIXELS="$MAX_PIXELS" NPROC_PER_NODE="$NPROC_PER_NODE" \
swift rlhf \
  --rlhf_type grpo \
  --model "$MODEL" \
  --tuner_type "$TUNER_TYPE" \
  --external_plugins "$HERE/reward_accuracy.py" \
  --reward_funcs vqa_accuracy vqa_format \
  --dataset "$DATA" \
  --load_from_cache_file true \
  --torch_dtype bfloat16 \
  --system "$SYSTEM_PROMPT" \
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --learning_rate "$LEARNING_RATE" \
  --lr_scheduler_type cosine \
  --max_completion_length "$MAX_COMPLETION_LENGTH" \
  --max_length "$MAX_LENGTH" \
  --num_generations "$NUM_GENERATIONS" \
  --temperature "$TEMPERATURE" \
  --logging_steps 1 \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit 5 \
  --warmup_ratio 0.05 \
  --dataloader_num_workers 4 \
  --deepspeed "$DEEPSPEED" \
  --output_dir "$OUTPUT_DIR" \
  --log_completions true \
  --report_to tensorboard \
  "${EXTRA_ARGS[@]}"
