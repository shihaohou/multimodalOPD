#!/usr/bin/env bash
set -euo pipefail

# OPD preset: Qwen3-VL-8B (frozen teacher) -> Qwen3-VL-2B (student), ViT FROZEN.
#
# Thin wrapper over the generic launcher scripts/train_opd.sh: it only
# pins the Qwen3-VL student/teacher pair and turns on FREEZE_VISION_TOWER. Every
# other knob keeps the generic script's paper-aligned (Vision-OPD/VGS Table 4)
# defaults and stays env-overridable. Freezing the vision tower (visual.*) only
# *reduces* memory vs. the verified unfrozen run, so the per_device/grad_accum/
# vllm-util defaults remain safe (don't override them).
#
# Required:
#   DATASET_NAME   HF id or local path to the training set (Vision-SR1-47K).
# Optional:
#   M              Models dir for local OFFLINE checkpoints (recommended on the
#                  cluster). If set, student/teacher default to
#                  $M/Qwen3-VL-2B-Instruct and $M/Qwen3-VL-8B-Instruct;
#                  otherwise they default to the HuggingFace hub ids.
#   MODEL_NAME_OR_PATH / TEACHER_MODEL   Override the pair explicitly (ignores $M).
#
# Verified-style launch (8xH800, offline weights, WandB online):
#   cd <repo> && git pull
#   export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NUM_PROCESSES=8 WANDB_MODE=online
#   unset  MAX_STEPS MAX_TRAIN_SAMPLES
#   export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
#   M=/home/web_server/antispam/project/houshihao/models \
#   DATASET_NAME=/home/web_server/antispam/project/houshihao/datasets/Vision-SR1-47K \
#   bash scripts/train_opd_qwen3_8b_to_2b.sh
#   # confirm the "[OPD] num_processes(world_size)=8 ... effective_batch=512" print.
# Smoke test: prepend  MAX_STEPS=5 SAVE_STEPS=5 REPORT_TO=none WANDB_MODE=offline.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT_DIR"

: "${DATASET_NAME:?Set DATASET_NAME (HF id or local path to the training set).}"

# Resolve the student/teacher checkpoints. Prefer a local models dir ($M) for
# offline runs; fall back to the HuggingFace hub ids.
M="${M:-}"
if [[ -n "$M" ]]; then
  export MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${M%/}/Qwen3-VL-2B-Instruct}"
  export TEACHER_MODEL="${TEACHER_MODEL:-${M%/}/Qwen3-VL-8B-Instruct}"
else
  export MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-VL-2B-Instruct}"
  export TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
fi

# The point of this preset: freeze the Qwen3-VL vision tower under full FT.
export FREEZE_VISION_TOWER="${FREEZE_VISION_TOWER:-true}"

# Distinct run id; basename(OUTPUT_DIR) == RUN_CONFIG avoids subdir nesting in
# baseline/train_opd.py.
export RUN_CONFIG="${RUN_CONFIG:-opd_qwen3_8b_to_2b_freezevit}"
export OUTPUT_DIR="${OUTPUT_DIR:-runs/${RUN_CONFIG}}"

# sdpa so the HF training forward runs without flash-attn built (override both to
# flash_attention_2 once it is installed).
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
export TEACHER_ATTN_IMPLEMENTATION="${TEACHER_ATTN_IMPLEMENTATION:-sdpa}"

echo "[opd-qwen3-freezevit] student=$MODEL_NAME_OR_PATH"
echo "[opd-qwen3-freezevit] teacher=$TEACHER_MODEL  (frozen)"
echo "[opd-qwen3-freezevit] freeze_vision_tower=$FREEZE_VISION_TOWER  output_dir=$OUTPUT_DIR"

exec bash "$ROOT_DIR/scripts/train_opd.sh"
