#!/usr/bin/env bash
set -euo pipefail

# Vanilla OPD preset for Qwen3.5 9B teacher -> Qwen3.5 2B student.
#
# Defaults target the same Visual-CoT-style comparison as the previous Qwen3 runs:
#   student: /home/web_server/antispam/project/houshihao/models/Qwen3.5-2B
#   teacher: /home/web_server/antispam/project/houshihao/models/Qwen3.5-9B
#   data:    /home/web_server/antispam/project/houshihao/datasets/Visual-CoT
#
# Override MODEL_NAME_OR_PATH, TEACHER_MODEL, DATASET_NAME, or M/D as needed.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT_DIR"

M="${M:-/home/web_server/antispam/project/houshihao/models}"
D="${D:-/home/web_server/antispam/project/houshihao/datasets}"

export MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${M%/}/Qwen3.5-2B}"
export TEACHER_MODEL="${TEACHER_MODEL:-${M%/}/Qwen3.5-9B}"
export DATASET_NAME="${DATASET_NAME:-${D%/}/Visual-CoT}"
export ANSWER_FIELD="${ANSWER_FIELD:-answer}"
# Cap Qwen3.5 visual tokens. Without this, high-res Visual-CoT images can expand
# to ~16k image placeholders and be cut by MAX_PROMPT_LENGTH truncation.
export MAX_PIXELS="${MAX_PIXELS:-1048576}"
# Qwen3.5 needs a newer vLLM than the repo's Qwen3 stack. Default to HF rollout
# for compatibility; set USE_VLLM=true only in an environment with Qwen3.5 vLLM support.
export USE_VLLM="${USE_VLLM:-false}"

DATASET_TAG="$(basename "${DATASET_NAME%/}")"
DATASET_TAG="${DATASET_TAG//[^A-Za-z0-9._-]/_}"
export RUN_CONFIG="${RUN_CONFIG:-opd_qwen3.5_9B-to-2B_${DATASET_TAG}}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-13382}"

echo "[opd-qwen35] student=$MODEL_NAME_OR_PATH"
echo "[opd-qwen35] teacher=$TEACHER_MODEL"
echo "[opd-qwen35] dataset=$DATASET_NAME answer_field=$ANSWER_FIELD"
echo "[opd-qwen35] use_vllm=$USE_VLLM"
echo "[opd-qwen35] run_config=${RUN_CONFIG}_<RUN_ID>"

exec bash scripts/train_opd.sh
