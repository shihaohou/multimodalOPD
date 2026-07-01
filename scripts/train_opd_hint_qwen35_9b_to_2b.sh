#!/usr/bin/env bash
set -euo pipefail

# Grounding-Hint OPD preset for Qwen3.5 9B teacher -> Qwen3.5 2B student.
#
# This mirrors the Qwen3 Visual-CoT no-verbalize hint run, but swaps in:
#   student: /home/web_server/antispam/project/houshihao/models/Qwen3.5-2B
#   teacher: /home/web_server/antispam/project/houshihao/models/Qwen3.5-9B
#   data:    /home/web_server/antispam/project/houshihao/datasets/Visual-CoT
#
# The default hint template already forbids verbalizing the box/coordinates. Use
# TEACHER_PRIVILEGE_MODE=crop for the crop ablation, or override HINT_TEMPLATE for
# a prompt ablation.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT_DIR"

M="${M:-/home/web_server/antispam/project/houshihao/models}"
D="${D:-/home/web_server/antispam/project/houshihao/datasets}"

export MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${M%/}/Qwen3.5-2B}"
export TEACHER_MODEL="${TEACHER_MODEL:-${M%/}/Qwen3.5-9B}"
export DATASET_NAME="${DATASET_NAME:-${D%/}/Visual-CoT}"
export ANSWER_FIELD="${ANSWER_FIELD:-answer}"
export BBOX_FIELD="${BBOX_FIELD:-bbox}"
export TEACHER_PRIVILEGE_MODE="${TEACHER_PRIVILEGE_MODE:-hint}"
export FILTER_NO_BBOX="${FILTER_NO_BBOX:-true}"
# Qwen3.5 needs a newer vLLM than the repo's Qwen3 stack. Default to HF rollout
# for compatibility; set USE_VLLM=true only in an environment with Qwen3.5 vLLM support.
export USE_VLLM="${USE_VLLM:-false}"

DATASET_TAG="$(basename "${DATASET_NAME%/}")"
DATASET_TAG="${DATASET_TAG//[^A-Za-z0-9._-]/_}"
if [[ "$TEACHER_PRIVILEGE_MODE" == "hint" ]]; then
  PRIVILEGE_TAG="noverbalize"
else
  PRIVILEGE_TAG="$TEACHER_PRIVILEGE_MODE"
fi
export RUN_CONFIG="${RUN_CONFIG:-hint_opd_qwen3.5_9B-to-2B_${DATASET_TAG}_${PRIVILEGE_TAG}}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-13383}"

echo "[hint-opd-qwen35] student=$MODEL_NAME_OR_PATH"
echo "[hint-opd-qwen35] teacher=$TEACHER_MODEL"
echo "[hint-opd-qwen35] dataset=$DATASET_NAME answer_field=$ANSWER_FIELD bbox_field=$BBOX_FIELD"
echo "[hint-opd-qwen35] use_vllm=$USE_VLLM"
echo "[hint-opd-qwen35] privilege=$TEACHER_PRIVILEGE_MODE run_config=${RUN_CONFIG}_<RUN_ID>"

exec bash scripts/train_opd_hint_qwen3_2b.sh
