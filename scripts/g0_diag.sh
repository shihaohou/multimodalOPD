#!/usr/bin/env bash
set -euo pipefail

# G0 grounding diagnostic (looking-vs-using). Runs the 3 conditions (C1 teacher,
# C2 teacher+silent-hint, C3 student) over saliency-r1-8k, then the 4 analyses.
# No training. Single GPU: an 8B teacher + 2B student + the GLIMPSE grad forward
# fit one H800/A100-80G (the grad forward's memory ~ S^2, so MAX_PIXELS caps it).
#
# Required: STUDENT_MODEL. Optional TEACHER_MODEL (omit → C3 student only).
#
#   STUDENT_MODEL=$M/Qwen3-VL-2B-Instruct TEACHER_MODEL=$M/Qwen3-VL-8B-Instruct \
#     SUBSETS=textvqa,docvqa,gqa,openimages LIMIT=80 bash scripts/g0_diag.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT_DIR"

: "${STUDENT_MODEL:?Set STUDENT_MODEL to a model dir or HuggingFace model id (the 2B student).}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TRANSFORMERS_NO_TF=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

TEACHER_MODEL="${TEACHER_MODEL:-}"
RUN_NAME="${RUN_NAME:-run1}"
OUTPUT_DIR="${OUTPUT_DIR:-eval_outputs/g0/$RUN_NAME}"

DATASET="${DATASET:-peterant330/saliency-r1-8k}"
SPLIT="${SPLIT:-train}"
SUBSETS="${SUBSETS:-textvqa,docvqa,gqa,openimages}"   # single-region, small-box subsets
LIMIT="${LIMIT:-80}"                 # per-subset eval cap
CALIB_LIMIT="${CALIB_LIMIT:-40}"     # per-subset head-calibration cap (subset of eval)
MAX_BBOX_AREA="${MAX_BBOX_AREA:-0.5}"
MIN_BBOX_AREA="${MIN_BBOX_AREA:-}"
CONDITIONS="${CONDITIONS:-c1,c2,c3}"

ATTN="${ATTN:-eager}"                # MUST be eager for output_attentions
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
MAX_PIXELS="${MAX_PIXELS:-602112}"   # ~768 merged visual tokens; lower if GLIMPSE OOMs
MIN_PIXELS="${MIN_PIXELS:-}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-320}"
SAMPLE="${SAMPLE:-}"                 # set 1 to sample instead of greedy
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
SEED="${SEED:-0}"

TOP_K_HEADS="${TOP_K_HEADS:-3}"
MIN_LAYER="${MIN_LAYER:-2}"
LH_SIGMA="${LH_SIGMA:-1.0}"

GLIMPSE_LAYERS="${GLIMPSE_LAYERS:-}"             # default all
GLIMPSE_LAMBDA="${GLIMPSE_LAMBDA:-1.0}"
GLIMPSE_LAMBDA_DEPTH="${GLIMPSE_LAMBDA_DEPTH:-0.1}"
THRESHOLD="${THRESHOLD:-mean}"
VIZ_N="${VIZ_N:-8}"

CMD=(
  uv run python -m baseline.g0.run_g0
  --student-model "$STUDENT_MODEL"
  --output-dir "$OUTPUT_DIR"
  --dataset "$DATASET"
  --split "$SPLIT"
  --subsets "$SUBSETS"
  --limit "$LIMIT"
  --calib-limit "$CALIB_LIMIT"
  --max-bbox-area "$MAX_BBOX_AREA"
  --conditions "$CONDITIONS"
  --attn "$ATTN"
  --dtype "$DTYPE"
  --device "$DEVICE"
  --max-pixels "$MAX_PIXELS"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --seed "$SEED"
  --top-k-heads "$TOP_K_HEADS"
  --min-layer "$MIN_LAYER"
  --lh-sigma "$LH_SIGMA"
  --glimpse-lambda "$GLIMPSE_LAMBDA"
  --glimpse-lambda-depth "$GLIMPSE_LAMBDA_DEPTH"
  --threshold "$THRESHOLD"
  --viz-n "$VIZ_N"
)
if [[ -n "$TEACHER_MODEL" ]]; then CMD+=(--teacher-model "$TEACHER_MODEL"); fi
if [[ -n "$MIN_BBOX_AREA" ]]; then CMD+=(--min-bbox-area "$MIN_BBOX_AREA"); fi
if [[ -n "$MIN_PIXELS" ]]; then CMD+=(--min-pixels "$MIN_PIXELS"); fi
if [[ -n "$GLIMPSE_LAYERS" ]]; then CMD+=(--glimpse-layers "$GLIMPSE_LAYERS"); fi
if [[ "$SAMPLE" == "1" || "$SAMPLE" == "true" ]]; then CMD+=(--sample); fi

echo "[g0_diag] student=$STUDENT_MODEL teacher=${TEACHER_MODEL:-<none>} gpu=$CUDA_VISIBLE_DEVICES out=$OUTPUT_DIR"
"${CMD[@]}"

echo "[g0_diag] analyzing ..."
uv run python -m baseline.g0.analyze_g0 --run-dir "$OUTPUT_DIR"
echo "[g0_diag] done → $OUTPUT_DIR (report.md, analysis.json, figs/)"
