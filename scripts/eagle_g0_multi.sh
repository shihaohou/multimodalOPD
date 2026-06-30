#!/usr/bin/env bash
set -uo pipefail

# EAGLE-G0 across the 4 models, each pinned to its own GPU group, run CONCURRENTLY.
# Then (optionally) LLM-judge each, and one cross-model EAGLE analysis (tables
# 1-4 + per-task-type). Mirrors the bbox-A/B/C orchestrator: one worker/model.
#
# 5 models (override MODELS=...): teacher CapCurriculum-8B, ref Qwen3-VL-8B-Instruct,
# RAW base student Qwen3-VL-2B-Instruct, vanilla-OPD 2B, hint-OPD 2B. The raw 2B is
# REQUIRED for table 4 (raw → OPD → hint-OPD: did training raise image-reliance?).
# Format: "name=path;name=path;...". GPU_GROUPS pairs 1:1 with models (";"-separated,
# each a comma list of GPU ids).
#
#   export M=/.../models D=/.../datasets
#   JUDGE=1 JUDGE_API_URL=http://localhost:8000/v1 JUDGE_MODEL=Qwen3-30B-A3B \
#     bash scripts/eagle_g0_multi.sh
# → eval_outputs/eagle_g0/{<name>/...}  +  eval_outputs/eagle_g0/eagle_report.md

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT_DIR"

M="${M:-models}"
OUTPUT_BASE="${OUTPUT_BASE:-eval_outputs/eagle_g0}"
mkdir -p "$OUTPUT_BASE"

# Python runner — use an already-active venv directly (see eagle_g0.sh). Workers go
# through eagle_g0.sh (which inherits $VIRTUAL_ENV and does the same), so this only
# covers the judge/analyze calls below. Override with PYRUN=...
if [[ -n "${PYRUN:-}" ]]; then read -r -a PY <<< "$PYRUN";
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then PY=(python);
else PY=(uv run python); fi

# 5 models, "name=path" pairs separated by ';'. Students keep relative run paths.
MODELS="${MODELS:-\
qwen3vl-8b=$M/Qwen3-VL-8B-Instruct;\
capcurriculum-8b=$M/CapCurriculum-8B;\
qwen3vl-2b=$M/Qwen3-VL-2B-Instruct;\
qwen3vl-2b-opd=runs/opd_qwen3_CapCurriculum-8B-to-2B_Visual-CoT_20260628-044124/checkpoint-100;\
qwen3vl-2b-hint=runs/hint_opd_qwen3_CapCurriculum-8B-to-2B_Visual-CoT_noverbalize_20260628-090259/checkpoint-100}"
# 8 GPUs over 5 models: the two 8B get 2 cards, the three 2B get 1-2 cards.
GPU_GROUPS="${GPU_GROUPS:-0,1;2,3;4;5;6,7}"

# pass-through knobs
export DATASET="${DATASET:-${D:+$D/saliency-r1-8k}}"; export DATASET="${DATASET:-peterant330/saliency-r1-8k}"
# Default = ALL 10 saliency-r1-8k subsets (flickr30k/gqa/openimages/docvqa/textcap/
# v7w/textvqa/infographicsvqa/cub/vsr). Free-form ones (flickr30k/textcap/v7w) need
# the LLM judge — keep JUDGE=1. Override e.g. SUBSETS=gqa,openimages,v7w,vsr,textvqa.
export SUBSETS="${SUBSETS-}"
export LIMIT="${LIMIT:-60}"
export CONDITIONS="${CONDITIONS:-plain,hint}"
export HINT_MODE="${HINT_MODE:-generate}"          # generate | score_plain_y (OPD-faithful table 3)
export EAGLE_IMAGE_SIZE="${EAGLE_IMAGE_SIZE:-448}"
export N_REGIONS="${N_REGIONS:-49}"
export EAGLE_BATCH_SIZE="${EAGLE_BATCH_SIZE:-8}"
export EAGLE_THRESHOLD="${EAGLE_THRESHOLD:-mean}"  # run again with top_frac for a sensitivity check
export EAGLE_TOP_FRAC="${EAGLE_TOP_FRAC:-0.25}"
export GRAD_PROBES="${GRAD_PROBES:-1}"
export SALR1="${SALR1:-1}"                         # Saliency-R1 secondary attribution baseline
export SALR1_LAYERS="${SALR1_LAYERS:-all}"
export SALR1_THINK_ROW_MODE="${SALR1_THINK_ROW_MODE:-state}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-192}"

JUDGE="${JUDGE:-}"
JUDGE_API_URL="${JUDGE_API_URL:-http://localhost:8000/v1}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen3-30B-A3B}"
JUDGE_NO_THINK="${JUDGE_NO_THINK:-1}"

IFS=';' read -r -a MODEL_ARR <<< "$MODELS"
IFS=';' read -r -a GROUP_ARR <<< "$GPU_GROUPS"
if [[ "${#MODEL_ARR[@]}" -ne "${#GROUP_ARR[@]}" ]]; then
  echo "[eagle_multi] ${#MODEL_ARR[@]} models but ${#GROUP_ARR[@]} GPU groups — must match." >&2
  exit 1
fi

echo "[eagle_multi] dataset=$DATASET subsets='${SUBSETS}' limit=$LIMIT conditions=$CONDITIONS"
run_dirs=()
pids=()
for idx in "${!MODEL_ARR[@]}"; do
  entry="${MODEL_ARR[$idx]}"; name="${entry%%=*}"; path="${entry#*=}"; gpus="${GROUP_ARR[$idx]}"
  [[ -z "$name" || -z "$path" ]] && continue
  outdir="$OUTPUT_BASE/$name"; run_dirs+=("$outdir")
  echo "[eagle_multi] $name  path=$path  GPUs=$gpus  → $outdir"
  MODEL="$path" MODEL_NAME="$name" GPUS="$gpus" OUTPUT_DIR="$outdir" SKIP_ANALYZE=1 \
    bash scripts/eagle_g0.sh > "$OUTPUT_BASE/${name}.log" 2>&1 &
  pids+=($!)
done
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail + 1)); done
[[ $fail -gt 0 ]] && echo "[eagle_multi] WARNING: $fail worker(s) exited non-zero (see $OUTPUT_BASE/*.log)."

# optional per-dir LLM judge
USE_JUDGE_FLAG=()
if [[ "$JUDGE" == "1" || "$JUDGE" == "true" ]]; then
  for d in "${run_dirs[@]}"; do
    JFLAGS=(--run-dir "$d" --judge-api-url "$JUDGE_API_URL" --judge-model "$JUDGE_MODEL")
    [[ "$JUDGE_NO_THINK" == "1" || "$JUDGE_NO_THINK" == "true" ]] && JFLAGS+=(--judge-no-think)
    echo "[eagle_multi] judging $d ..."
    "${PY[@]}" -m baseline.g0.judge_g0 "${JFLAGS[@]}" || echo "[eagle_multi] judge failed for $d; using rule."
  done
  USE_JUDGE_FLAG=(--use-judge)
fi

echo "[eagle_multi] cross-model analysis ..."
"${PY[@]}" -m baseline.g0.analyze_eagle_g0 --run-dirs "${run_dirs[@]}" \
  --output-dir "$OUTPUT_BASE" "${USE_JUDGE_FLAG[@]}"
echo "[eagle_multi] done → $OUTPUT_BASE/eagle_report.md"
