#!/usr/bin/env bash
set -uo pipefail

# EAGLE-G0 faithful-attribution diagnostic for ONE model. Data-parallel shards
# across the GPUs in $GPUS (one process/GPU), then (unless SKIP_ANALYZE) an
# optional LLM-judge pass + the EAGLE analysis. EAGLE is perturbation-based
# (~hundreds of forwards/sample), so keep LIMIT small and the image downsized
# (EAGLE_IMAGE_SIZE) — this is a small-budget diagnostic, not a full-8k pass.
#
# Required: MODEL. Optional MODEL_NAME, GPUS (this model's group), CONDITIONS.
#
#   MODEL=$M/CapCurriculum-8B GPUS=0,1 SUBSETS=gqa,openimages,vsr,textvqa \
#     LIMIT=50 CONDITIONS=plain,hint bash scripts/eagle_g0.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT_DIR"

: "${MODEL:?Set MODEL to a model dir / HF id (the model to diagnose).}"
MODEL_NAME="${MODEL_NAME:-$(basename "${MODEL%/}")}"
GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
OUTPUT_DIR="${OUTPUT_DIR:-eval_outputs/eagle_g0/$MODEL_NAME}"

export TRANSFORMERS_NO_TF=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
# Reduce allocator fragmentation (the salr1/EAGLE forwards alloc large transient
# attention tensors); the OOM message recommends this.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Python runner. If a venv is already active, use it DIRECTLY — `uv run` ignores a
# $VIRTUAL_ENV that doesn't match the project's .venv and bootstraps a fresh
# ./.venv on the (shared) repo, re-downloading everything. Override with
# PYRUN="uv run python" or PYRUN=/abs/path/to/python.
if [[ -n "${PYRUN:-}" ]]; then read -r -a PY <<< "$PYRUN";
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then PY=(python);
else PY=(uv run python); fi

DATASET="${DATASET:-peterant330/saliency-r1-8k}"
SPLIT="${SPLIT:-train}"
# Fail fast on the classic offline trap (HF id under HF_HUB_OFFLINE=1).
if [[ "$HF_HUB_OFFLINE" == "1" && "$DATASET" != /* && ! -d "$DATASET" ]]; then
  echo "[eagle_g0] HF_HUB_OFFLINE=1 but DATASET is not a local dir: '$DATASET'. Pass DATASET=\$D/saliency-r1-8k." >&2
  exit 1
fi
SUBSETS="${SUBSETS-gqa,openimages,vsr,textvqa}"   # single-dash: ""=all 10 subsets
# LIMIT = EFFECTIVE per-subset eval cap PER SHARD (calib hold-out already accounted
# for). Total ≈ (#GPUs in $GPUS) × LIMIT per subset.
LIMIT="${LIMIT:-60}"
CONDITIONS="${CONDITIONS:-plain,hint}"            # plain (no hint) and/or hint (silent GT-box)
HINT_MODE="${HINT_MODE:-generate}"                # generate | score_plain_y (OPD-faithful: rescore plain rollout)
MAX_BBOX_AREA="${MAX_BBOX_AREA:-0.5}"
SKIP_ANALYZE="${SKIP_ANALYZE:-}"

DTYPE="${DTYPE:-bfloat16}"
MAX_PIXELS="${MAX_PIXELS:-602112}"                 # grad-probe (full-res) cap
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-192}"
SEED="${SEED:-0}"

# EAGLE cost levers
EAGLE_IMAGE_SIZE="${EAGLE_IMAGE_SIZE:-448}"
N_REGIONS="${N_REGIONS:-49}"
SEARCH_SCOPE="${SEARCH_SCOPE:-8}"
PENDING_SAMPLES="${PENDING_SAMPLES:-4}"
UPDATE_STEP="${UPDATE_STEP:-10}"
EAGLE_BATCH_SIZE="${EAGLE_BATCH_SIZE:-8}"
REGION_MODE="${REGION_MODE:-auto}"                 # auto|slico|slic|grid
EAGLE_THRESHOLD="${EAGLE_THRESHOLD:-mean}"         # mean|top_frac — DO a sensitivity check both ways
EAGLE_TOP_FRAC="${EAGLE_TOP_FRAC:-0.25}"
ANSWER_TOKENS="${ANSWER_TOKENS:-8}"
GRAD_PROBES="${GRAD_PROBES:-1}"                    # 1 = also run LH+GLIMPSE (EAGLE-vs-LH); 0 = EAGLE only (cleaner n, faster)
SALR1="${SALR1:-1}"                                # 1 = also compute Saliency-R1 map (secondary baseline); 0 = off
SALR1_LAYERS="${SALR1_LAYERS:-all}"                # layers summed for SalR1 ('all' faithful; 'last8' = SalR1-lite)
SALR1_THINK_ROW_MODE="${SALR1_THINK_ROW_MODE:-state}"  # state | predictor (think-row ablation)
CALIB_LIMIT="${CALIB_LIMIT:-30}"
VIZ_PER_SUBSET="${VIZ_PER_SUBSET:-2}"

# judge (only used when SKIP_ANALYZE is empty AND JUDGE=1)
JUDGE="${JUDGE:-}"
JUDGE_API_URL="${JUDGE_API_URL:-http://localhost:8000/v1}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen3-30B-A3B}"
JUDGE_NO_THINK="${JUDGE_NO_THINK:-1}"

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
NUM_SHARDS="${#GPU_ARR[@]}"
mkdir -p "$OUTPUT_DIR"

common_flags=(
  --model "$MODEL" --model-name "$MODEL_NAME" --output-dir "$OUTPUT_DIR"
  --dataset "$DATASET" --split "$SPLIT" --subsets "$SUBSETS" --limit "$LIMIT"
  --conditions "$CONDITIONS" --hint-mode "$HINT_MODE" --max-bbox-area "$MAX_BBOX_AREA"
  --dtype "$DTYPE" --max-pixels "$MAX_PIXELS" --max-new-tokens "$MAX_NEW_TOKENS" --seed "$SEED"
  --eagle-image-size "$EAGLE_IMAGE_SIZE" --n-regions "$N_REGIONS"
  --search-scope "$SEARCH_SCOPE" --pending-samples "$PENDING_SAMPLES" --update-step "$UPDATE_STEP"
  --eagle-batch-size "$EAGLE_BATCH_SIZE" --region-mode "$REGION_MODE" --answer-tokens "$ANSWER_TOKENS"
  --eagle-threshold "$EAGLE_THRESHOLD" --eagle-top-frac "$EAGLE_TOP_FRAC"
  --salr1-layers "$SALR1_LAYERS" --salr1-think-row-mode "$SALR1_THINK_ROW_MODE"
  --calib-limit "$CALIB_LIMIT" --viz-per-subset "$VIZ_PER_SUBSET"
)
[[ "$GRAD_PROBES" == "1" || "$GRAD_PROBES" == "true" ]] || common_flags+=(--no-grad-probes)
[[ "$SALR1" == "1" || "$SALR1" == "true" ]] || common_flags+=(--no-salr1)

echo "[eagle_g0] model=$MODEL_NAME gpus=$GPUS shards=$NUM_SHARDS conditions=$CONDITIONS out=$OUTPUT_DIR"
pids=()
for ((i = 0; i < NUM_SHARDS; i++)); do
  gpu="${GPU_ARR[$i]}"
  CUDA_VISIBLE_DEVICES="$gpu" \
    "${PY[@]}" -m baseline.g0.run_eagle_g0 "${common_flags[@]}" \
      --num-shards "$NUM_SHARDS" --shard-index "$i" \
      > "$OUTPUT_DIR/shard${i}.log" 2>&1 &
  pids+=($!)
  echo "[eagle_g0]   shard $i → GPU $gpu (pid ${pids[-1]}, log $OUTPUT_DIR/shard${i}.log)"
done
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail + 1)); done
[[ $fail -gt 0 ]] && echo "[eagle_g0] WARNING: $fail/$NUM_SHARDS shards exited non-zero (see logs)."

if [[ -z "$SKIP_ANALYZE" ]]; then
  if [[ "$JUDGE" == "1" || "$JUDGE" == "true" ]]; then
    echo "[eagle_g0] LLM-judge ..."
    JFLAGS=(--run-dir "$OUTPUT_DIR" --judge-api-url "$JUDGE_API_URL" --judge-model "$JUDGE_MODEL")
    [[ "$JUDGE_NO_THINK" == "1" || "$JUDGE_NO_THINK" == "true" ]] && JFLAGS+=(--judge-no-think)
    "${PY[@]}" -m baseline.g0.judge_g0 "${JFLAGS[@]}" || echo "[eagle_g0] judge failed; continuing with rule grader."
    "${PY[@]}" -m baseline.g0.analyze_eagle_g0 --run-dirs "$OUTPUT_DIR" --use-judge
  else
    "${PY[@]}" -m baseline.g0.analyze_eagle_g0 --run-dirs "$OUTPUT_DIR"
  fi
  echo "[eagle_g0] done → $OUTPUT_DIR (eagle_report.md, eagle_analysis.json, viz/)"
fi
