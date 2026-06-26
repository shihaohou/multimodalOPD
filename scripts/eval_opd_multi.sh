#!/usr/bin/env bash
set -euo pipefail

# Multi-model x multi-dataset fan-out over N GPUs for the general OPD eval.
#
# Each (model, dataset) pair is ONE `scripts/eval_opd.sh` run. eval_opd.sh is reused
# verbatim, so the LLM judge / pass@k / generation knobs are its usual env vars
# (JUDGE_API_URL, JUDGE_MODEL, JUDGE_KEY_ENV / OPENAI_API_KEY, PASS_K,
# GEN_TEMPERATURE, ...) and are inherited by every job.
#
# PHASE (default 'all') splits generation from judging so you can saturate the GPUs
# first and stand up a judge model only afterwards:
#   PHASE=generate : sample + SAVE responses only (SKIP_JUDGE), one job pinned per
#                    GPU, up to NGPU concurrent (free-card scheduler keeps every card
#                    busy). NO judge needed. Multiple ckpts in MODELS just add jobs to
#                    the queue -> as a card frees it picks up the next (ckpt,dataset).
#   PHASE=judge    : judge the SAVED responses (JUDGE_ONLY, no GPU), ONE benchmark at
#                    a time (each call already drives JUDGE_WORKERS concurrent requests)
#                    so the log says exactly which benchmark is judging and a single
#                    controlled pool can't swamp the judge. Reuse the SAME OUTPUT_ROOT.
#   PHASE=all      : generate + judge inline per job (the original behavior).
#
# Required:
#   MODELS="tag1=path1;tag2=path2;..."   (tag becomes MODEL_NAME + output subdir)
# Datasets (pick one):
#   DSROOT=/abs/dir  DATASETS="name1 name2 ..."   (joined as DSROOT/name; missing skipped)
#   DATASET_DIRS="/abs/d1,/abs/d2,..."            (explicit full paths)
# Optional: NGPU (8), GPUS="0,1,2,..." (overrides NGPU), OUTPUT_ROOT, RUN_ID, DRYRUN=1.
#
# Two-phase example (saturate 8 cards, then judge with a model you deploy later):
#   # 1) generate everything across all 8 GPUs (multiple ckpts run back-to-back):
#   OUTPUT_ROOT=eval_outputs/bench_myrun PHASE=generate NGPU=8 \
#   MODELS="ckptA=runs/a/checkpoint-100;ckptB=runs/b/checkpoint-100" \
#   DSROOT=$D/zli12321 DATASETS="mathvista mathverse MMMU mmmu_pro_10options mmstar hallusionbench" \
#   PASS_K=1 GEN_TEMPERATURE=0 bash scripts/eval_opd_multi.sh
#   # 2) ...deploy your judge on the 8 GPUs (OpenAI-compatible server)...
#   # 3) judge the saved responses (SAME OUTPUT_ROOT/MODELS/DATASETS), clean per-bench log:
#   OUTPUT_ROOT=eval_outputs/bench_myrun PHASE=judge \
#   JUDGE_API_URL=http://127.0.0.1:8000/v1 JUDGE_MODEL=<served> OPENAI_API_KEY=x \
#   MODELS="ckptA=runs/a/checkpoint-100;ckptB=runs/b/checkpoint-100" \
#   DSROOT=$D/zli12321 DATASETS="mathvista mathverse MMMU mmmu_pro_10options mmstar hallusionbench" \
#   bash scripts/eval_opd_multi.sh
#
# One-shot example (generate + judge inline, all 8 cards):
#   export JUDGE_API_URL=... JUDGE_MODEL=... OPENAI_API_KEY=...
#   MODELS="before=$M/Qwen3-VL-2B-Instruct;after=runs/opd_qwen3_8b_to_2b/checkpoint-65" \
#   DSROOT=$D/zli12321 DATASETS="mm-vet MMMU mathvista mathverse MMSI realWorldQA" \
#   PASS_K=1 GEN_TEMPERATURE=0 NGPU=8 bash scripts/eval_opd_multi.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT_DIR"

: "${MODELS:?Set MODELS='tag=path;tag=path;...'}"
NGPU="${NGPU:-8}"
if [[ -n "${GPUS:-}" ]]; then
  IFS=',' read -r -a CARDS <<< "$GPUS"
else
  CARDS=(); for ((c = 0; c < NGPU; c++)); do CARDS+=("$c"); done
fi
NGPU=${#CARDS[@]}

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-eval_outputs/bench_${RUN_ID}}"
DRYRUN="${DRYRUN:-0}"
PHASE="${PHASE:-all}"   # generate | judge | all

case "$PHASE" in
  generate|judge|all) ;;
  *) echo "ERROR: PHASE must be generate|judge|all (got '$PHASE')." >&2; exit 1 ;;
esac

# The judge phase reads what the generate phase saved -> it MUST reuse the same dir.
# OUTPUT_ROOT defaults to a fresh timestamp, so pin it (or RUN_ID) across both phases.
if [[ "$PHASE" == "judge" && ! -d "$OUTPUT_ROOT" ]]; then
  echo "ERROR: PHASE=judge needs the responses from a prior PHASE=generate run, but" >&2
  echo "  OUTPUT_ROOT='$OUTPUT_ROOT' does not exist. Set OUTPUT_ROOT (or RUN_ID) to the" >&2
  echo "  directory your generate phase printed." >&2
  exit 1
fi

# Fail early if the judging phases have no judge configured (unless rule-grading).
if [[ "$PHASE" != "generate" && "${GRADER:-llm}" != "rule" ]]; then
  JUDGE_KEY_ENV="${JUDGE_KEY_ENV:-DEEPSEEK_API_KEY}"
  if [[ -z "${!JUDGE_KEY_ENV:-}" && -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: PHASE=$PHASE needs an LLM judge key in \$$JUDGE_KEY_ENV or \$OPENAI_API_KEY" >&2
    echo "  (point JUDGE_API_URL/JUDGE_MODEL at your deployed judge), or rule-grade with GRADER=rule." >&2
    exit 1
  fi
fi

# ---- dataset dir list ----
DATASET_LIST=()
if [[ -n "${DATASET_DIRS:-}" ]]; then
  IFS=',' read -r -a _dirs <<< "$DATASET_DIRS"
  for d in "${_dirs[@]}"; do [[ -n "$d" ]] && DATASET_LIST+=("$d"); done
else
  : "${DSROOT:?Set DSROOT (with DATASETS), or DATASET_DIRS.}"
  : "${DATASETS:?Set DATASETS='name1 name2 ...' (subdirs under DSROOT).}"
  for name in $DATASETS; do DATASET_LIST+=("$DSROOT/$name"); done
fi

# ---- job list (model x dataset), skipping datasets that are absent on disk ----
SPECS=()
IFS=';' read -r -a _models <<< "$MODELS"
for tm in "${_models[@]}"; do
  [[ -z "$tm" ]] && continue
  tag="${tm%%=*}"
  model="${tm#*=}"
  for ddir in "${DATASET_LIST[@]}"; do
    if [[ ! -e "$ddir" ]]; then
      echo "[skip] dataset not found, skipping for all models: $ddir" >&2
      continue
    fi
    SPECS+=("${tag}|${model}|${ddir}")
  done
done
[[ ${#SPECS[@]} -eq 0 ]] && { echo "No jobs to run (check MODELS / datasets)."; exit 1; }

echo "Planned ${#SPECS[@]} jobs over ${NGPU} GPU(s) [${CARDS[*]}] -> ${OUTPUT_ROOT}"
mkdir -p "$OUTPUT_ROOT/logs"

run_job() {  # card tag model ddir
  local card="$1" tag="$2" model="$3" ddir="$4"
  local safe out
  safe="$(basename "$ddir" | tr -c 'A-Za-z0-9_.-' '_')"
  out="$OUTPUT_ROOT/$tag/$safe"
  local -a job_env=(MODEL_PATH="$model" MODEL_NAME="$tag" EVAL_DATASETS="$ddir" OUTPUT_DIR="$out")
  case "$PHASE" in
    generate) job_env+=(CUDA_VISIBLE_DEVICES="$card" SKIP_JUDGE=true) ;;
    judge)    job_env+=(JUDGE_ONLY=true) ;;               # no GPU; judge server is remote
    *)        job_env+=(CUDA_VISIBLE_DEVICES="$card") ;;  # all: generate + judge inline
  esac
  env "${job_env[@]}" bash scripts/eval_opd.sh > "$OUTPUT_ROOT/logs/${PHASE}_${tag}_${safe}.log" 2>&1
}

if [[ "$DRYRUN" == "1" ]]; then
  echo "--- DRYRUN: PHASE=$PHASE planned jobs ---"
  i=0
  for spec in "${SPECS[@]}"; do
    IFS='|' read -r tag model ddir <<< "$spec"
    if [[ "$PHASE" == "judge" ]]; then
      echo "judge (no GPU) | ${tag} | $(basename "$ddir") | $model"
    else
      echo "card ${CARDS[$((i % NGPU))]} | ${tag} | $(basename "$ddir") | $model"
    fi
    i=$((i + 1))
  done
  exit 0
fi

if [[ "$PHASE" == "judge" ]]; then
  # ---- judge phase: network-bound, run ONE benchmark at a time (readable log) ----
  i=0
  for spec in "${SPECS[@]}"; do
    IFS='|' read -r tag model ddir <<< "$spec"
    i=$((i + 1))
    echo "[judge ${i}/${#SPECS[@]}] ${tag} | $(basename "$ddir")"
    run_job "" "$tag" "$model" "$ddir" || echo "  (job exited non-zero; see log)"
  done
  echo "All ${#SPECS[@]} judge jobs finished. Logs in ${OUTPUT_ROOT}/logs/"
else
  # ---- generate / all: <=NGPU concurrent, one job pinned per free card ----
  declare -A SLOT  # SLOT[card]=pid of the job currently on that card
  for spec in "${SPECS[@]}"; do
    card=""
    while [[ -z "$card" ]]; do
      for c in "${CARDS[@]}"; do
        pid="${SLOT[$c]:-}"
        if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
          card="$c"
          break
        fi
      done
      # All cards busy: block until any job exits (tolerate its exit code).
      [[ -z "$card" ]] && { wait -n 2>/dev/null || sleep 3; }
    done
    IFS='|' read -r tag model ddir <<< "$spec"
    echo "[launch] card ${card} | ${tag} | $(basename "$ddir")"
    run_job "$card" "$tag" "$model" "$ddir" &
    SLOT[$card]=$!
  done
  wait
  echo "All ${#SPECS[@]} jobs finished. Logs in ${OUTPUT_ROOT}/logs/"
fi

# Generate phase saves responses only (no scores yet) -> skip aggregation, print the
# exact judge-phase command to run once a judge is deployed.
if [[ "$PHASE" == "generate" ]]; then
  echo
  echo "Phase 1 (generate) complete -> responses under ${OUTPUT_ROOT}/<tag>/<dataset>/responses/"
  echo "Deploy your judge, then judge with the SAME output root:"
  echo "  OUTPUT_ROOT=${OUTPUT_ROOT} PHASE=judge \\"
  echo "  JUDGE_API_URL=<url> JUDGE_MODEL=<served> OPENAI_API_KEY=<key> \\"
  echo "  MODELS='${MODELS}' \\"
  if [[ -n "${DATASET_DIRS:-}" ]]; then
    echo "  DATASET_DIRS='${DATASET_DIRS}' \\"
  else
    echo "  DSROOT='${DSROOT}' DATASETS='${DATASETS}' \\"
  fi
  echo "  bash scripts/eval_opd_multi.sh"
  exit 0
fi

# ---- aggregate per (tag x dataset) into a comparison matrix (stdlib only) ----
python3 - "$OUTPUT_ROOT" <<'PY' || echo "(aggregation skipped; read $OUTPUT_ROOT/*/*/summary.json)"
import glob, json, os, sys

root = sys.argv[1]
matrix, tags = {}, set()
for path in sorted(glob.glob(os.path.join(root, "*", "*", "summary.json"))):
    try:
        summary = json.load(open(path))
    except Exception:
        continue
    tag = summary.get("model_name") or os.path.basename(os.path.dirname(os.path.dirname(path)))
    tags.add(tag)
    for entry in (summary.get("datasets", []) + summary.get("benchmarks", [])):
        name = os.path.basename(str(entry.get("dataset") or entry.get("benchmark") or "?").rstrip("/"))
        matrix.setdefault(name, {})[tag] = entry.get("pass_at_k")

tags = sorted(tags)
if matrix:
    width = max([len(n) for n in matrix] + [16])
    print(f"\n{'dataset':<{width}} " + " ".join(f"{t:>10}" for t in tags))
    for name in sorted(matrix):
        row = matrix[name]
        cells = " ".join(
            (f"{row[t]:>10.4f}" if isinstance(row.get(t), (int, float)) else f"{'-':>10}")
            for t in tags
        )
        print(f"{name:<{width}} {cells}")
out = os.path.join(root, "matrix.json")
json.dump({"tags": tags, "matrix": matrix}, open(out, "w"), indent=2, ensure_ascii=False)
print(f"\nWrote {out}")
PY
