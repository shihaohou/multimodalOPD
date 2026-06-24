#!/usr/bin/env bash
set -euo pipefail

# Multi-model x multi-dataset fan-out over N GPUs for the general OPD eval.
#
# Each (model, dataset) pair is ONE `scripts/eval_opd.sh` run pinned to ONE GPU;
# up to NGPU run concurrently (a free-card scheduler keeps every card busy). This
# saturates an 8-GPU box far better than running one job per model. eval_opd.sh is
# reused verbatim, so the LLM judge / pass@k / generation knobs are its usual env
# vars (JUDGE_API_URL, JUDGE_MODEL, JUDGE_KEY_ENV / OPENAI_API_KEY, PASS_K,
# GEN_TEMPERATURE, ...) and are inherited by every job.
#
# Required:
#   MODELS="tag1=path1;tag2=path2;..."   (tag becomes MODEL_NAME + output subdir)
# Datasets (pick one):
#   DSROOT=/abs/dir  DATASETS="name1 name2 ..."   (joined as DSROOT/name; missing skipped)
#   DATASET_DIRS="/abs/d1,/abs/d2,..."            (explicit full paths)
# Optional: NGPU (8), GPUS="0,1,2,..." (overrides NGPU), OUTPUT_ROOT, RUN_ID, DRYRUN=1.
#
# Example (3 models, Acc@1 greedy, all 8 cards):
#   export JUDGE_API_URL=... JUDGE_MODEL=... OPENAI_API_KEY=...
#   MODELS="before=$M/Qwen3-VL-2B-Instruct;after=runs/opd_qwen3_8b_to_2b/checkpoint-65;teacher=$M/Qwen3-VL-8B-Instruct" \
#   DSROOT=$D/zli12321 DATASETS="mm-vet MMMU mmmu_pro_10options mmmu-pro-vision mathvista mathverse MMSI realWorldQA" \
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
  CUDA_VISIBLE_DEVICES="$card" MODEL_PATH="$model" MODEL_NAME="$tag" \
    EVAL_DATASETS="$ddir" OUTPUT_DIR="$out" \
    bash scripts/eval_opd.sh > "$OUTPUT_ROOT/logs/${tag}_${safe}.log" 2>&1
}

if [[ "$DRYRUN" == "1" ]]; then
  echo "--- DRYRUN: planned jobs (card shown is indicative round-robin) ---"
  i=0
  for spec in "${SPECS[@]}"; do
    IFS='|' read -r tag model ddir <<< "$spec"
    echo "card ${CARDS[$((i % NGPU))]} | ${tag} | $(basename "$ddir") | $model"
    i=$((i + 1))
  done
  exit 0
fi

# ---- scheduler: <=NGPU concurrent, one job pinned per free card ----
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
