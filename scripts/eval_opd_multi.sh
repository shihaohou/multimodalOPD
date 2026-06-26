#!/usr/bin/env bash
set -euo pipefail

# Multi-model x multi-benchmark fan-out over N GPUs for the OPD eval suite.
#
# ONE command runs the whole standard benchmark set. Benchmarks split by grading:
#   judged group  -> scripts/eval_opd.sh   (LLM judge; generate+judge can be split)
#                    mathvista mathverse mathvision MMMU mmmu_pro_10options
#                    mmmu-pro-vision mmstar hallusionbench
#   deterministic -> scripts/eval_vqa.sh   (official metric, NO judge, lmms-lab/* src)
#                    pope (F1), chartqa (relaxed acc), vqav2 (soft acc)
# The judged datasets live under DSROOT/<name>; the deterministic three are matched
# by name (case-insensitive) and routed to eval_vqa.sh with their own repos
# (POPE_REPO/CHARTQA_REPO/VQAV2_REPO — set these to local snapshot dirs on an offline
# box). Each (model, judged-dataset) is one eval_opd.sh job; each model's whole
# deterministic group is ONE eval_vqa.sh job (one engine load for all three).
#
# eval_opd.sh / eval_vqa.sh are reused verbatim, so their generation / judge knobs
# are the usual env vars (JUDGE_API_URL, JUDGE_MODEL, JUDGE_KEY_ENV / OPENAI_API_KEY,
# PASS_K, GEN_TEMPERATURE, POPE_REPO, ...) and are inherited by every job.
#
# PHASE (default 'all') splits generation from judging so you can saturate the GPUs
# first and stand up a judge model only afterwards:
#   PHASE=generate : judged datasets -> SAVE responses only (SKIP_JUDGE); the
#                    deterministic group runs FULLY here (it needs no judge). One job
#                    pinned per GPU, up to NGPU concurrent (free-card scheduler keeps
#                    every card busy). Multiple ckpts in MODELS just add jobs to the
#                    queue -> as a card frees it picks up the next (ckpt,benchmark).
#   PHASE=judge    : judge the SAVED judged responses (JUDGE_ONLY, no GPU), ONE at a
#                    time (each call drives JUDGE_WORKERS concurrent requests) so the
#                    log says exactly which benchmark is judging and one controlled
#                    pool can't swamp the judge. Deterministic group already done ->
#                    skipped. Reuse the SAME OUTPUT_ROOT.
#   PHASE=all      : generate + judge inline (judged) and generate + score (det), all
#                    on the GPU scheduler (the original behavior, one shot).
#
# Required:
#   MODELS="path1;path2;..."  or  "tag1=path1;tag2=path2;..."
#       Each entry is a checkpoint path, optionally "tag=path". With NO tag, the
#       output subdir + matrix label are derived from the path's last two components:
#       runs/<run>/checkpoint-93 -> "<run>/checkpoint-93". With a tag, the tag is used
#       (short, generic). Per-job logs go INSIDE each output folder
#       (<id>/<dataset>/<phase>.log) — there is no central logs/ dir.
# Benchmarks (default = full standard set; override to run a subset):
#   DSROOT=/abs/dir  DATASETS="name1 name2 ... pope chartqa vqav2"
#       judged names join as DSROOT/name (missing skipped); pope/chartqa/vqav2 route
#       to eval_vqa.sh regardless of DSROOT.
#   DATASET_DIRS="/abs/d1,/abs/d2,..."   explicit JUDGED dirs only (no det routing).
# Optional: NGPU (8), GPUS="0,1,2,..." (overrides NGPU), OUTPUT_ROOT, RUN_ID, DRYRUN=1.
#
# Two-phase example (saturate 8 cards on everything, judge with a model you deploy later):
#   # 1) generate judged + fully score the deterministic group, across all 8 GPUs:
#   OUTPUT_ROOT=eval_outputs/bench_myrun PHASE=generate NGPU=8 \
#   MODELS="runs/a/checkpoint-100;runs/b/checkpoint-100" \   # no tag -> folder = <run>/checkpoint-100
#   DSROOT=$D/zli12321 \
#   POPE_REPO=$D/lmms-lab/POPE CHARTQA_REPO=$D/lmms-lab/ChartQA VQAV2_REPO=$D/lmms-lab/VQAv2 \
#   PASS_K=1 GEN_TEMPERATURE=0 bash scripts/eval_opd_multi.sh
#   # 2) ...deploy your judge on the 8 GPUs (OpenAI-compatible server)...
#   # 3) judge the saved judged responses (SAME OUTPUT_ROOT/MODELS/DATASETS):
#   OUTPUT_ROOT=eval_outputs/bench_myrun PHASE=judge \
#   JUDGE_API_URL=http://127.0.0.1:8000/v1 JUDGE_MODEL=<served> OPENAI_API_KEY=x \
#   MODELS="runs/a/checkpoint-100;runs/b/checkpoint-100" \   # no tag -> folder = <run>/checkpoint-100
#   DSROOT=$D/zli12321 bash scripts/eval_opd_multi.sh
#
# One-shot example (judge already up, do everything inline on all 8 cards):
#   export JUDGE_API_URL=... JUDGE_MODEL=... OPENAI_API_KEY=...
#   MODELS="before=$M/Qwen3-VL-2B-Instruct;after=runs/opd_qwen3_8b_to_2b/checkpoint-65" \
#   DSROOT=$D/zli12321 NGPU=8 PASS_K=1 GEN_TEMPERATURE=0 bash scripts/eval_opd_multi.sh

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

# Default benchmark set (override DATASETS to run a subset). Judged group + the three
# deterministic benchmarks (matched by name and routed to eval_vqa.sh).
JUDGED_DEFAULT="mathvista mathverse mathvision MMMU mmmu_pro_10options mmmu-pro-vision mmstar hallusionbench"
DET_DEFAULT="pope chartqa vqav2"
DATASETS="${DATASETS:-$JUDGED_DEFAULT $DET_DEFAULT}"
# Names (case-insensitive) that are deterministic -> eval_vqa.sh, never the judge.
is_det() { case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in pope|chartqa|vqav2) return 0 ;; *) return 1 ;; esac; }

# A model's output-subdir + matrix label. Explicit "tag=path" in MODELS wins;
# otherwise it is derived from the checkpoint PATH (its last two components), so
# runs/<run>/checkpoint-93 -> "<run>/checkpoint-93" instead of a generic ckptA.
model_id() {  # path -> "<parent>/<base>" (or "<base>")
  local p="${1%/}" base parent
  base="$(basename "$p")"
  parent="$(basename "$(dirname "$p")")"
  if [[ -z "$parent" || "$parent" == "." || "$parent" == "/" || "$parent" == "$base" ]]; then
    printf '%s' "$base"
  else
    printf '%s/%s' "$parent" "$base"
  fi
}

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

# ---- classify requested benchmarks into judged dirs + deterministic group ----
JUDGED_DIRS=()      # absolute dirs for the judged group
DET_GROUP=()        # lowercase det benchmark names (pope/chartqa/vqav2)
if [[ -n "${DATASET_DIRS:-}" ]]; then
  IFS=',' read -r -a _dirs <<< "$DATASET_DIRS"
  for d in "${_dirs[@]}"; do [[ -n "$d" ]] && JUDGED_DIRS+=("$d"); done
else
  for name in $DATASETS; do
    if is_det "$name"; then
      DET_GROUP+=("$(printf '%s' "$name" | tr '[:upper:]' '[:lower:]')")
    else
      : "${DSROOT:?Set DSROOT (judged datasets join as DSROOT/name), or DATASET_DIRS.}"
      JUDGED_DIRS+=("$DSROOT/$name")
    fi
  done
fi
DET_CSV="$(IFS=,; printf '%s' "${DET_GROUP[*]:-}")"

# ---- job list: kind|tag|model|payload (judged dataset dir, or det csv group) ----
SPECS=()
JUDGED_N=0
IFS=';' read -r -a _models <<< "$MODELS"
for tm in "${_models[@]}"; do
  [[ -z "$tm" ]] && continue
  if [[ "$tm" == *=* ]]; then
    tag="${tm%%=*}"; model="${tm#*=}"      # explicit tag=path
  else
    model="$tm"; tag="$(model_id "$tm")"   # no tag -> derive id from the path
  fi
  for ddir in ${JUDGED_DIRS[@]+"${JUDGED_DIRS[@]}"}; do
    if [[ ! -e "$ddir" ]]; then
      echo "[skip] judged dataset not found, skipping for all models: $ddir" >&2
      continue
    fi
    SPECS+=("judged|${tag}|${model}|${ddir}")
    JUDGED_N=$((JUDGED_N + 1))
  done
  if [[ -n "$DET_CSV" ]]; then
    SPECS+=("det|${tag}|${model}|${DET_CSV}")
  fi
done
[[ ${#SPECS[@]} -eq 0 ]] && { echo "No jobs to run (check MODELS / datasets)."; exit 1; }

# Fail early if a judging phase has judged work but no judge configured (unless rule).
if [[ "$PHASE" != "generate" && "$JUDGED_N" -gt 0 && "${GRADER:-llm}" != "rule" ]]; then
  JUDGE_KEY_ENV="${JUDGE_KEY_ENV:-DEEPSEEK_API_KEY}"
  if [[ -z "${!JUDGE_KEY_ENV:-}" && -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: PHASE=$PHASE needs an LLM judge key in \$$JUDGE_KEY_ENV or \$OPENAI_API_KEY" >&2
    echo "  (point JUDGE_API_URL/JUDGE_MODEL at your deployed judge), or rule-grade with GRADER=rule." >&2
    exit 1
  fi
fi

DET_N=0; [[ -n "$DET_CSV" ]] && DET_N=$(( ${#_models[@]} ))
echo "Planned ${#SPECS[@]} jobs (${JUDGED_N} judged + ${DET_N} deterministic) over ${NGPU} GPU(s) [${CARDS[*]}] -> ${OUTPUT_ROOT}"
[[ -n "$DET_CSV" ]] && echo "  deterministic group: ${DET_CSV} (eval_vqa.sh, no judge)"
mkdir -p "$OUTPUT_ROOT"

run_job() {  # kind card tag model payload
  local kind="$1" card="$2" tag="$3" model="$4" payload="$5"
  local out logf safe
  # Each job's log lives INSIDE its own output folder (no central logs/ dir), named
  # by phase so generate.log and judge.log sit next to that job's responses/summary.
  if [[ "$kind" == "det" ]]; then
    out="$OUTPUT_ROOT/$tag/_vqa"
    mkdir -p "$out"
    logf="$out/${PHASE}.log"
    CUDA_VISIBLE_DEVICES="$card" MODEL_PATH="$model" MODEL_NAME="$tag" \
      BENCHMARKS="$payload" OUTPUT_DIR="$out" \
      bash scripts/eval_vqa.sh > "$logf" 2>&1
    return
  fi
  safe="$(basename "$payload")"; safe="${safe//[^A-Za-z0-9_.-]/_}"  # sanitize (no trailing _)
  out="$OUTPUT_ROOT/$tag/$safe"
  mkdir -p "$out"
  logf="$out/${PHASE}.log"
  local -a job_env=(MODEL_PATH="$model" MODEL_NAME="$tag" EVAL_DATASETS="$payload" OUTPUT_DIR="$out")
  case "$PHASE" in
    generate) job_env+=(CUDA_VISIBLE_DEVICES="$card" SKIP_JUDGE=true) ;;
    judge)    job_env+=(JUDGE_ONLY=true) ;;               # no GPU; judge server is remote
    *)        job_env+=(CUDA_VISIBLE_DEVICES="$card") ;;  # all: generate + judge inline
  esac
  env "${job_env[@]}" bash scripts/eval_opd.sh > "$logf" 2>&1
}

# Human label for a spec (kind|tag|model|payload).
spec_label() { # kind tag payload
  if [[ "$1" == "det" ]]; then echo "vqa[$3]"; else echo "$(basename "$3")"; fi
}

if [[ "$DRYRUN" == "1" ]]; then
  echo "--- DRYRUN: PHASE=$PHASE planned jobs ---"
  i=0
  for spec in "${SPECS[@]}"; do
    IFS='|' read -r kind tag model payload <<< "$spec"
    label="$(spec_label "$kind" "$tag" "$payload")"
    if [[ "$PHASE" == "judge" && "$kind" == "det" ]]; then
      echo "skip (det scored in generate) | ${tag} | ${label}"
    elif [[ "$PHASE" == "judge" ]]; then
      echo "judge (no GPU) | ${tag} | ${label} | $model"
    else
      echo "card ${CARDS[$((i % NGPU))]} | ${tag} | ${label} | $model"
      i=$((i + 1))
    fi
  done
  exit 0
fi

if [[ "$PHASE" == "judge" ]]; then
  # ---- judge phase: judged group only, ONE benchmark at a time (readable log) ----
  total=0; for spec in "${SPECS[@]}"; do [[ "${spec%%|*}" == "judged" ]] && total=$((total + 1)); done
  i=0
  for spec in "${SPECS[@]}"; do
    IFS='|' read -r kind tag model payload <<< "$spec"
    [[ "$kind" == "det" ]] && continue   # deterministic already scored in generate
    i=$((i + 1))
    echo "[judge ${i}/${total}] ${tag} | $(basename "$payload")"
    run_job "$kind" "" "$tag" "$model" "$payload" || echo "  (job exited non-zero; see log)"
  done
  [[ "$total" -eq 0 ]] && echo "(deterministic-only request: nothing to judge — scored in the generate phase)"
  echo "All ${total} judge jobs finished. Per-job log: <id>/<dataset>/judge.log under ${OUTPUT_ROOT}/"
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
    IFS='|' read -r kind tag model payload <<< "$spec"
    echo "[launch] card ${card} | ${tag} | $(spec_label "$kind" "$tag" "$payload")"
    run_job "$kind" "$card" "$tag" "$model" "$payload" &
    SLOT[$card]=$!
  done
  wait
  echo "All ${#SPECS[@]} jobs finished. Per-job log: <id>/<dataset>/${PHASE}.log under ${OUTPUT_ROOT}/"
fi

# Generate phase: judged group is generated-only (no scores yet) -> skip aggregation
# and print the exact judge-phase command. The deterministic group is already scored.
if [[ "$PHASE" == "generate" ]]; then
  echo
  echo "Phase 1 (generate) complete -> ${OUTPUT_ROOT}/<id>/  (logs: <id>/<dataset>/generate.log)"
  echo "  judged group: responses saved (awaiting judge);  deterministic group: already scored."
  echo "Deploy your judge, then judge the judged group with the SAME output root + MODELS:"
  echo "  OUTPUT_ROOT=${OUTPUT_ROOT} PHASE=judge${PASS_K:+ PASS_K=$PASS_K} \\"
  echo "  JUDGE_API_URL=<url> JUDGE_MODEL=<served> OPENAI_API_KEY=<key> \\"
  echo "  MODELS='${MODELS}' \\"
  if [[ -n "${DATASET_DIRS:-}" ]]; then
    echo "  DATASET_DIRS='${DATASET_DIRS}' \\"
  else
    echo "  DSROOT='${DSROOT:-}' DATASETS='${DATASETS}' \\"
  fi
  echo "  bash scripts/eval_opd_multi.sh"
  exit 0
fi

# ---- aggregate per (tag x benchmark) into a comparison matrix (stdlib only) ----
# Reads BOTH summary shapes: eval_opd (datasets/benchmarks lists -> pass_at_k) and
# eval_vqa (benchmarks dict -> official metric: pope F1, chartqa relaxed, vqav2 soft).
python3 - "$OUTPUT_ROOT" <<'PY' || echo "(aggregation skipped; read $OUTPUT_ROOT/*/*/summary.json)"
import glob, json, os, sys

root = sys.argv[1]
DET_METRIC = {  # det benchmark -> (metric key in summary['benchmarks'][name]['metrics'], row label)
    "pope": ("f1", "pope(F1)"),
    "chartqa": ("relaxed_accuracy", "chartqa(relax)"),
    "vqav2": ("vqa_accuracy", "vqav2(soft)"),
}
matrix, tags = {}, set()
# Recursive: a model's output folder may be nested (e.g. "<run>/checkpoint-93"), so
# summary.json can sit deeper than two levels under root.
for path in sorted(glob.glob(os.path.join(root, "**", "summary.json"), recursive=True)):
    try:
        summary = json.load(open(path))
    except Exception:
        continue
    tag = summary.get("model_name") or os.path.relpath(os.path.dirname(os.path.dirname(path)), root)
    tags.add(tag)
    bms = summary.get("benchmarks")
    if isinstance(bms, dict):  # eval_vqa deterministic summary (name -> {metrics: {...}})
        for name, score in bms.items():
            key, label = DET_METRIC.get(name, (None, name))
            val = ((score or {}).get("metrics") or {}).get(key) if key else None
            matrix.setdefault(label, {})[tag] = val
    else:  # eval_opd judged summary (datasets/benchmarks lists -> pass_at_k)
        for entry in (summary.get("datasets", []) + (bms or [])):
            name = os.path.basename(str(entry.get("dataset") or entry.get("benchmark") or "?").rstrip("/"))
            matrix.setdefault(name, {})[tag] = entry.get("pass_at_k")

tags = sorted(tags)
if matrix:
    width = max([len(n) for n in matrix] + [16])
    print(f"\n{'benchmark':<{width}} " + " ".join(f"{t:>10}" for t in tags))
    for name in sorted(matrix):
        row = matrix[name]
        cells = " ".join(
            (f"{row[t]:>10.4f}" if isinstance(row.get(t), (int, float)) else f"{'-':>10}")
            for t in tags
        )
        print(f"{name:<{width}} {cells}")
    print("\n(judged rows = pass@k via LLM judge; det rows = official metric, no judge)")
out = os.path.join(root, "matrix.json")
json.dump({"tags": tags, "matrix": matrix}, open(out, "w"), indent=2, ensure_ascii=False)
print(f"\nWrote {out}")
PY
