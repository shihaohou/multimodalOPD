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
#   V*Bench       -> scripts/eval_vstar.sh (official MCQ acc, NO judge, $D/VStarBench)
#                    vstar (overall + per-category accuracy)
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
# Shared-disk defaults (same paths on every machine; override via env if a box differs)
#   D=/.../datasets  M=/.../models  DSROOT=$D/zli12321
#   POPE_REPO=$D/POPE  CHARTQA_REPO=$D/ChartQA  VQAV2_REPO=$D/VQAv2
#   -> the common run needs only MODELS + OUTPUT_ROOT + PHASE.
#
# Required:
#   MODELS="path1;path2;..."  or  "tag1=path1;tag2=path2;..."
#       Each entry is a checkpoint path, optionally "tag=path"; a BARE name resolves
#       against $M ("Qwen3-VL-8B-Instruct" -> $M/Qwen3-VL-8B-Instruct). With NO tag, the
#       output subdir + matrix label are derived from the path: the basename
#       (.../models/CapCurriculum-8B -> "CapCurriculum-8B"), or "<run>/checkpoint-N"
#       when the basename is a generic checkpoint dir. With a tag, the tag is used.
#       Per-job logs go INSIDE each output folder (<id>/<dataset>/<phase>.log) —
#       there is no central logs/ dir.
# Benchmarks (default = full standard set; override DATASETS to run a subset):
#   DATASETS="name1 name2 ... pope chartqa vqav2"  (judged names join as DSROOT/name,
#       missing skipped; pope/chartqa/vqav2 route to eval_vqa.sh via *_REPO).
#   DATASET_DIRS="/abs/d1,/abs/d2,..."   explicit JUDGED dirs only (no det routing).
# Optional: NGPU (8), GPUS="0,1,2,..." (overrides NGPU), OUTPUT_ROOT, RUN_ID, DRYRUN=1,
#   RESUME=1 (rerun the same command -> only the failed/missing jobs are redone),
#   LAUNCH_STAGGER=30 (seconds between job launches; spreads the CPU-bound image
#   preprocessing so 8 jobs don't thrash the CPU while the GPUs idle),
#   DET_SPLIT (default ON: pope/chartqa/vqav2 run as 3 separate scheduler jobs -> 3
#   cards; set DET_SPLIT=0 to pack all three onto one card with a single model load).
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
EVAL_BACKEND="${EVAL_BACKEND:-lmms_fast}"  # lmms_fast | opd
# RESUME -> skip jobs whose output is already complete, so rerunning the SAME command
# only redoes the failed/missing ones (generate/all: needs summary.json; judge: needs
# judgments/*.jsonl). Accept 1/true/yes/on.
case "$(printf '%s' "${RESUME:-}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on) RESUME=true ;;
  *) RESUME=false ;;
esac
# Seconds to wait between launching jobs. The per-dataset "Adding requests" phase is
# CPU-bound (VL image preprocessing); launching all NGPU jobs at once makes them
# thrash the CPU while the GPUs idle. A stagger (e.g. 20-40) offsets their CPU-heavy
# phases against each other's GPU-heavy phases -> better CPU/GPU overlap. 0 = off.
LAUNCH_STAGGER="${LAUNCH_STAGGER:-0}"
# DET_SPLIT (DEFAULT on) -> run each deterministic benchmark (pope/chartqa/vqav2) as
# its OWN job (its own card via the scheduler), so they fan out across free cards
# instead of one job per ckpt running all three in turn on a single card. Costs one
# model reload per benchmark. Opt out with DET_SPLIT=0 to pack all three onto one card.
case "$(printf '%s' "${DET_SPLIT:-1}" | tr '[:upper:]' '[:lower:]')" in
  0|false|no|off) DET_SPLIT=false ;;
  *) DET_SPLIT=true ;;
esac

# Default benchmark set (override DATASETS to run a subset). Judged group + the three
# deterministic benchmarks (routed to eval_vqa.sh) + V*Bench (routed to eval_vstar.sh).
# V*Bench needs the VStarBench snapshot ($D/VStarBench, or VSTAR_REPO); if that dir is
# absent the vstar job is skipped with a message, so it's safe to keep in the default.
JUDGED_DEFAULT="mathvista mathverse mathvision MMMU mmmu_pro_10options mmmu-pro-vision mmstar hallusionbench"
DET_DEFAULT="pope chartqa vqav2"
VSTAR_DEFAULT="vstar"
LMMS_FAST_DEFAULT="mathvista mathverse mathvision MMMU MMMU-Pro MMStar HallusionBench POPE ChartQA vstar HRBench4K HRBench8K MME-RealWorld-Lite"
if [[ -z "${DATASETS:-}" ]]; then
  if [[ "$EVAL_BACKEND" == "lmms_fast" ]]; then
    DATASETS="$LMMS_FAST_DEFAULT"
  else
    DATASETS="$JUDGED_DEFAULT $DET_DEFAULT $VSTAR_DEFAULT"
  fi
fi

# Shared-disk layout (identical on every machine -> sane defaults so the common
# command needs only MODELS/OUTPUT_ROOT/PHASE). Override any of these via env on a box
# whose paths differ. D = datasets root, M = models root.
D="${D:-/home/web_server/antispam/project/houshihao/datasets}"
M="${M:-/home/web_server/antispam/project/houshihao/models}"
DSROOT="${DSROOT:-$D/zli12321}"
export POPE_REPO="${POPE_REPO:-$D/POPE}"        # passed through to eval_vqa.sh (det group)
export CHARTQA_REPO="${CHARTQA_REPO:-$D/ChartQA}"
export VQAV2_REPO="${VQAV2_REPO:-$D/VQAv2}"
export VSTAR_REPO="${VSTAR_REPO:-$D/VStarBench}"  # passed to eval_vstar.sh (det-style MCQ, no judge)
# Names (case-insensitive) that are deterministic -> eval_vqa.sh, never the judge.
is_det() { case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in pope|chartqa|vqav2) return 0 ;; *) return 1 ;; esac; }
# V*Bench: deterministic MCQ too, but its OWN evaluator (eval_vstar.sh), never the judge.
is_vstar() { case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in vstar|vstar_bench|vstarbench|v-star) return 0 ;; *) return 1 ;; esac; }

# A model's output-subdir + matrix label. Explicit "tag=path" in MODELS wins;
# otherwise it is derived from the PATH. Usually the basename is descriptive enough
# (.../models/CapCurriculum-8B -> "CapCurriculum-8B"); only when the basename is a
# generic checkpoint dir name (checkpoint-93, global_step500, ...) — ambiguous on its
# own — do we prepend the run dir (runs/<run>/checkpoint-93 -> "<run>/checkpoint-93").
model_id() {  # path -> "<base>" (or "<run>/<base>" for a checkpoint dir)
  local p="${1%/}" base parent
  base="$(basename "$p")"
  case "$base" in
    checkpoint-*|checkpoint_*|global_step*|global-step*|step-*|step_*|epoch-*|epoch_*)
      parent="$(basename "$(dirname "$p")")"
      if [[ -n "$parent" && "$parent" != "." && "$parent" != "/" && "$parent" != "$base" ]]; then
        printf '%s/%s' "$parent" "$base"
      else
        printf '%s' "$base"
      fi
      ;;
    *) printf '%s' "$base" ;;  # already a descriptive model name
  esac
}

case "$PHASE" in
  generate|judge|all) ;;
  *) echo "ERROR: PHASE must be generate|judge|all (got '$PHASE')." >&2; exit 1 ;;
esac
case "$EVAL_BACKEND" in
  opd|lmms_fast) ;;
  *) echo "ERROR: EVAL_BACKEND must be opd|lmms_fast (got '$EVAL_BACKEND')." >&2; exit 1 ;;
esac

if [[ "$EVAL_BACKEND" == "lmms_fast" ]]; then
  LMMS_DATASET_ITEMS=()
  while IFS= read -r bench; do
    [[ -n "$bench" ]] && LMMS_DATASET_ITEMS+=("$bench")
  done < <(python3 baseline/eval/lmms_eval_bridge.py benchmarks --benchmarks "$DATASETS")

  LMMS_SPECS=()
  IFS=';' read -r -a _models <<< "$MODELS"
  for tm in "${_models[@]}"; do
    [[ -z "$tm" ]] && continue
    if [[ "$tm" == *=* ]]; then
      tag="${tm%%=*}"; model="${tm#*=}"
    else
      tag=""; model="$tm"
    fi
    [[ "$model" != /* && -n "${M:-}" && -e "$M/$model" ]] && model="$M/$model"
    [[ -z "$tag" ]] && tag="$(model_id "$model")"
    for bench in "${LMMS_DATASET_ITEMS[@]}"; do
      [[ -n "$bench" ]] && LMMS_SPECS+=("lmms_fast|${tag}|${model}|${bench}")
    done
  done
  [[ ${#LMMS_SPECS[@]} -eq 0 ]] && { echo "No lmms_fast jobs to run (check MODELS / DATASETS)."; exit 1; }

  mkdir -p "$OUTPUT_ROOT"
  lmms_job_out() {
    local s; s="$3"; s="${s//[^A-Za-z0-9_.-]/_}"
    printf '%s/%s/_lmms_%s' "$OUTPUT_ROOT" "$2" "$s"
  }
  lmms_job_done() {
    case "$PHASE" in
      generate) [[ -f "$1/generation_complete.json" ]] ;;
      judge|all) [[ -f "$1/summary.json" ]] ;;
      *) return 1 ;;
    esac
  }
  lmms_label() { echo "$1"; }
  run_lmms_job() {
    local card="$1" tag="$2" model="$3" bench="$4" out logf
    out="$(lmms_job_out lmms_fast "$tag" "$bench")"
    mkdir -p "$out"
    logf="$out/${PHASE}.log"
    CUDA_VISIBLE_DEVICES="$card" MODEL_PATH="$model" MODEL_NAME="$tag" \
      DATASETS="$bench" OUTPUT_DIR="$out" LMMS_PHASE="$PHASE" \
      bash scripts/eval_lmms_aligned.sh > "$logf" 2>&1
  }

  if [[ "$PHASE" == "judge" ]]; then
    echo "Planned ${#LMMS_SPECS[@]} lmms_fast score jobs (no local eval GPU) -> ${OUTPUT_ROOT}"
  else
    echo "Planned ${#LMMS_SPECS[@]} lmms_fast jobs over ${NGPU} GPU(s) [${CARDS[*]}] -> ${OUTPUT_ROOT}"
  fi
  echo "  DATASETS=${DATASETS}"
  echo "  PROMPT_MODE=${PROMPT_MODE:-lmms}  LMMS_EVAL_DIR=${LMMS_EVAL_DIR:-/Users/houshihao/project/code/lmms-eval-main}"
  [[ "$PHASE" != "judge" ]] && echo "  BUILD_WORKERS=${BUILD_WORKERS:-1}"
  [[ "$PHASE" != "judge" && -n "${BATCH_SIZE:-}" ]] && echo "  BATCH_SIZE=${BATCH_SIZE}"
  [[ "$PHASE" != "judge" && -n "${VLLM_MAX_NUM_SEQS:-}" ]] && echo "  VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS}"
  [[ "$PHASE" != "judge" && -n "${VLLM_MAX_NUM_BATCHED_TOKENS:-}" ]] && echo "  VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS}"
  [[ "$PHASE" != "judge" && -n "${VLLM_MM_PROCESSOR_CACHE_GB:-}" ]] && echo "  VLLM_MM_PROCESSOR_CACHE_GB=${VLLM_MM_PROCESSOR_CACHE_GB}"
  [[ "$PHASE" != "judge" && -n "${VLLM_MM_PROCESSOR_KWARGS:-}" ]] && echo "  VLLM_MM_PROCESSOR_KWARGS=${VLLM_MM_PROCESSOR_KWARGS}"
  [[ "$PHASE" == "judge" ]] && echo "  JUDGE_WORKERS=${JUDGE_WORKERS:-1}"
  [[ "$PHASE" == "judge" ]] && echo "  JUDGE_MODEL=${JUDGE_MODEL:-${MODEL_VERSION:-}}"
  [[ "$PHASE" == "judge" && -n "${JUDGE_EXTRA_BODY:-}" ]] && echo "  JUDGE_EXTRA_BODY=${JUDGE_EXTRA_BODY}"
  [[ "$RESUME" == "true" ]] && echo "  RESUME=true: jobs already complete will be skipped"

  if [[ "$DRYRUN" == "1" ]]; then
    echo "--- DRYRUN: EVAL_BACKEND=lmms_fast planned jobs ---"
    i=0
    for spec in "${LMMS_SPECS[@]}"; do
      IFS='|' read -r _kind tag model bench <<< "$spec"
      out="$(lmms_job_out lmms_fast "$tag" "$bench")"
      if [[ "$RESUME" == "true" ]] && lmms_job_done "$out"; then
        echo "skip done (RESUME) | ${tag} | ${bench}"
      elif [[ "$PHASE" == "judge" ]]; then
        echo "score | ${tag} | ${bench} | $model"
      else
        echo "card ${CARDS[$((i % NGPU))]} | ${tag} | ${bench} | $model"
        i=$((i + 1))
      fi
    done
    exit 0
  fi

  if [[ "$PHASE" == "judge" ]]; then
    for spec in "${LMMS_SPECS[@]}"; do
      IFS='|' read -r _kind tag model bench <<< "$spec"
      out="$(lmms_job_out lmms_fast "$tag" "$bench")"
      if [[ "$RESUME" == "true" ]] && lmms_job_done "$out"; then
        echo "[skip done] ${tag} | ${bench}  (RESUME)"
        continue
      fi
      echo "[score] ${tag} | ${bench}"
      run_lmms_job "" "$tag" "$model" "$bench"
    done
    echo "All ${#LMMS_SPECS[@]} lmms_fast score jobs finished. Per-job log: <id>/_lmms_<bench>/judge.log under ${OUTPUT_ROOT}/"
    python3 baseline/eval/make_report.py "$OUTPUT_ROOT" \
      || echo "(report skipped; rerun: python3 baseline/eval/make_report.py $OUTPUT_ROOT)"
    exit 0
  fi

  declare -A SLOT
  for spec in "${LMMS_SPECS[@]}"; do
    IFS='|' read -r _kind tag model bench <<< "$spec"
    out="$(lmms_job_out lmms_fast "$tag" "$bench")"
    if [[ "$RESUME" == "true" ]] && lmms_job_done "$out"; then
      echo "[skip done] ${tag} | ${bench}  (RESUME)"
      continue
    fi
    card=""
    while [[ -z "$card" ]]; do
      for c in "${CARDS[@]}"; do
        pid="${SLOT[$c]:-}"
        if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
          card="$c"
          break
        fi
      done
      [[ -z "$card" ]] && { wait -n 2>/dev/null || sleep 3; }
    done
    echo "[launch] card ${card} | ${tag} | ${bench}"
    run_lmms_job "$card" "$tag" "$model" "$bench" &
    SLOT[$card]=$!
    [[ "$LAUNCH_STAGGER" != "0" ]] && sleep "$LAUNCH_STAGGER"
  done
  wait
  echo "All ${#LMMS_SPECS[@]} lmms_fast jobs finished. Per-job log: <id>/_lmms_<bench>/${PHASE}.log under ${OUTPUT_ROOT}/"
  if [[ "$PHASE" == "all" ]]; then
    python3 baseline/eval/make_report.py "$OUTPUT_ROOT" \
      || echo "(report skipped; rerun: python3 baseline/eval/make_report.py $OUTPUT_ROOT)"
  fi
  exit 0
fi

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
VSTAR_REQUESTED=false   # V*Bench requested? routes to eval_vstar.sh (det-style, no judge)
if [[ -n "${DATASET_DIRS:-}" ]]; then
  IFS=',' read -r -a _dirs <<< "$DATASET_DIRS"
  for d in "${_dirs[@]}"; do [[ -n "$d" ]] && JUDGED_DIRS+=("$d"); done
else
  for name in $DATASETS; do
    if is_det "$name"; then
      lc="$(printf '%s' "$name" | tr '[:upper:]' '[:lower:]')"
      case "$lc" in
        pope)    repo="${POPE_REPO:-lmms-lab/POPE}" ;;
        chartqa) repo="${CHARTQA_REPO:-lmms-lab/ChartQA}" ;;
        vqav2)   repo="${VQAV2_REPO:-lmms-lab/VQAv2}" ;;
        *)       repo="" ;;
      esac
      # A local-looking repo path (/, ./, ../, ~) that doesn't exist -> skip with a
      # clear message instead of loading the model just to crash in the loader. A bare
      # hub id (e.g. lmms-lab/POPE) isn't path-checked — it's left for eval_vqa to fetch.
      case "$repo" in
        /*|./*|../*|"~"*)
          if [[ ! -e "$repo" ]]; then
            echo "[skip] deterministic '$lc' repo not found: $repo (check POPE_REPO/CHARTQA_REPO/VQAV2_REPO)" >&2
            continue
          fi ;;
      esac
      DET_GROUP+=("$lc")
    elif is_vstar "$name"; then
      # V*Bench: like the det group it needs no judge, but routes to eval_vstar.sh. A
      # local-looking VSTAR_REPO that doesn't exist -> skip with a clear message (a bare
      # hub id like craigwu/vstar_bench isn't path-checked; eval_vstar.sh fetches it).
      case "$VSTAR_REPO" in
        /*|./*|../*|"~"*)
          if [[ ! -e "$VSTAR_REPO" ]]; then
            echo "[skip] vstar repo not found: $VSTAR_REPO (set VSTAR_REPO to a snapshot dir, or a HF id)" >&2
            continue
          fi ;;
      esac
      VSTAR_REQUESTED=true
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
DET_N=0
IFS=';' read -r -a _models <<< "$MODELS"
for tm in "${_models[@]}"; do
  [[ -z "$tm" ]] && continue
  if [[ "$tm" == *=* ]]; then
    tag="${tm%%=*}"; model="${tm#*=}"      # explicit tag=path
  else
    tag=""; model="$tm"
  fi
  # A bare (non-absolute) model name resolves against the shared models dir $M, so
  # MODELS="Qwen3-VL-8B-Instruct;CapCurriculum-8B" works without full paths.
  [[ "$model" != /* && -n "${M:-}" && -e "$M/$model" ]] && model="$M/$model"
  [[ -z "$tag" ]] && tag="$(model_id "$model")"   # no tag -> derive id from the path
  for ddir in ${JUDGED_DIRS[@]+"${JUDGED_DIRS[@]}"}; do
    if [[ ! -e "$ddir" ]]; then
      echo "[skip] judged dataset not found, skipping for all models: $ddir" >&2
      continue
    fi
    SPECS+=("judged|${tag}|${model}|${ddir}")
    JUDGED_N=$((JUDGED_N + 1))
  done
  if [[ "$DET_SPLIT" == "true" ]]; then
    for b in ${DET_GROUP[@]+"${DET_GROUP[@]}"}; do   # one job per benchmark -> own card
      SPECS+=("det|${tag}|${model}|${b}")
      DET_N=$((DET_N + 1))
    done
  elif [[ -n "$DET_CSV" ]]; then
    SPECS+=("det|${tag}|${model}|${DET_CSV}")        # one job, all three on one card
    DET_N=$((DET_N + 1))
  fi
  if [[ "$VSTAR_REQUESTED" == "true" ]]; then
    SPECS+=("vstar|${tag}|${model}|${VSTAR_REPO}")   # its own card; needs no judge
    DET_N=$((DET_N + 1))
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

echo "Planned ${#SPECS[@]} jobs (${JUDGED_N} judged + ${DET_N} deterministic) over ${NGPU} GPU(s) [${CARDS[*]}] -> ${OUTPUT_ROOT}"
[[ -n "$DET_CSV" ]] && echo "  deterministic group: ${DET_CSV} (eval_vqa.sh, no judge$([[ "$DET_SPLIT" == "true" ]] && echo ", split: 1 card each"))"
[[ "$RESUME" == "true" ]] && echo "  RESUME=true: jobs already complete will be skipped"
mkdir -p "$OUTPUT_ROOT"

# A job's output dir. det combined (csv) -> <id>/_vqa; det split (single benchmark)
# -> <id>/_vqa_<bench> so concurrent split jobs don't clobber each other's summary.json;
# judged -> <id>/<sanitized-dataset>.
job_out() {  # kind tag payload
  if [[ "$1" == "det" ]]; then
    if [[ "$3" == *,* ]]; then
      printf '%s/%s/_vqa' "$OUTPUT_ROOT" "$2"
    else
      printf '%s/%s/_vqa_%s' "$OUTPUT_ROOT" "$2" "$3"
    fi
  elif [[ "$1" == "vstar" ]]; then
    printf '%s/%s/_vstar' "$OUTPUT_ROOT" "$2"
  else
    local s; s="$(basename "$3")"; s="${s//[^A-Za-z0-9_.-]/_}"
    printf '%s/%s/%s' "$OUTPUT_ROOT" "$2" "$s"
  fi
}

# Is this job's output already complete? generate/all finish with summary.json; the
# judge phase finishes with judgments/*.jsonl (generate leaves that dir empty).
job_done() {  # kind out
  local kind="$1" out="$2" f
  if [[ "$PHASE" == "judge" ]]; then
    for f in "$out"/judgments/*.jsonl; do [[ -e "$f" ]] && return 0; done
    return 1
  fi
  [[ -f "$out/summary.json" ]]
}

run_job() {  # kind card tag model payload
  local kind="$1" card="$2" tag="$3" model="$4" payload="$5"
  local out logf
  # Each job's log lives INSIDE its own output folder (no central logs/ dir), named
  # by phase so generate.log and judge.log sit next to that job's responses/summary.
  out="$(job_out "$kind" "$tag" "$payload")"
  mkdir -p "$out"
  logf="$out/${PHASE}.log"
  if [[ "$kind" == "det" ]]; then
    CUDA_VISIBLE_DEVICES="$card" MODEL_PATH="$model" MODEL_NAME="$tag" \
      BENCHMARKS="$payload" OUTPUT_DIR="$out" \
      bash scripts/eval_vqa.sh > "$logf" 2>&1
    return
  fi
  if [[ "$kind" == "vstar" ]]; then
    CUDA_VISIBLE_DEVICES="$card" MODEL_PATH="$model" MODEL_NAME="$tag" \
      VSTAR_REPO="$payload" OUTPUT_DIR="$out" \
      bash scripts/eval_vstar.sh > "$logf" 2>&1
    return
  fi
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
  if [[ "$1" == "det" ]]; then echo "vqa[$3]"; elif [[ "$1" == "vstar" ]]; then echo "vstar"; else echo "$(basename "$3")"; fi
}

if [[ "$DRYRUN" == "1" ]]; then
  echo "--- DRYRUN: PHASE=$PHASE planned jobs ---"
  i=0
  for spec in "${SPECS[@]}"; do
    IFS='|' read -r kind tag model payload <<< "$spec"
    label="$(spec_label "$kind" "$tag" "$payload")"
    if [[ "$RESUME" == "true" && ! ( "$PHASE" == "judge" && ( "$kind" == "det" || "$kind" == "vstar" ) ) ]] \
       && job_done "$kind" "$(job_out "$kind" "$tag" "$payload")"; then
      echo "skip done (RESUME) | ${tag} | ${label}"
    elif [[ "$PHASE" == "judge" && ( "$kind" == "det" || "$kind" == "vstar" ) ]]; then
      echo "skip (scored in generate) | ${tag} | ${label}"
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
    [[ "$kind" == "det" || "$kind" == "vstar" ]] && continue   # deterministic already scored in generate
    i=$((i + 1))
    jout="$(job_out "$kind" "$tag" "$payload")"
    if [[ "$RESUME" == "true" ]] && job_done "$kind" "$jout"; then
      echo "[judge ${i}/${total}] ${tag} | $(basename "$payload") -> skip done (RESUME)"
      continue
    fi
    echo "[judge ${i}/${total}] ${tag} | $(basename "$payload")"
    run_job "$kind" "" "$tag" "$model" "$payload" || echo "  (job exited non-zero; see log)"
    # Surface this benchmark's score on the main log as soon as it's judged.
    sc="$(grep -h 'pass@k=' "$jout/judge.log" 2>/dev/null | tail -1)"
    [[ -n "$sc" ]] && echo "    -> ${sc}"
  done
  [[ "$total" -eq 0 ]] && echo "(deterministic-only request: nothing to judge — scored in the generate phase)"
  echo "All ${total} judge jobs finished. Per-job log: <id>/<dataset>/judge.log under ${OUTPUT_ROOT}/"
else
  # ---- generate / all: <=NGPU concurrent, one job pinned per free card ----
  declare -A SLOT  # SLOT[card]=pid of the job currently on that card
  for spec in "${SPECS[@]}"; do
    IFS='|' read -r kind tag model payload <<< "$spec"
    if [[ "$RESUME" == "true" ]] && job_done "$kind" "$(job_out "$kind" "$tag" "$payload")"; then
      echo "[skip done] ${tag} | $(spec_label "$kind" "$tag" "$payload")  (RESUME)"
      continue
    fi
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
    echo "[launch] card ${card} | ${tag} | $(spec_label "$kind" "$tag" "$payload")"
    run_job "$kind" "$card" "$tag" "$model" "$payload" &
    SLOT[$card]=$!
    # Offset the next job's CPU-heavy preprocessing from this one's (better CPU/GPU overlap).
    [[ "$LAUNCH_STAGGER" != "0" ]] && sleep "$LAUNCH_STAGGER"
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
    "vstar": ("accuracy", "vstar(acc)"),
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
# Derived row: MMMU-Pro average over its two sub-scores (per tag). Name ends with ")"
# so the per-checkpoint "avg(judged pass@k)" rollup excludes it (same convention as the
# det rows). Added only when at least one sub-score is present.
_pro_subs = ("mmmu_pro_10options", "mmmu-pro-vision")
if any(s in matrix for s in _pro_subs):
    avg_row = {}
    for t in tags:
        vals = [matrix[s][t] for s in _pro_subs if isinstance(matrix.get(s, {}).get(t), (int, float))]
        avg_row[t] = sum(vals) / len(vals) if vals else None
    matrix["mmmu-pro(avg)"] = avg_row
per_model = {}
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

    # Per-checkpoint rollup: each ckpt's benchmark scores + a mean over the judged
    # (pass@k) rows. det rows (labelled "pope(F1)" etc.) use their own metric, so they
    # are listed but kept out of the judged average.
    print("\n=== per-checkpoint summary ===")
    for t in tags:
        scores = {n: matrix[n][t] for n in sorted(matrix) if isinstance(matrix[n].get(t), (int, float))}
        per_model[t] = scores
        print(f"\n[{t}]  ({len(scores)} benchmarks)")
        for n in sorted(scores):
            print(f"  {n:<22} {scores[n]:.4f}")
        judged = [v for n, v in scores.items() if not n.endswith(")")]
        if judged:
            print(f"  {'avg(judged pass@k)':<22} {sum(judged) / len(judged):.4f}")
out = os.path.join(root, "matrix.json")
json.dump({"tags": tags, "matrix": matrix, "per_model": per_model}, open(out, "w"), indent=2, ensure_ascii=False)
print(f"\nWrote {out}")
PY

# Final shareable report: methods (rows) x benchmarks (cols), metrics as PERCENTAGES,
# written as report.md + report.csv (+ report.xlsx if openpyxl) under OUTPUT_ROOT.
# Pure stdlib transpose of the matrix above; safe to re-run any time on this root:
#   python3 baseline/eval/make_report.py "$OUTPUT_ROOT"
python3 baseline/eval/make_report.py "$OUTPUT_ROOT" \
  || echo "(report skipped; rerun: python3 baseline/eval/make_report.py $OUTPUT_ROOT)"
