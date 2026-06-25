# Multimodal OPD

**Vanilla multimodal On-Policy Distillation (OPD)** for vision-language models.
This project is built on top of the [ViGOS](README.md) code framework (rollout,
vLLM colocate, exact full-vocabulary KL, DDP-normalized losses) but implements a
different and simpler objective, kept in **separate files** so the ViGOS / OPSD
code paths stay untouched.

> Status: training path implemented (`vigos/train_opd.py`). Evaluation framework
> and model/architecture modifications (e.g. attention changes) are planned — see
> [Roadmap](#roadmap).

## What is OPD here (vs the ViGOS OPSD baseline)

| | OPSD (ViGOS, upstream) | **OPD (this project)** |
|---|---|---|
| Teacher | the **same** weights with the LoRA adapter disabled | a **separate, frozen, stronger** same-family VLM checkpoint |
| Teacher prompt | **privileged** (contains the reference answer) | the **same non-privileged** prompt the student sees |
| Supervised tokens | description / think / answer spans | the **full completion** |
| Loss | `λ_perc·L_perc + λ_reas·L_reas + λ_ref·L_ref` | a single per-token KL (default **exact reverse KL** `KL(student‖teacher)`; top-k/forward/jsd configurable) |
| Prompt | ViGOS `<description>…</description><think>…</think>\boxed{}` | the **dataset's own `problem`** + optional boxed-answer suffix |

The **student is trained with full fine-tuning by default** (matching
[Vision-OPD](https://github.com/VisionOPD/Vision-OPD)); set `FINETUNING_MODE=lora`
for a cheap memory-constrained run.

OPD mechanism per step:

1. The **student** samples one on-policy rollout from the dataset prompt
   (vLLM colocate by default).
2. A **frozen teacher** (loaded from `TEACHER_MODEL`) runs a single forward pass
   over the *same* prompt + the sampled completion.
3. Loss = per-token KL between student and teacher over the completion — by
   default **exact reverse KL** `KL(student‖teacher)` (canonical OPD,
   mode-seeking); configurable to top-k / forward / JSD (top-k forward is what the
   vllm_server teacher uses) — re-normalized across DDP ranks by the global
   active-token count.

The teacher is never updated and is never synced into vLLM; only the student is.

References: [Agarwal et al., GKD (2023)](https://arxiv.org/pdf/2306.13649),
[Thinking Machines: On-Policy Distillation (2025)](https://thinkingmachines.ai/blog/on-policy-distillation/),
[awesome-on-policy-distillation](https://github.com/chrisliu298/awesome-on-policy-distillation).

## Files added by this project

All OPD code lives in the new top-level `baseline/` package; the `vigos/` package
is reused as a library (rollout/teacher/KL/DDP helpers) but its files are unchanged.

| File | Role |
|------|------|
| `baseline/opd_data_collator.py` | `OPDDataCollator` — builds only the non-privileged student prompt from the dataset's `problem`. Dataset-agnostic. |
| `baseline/opd_trainer.py` | `OPDTrainer(ViGOSTrainer)` — overrides `compute_loss` with on-policy reverse-KL vs a frozen teacher; reuses all ViGOS rollout/teacher/KL helpers. |
| `baseline/train_opd.py` | Standalone OPD entry point (loads student+LoRA, frozen teacher, OPD collator, trainer). |
| `baseline/__init__.py` | Package marker for the `baseline` namespace. |
| `baseline/eval/opd_eval_prompt.py` | General eval prompt (dataset problem + suffix, no prefill). |
| `baseline/eval/run_opd_eval.py` | General multi-benchmark eval harness (vLLM gen + LLM judge + pass@k/avg@k). |
| `baseline/eval/run_mmvp_eval.py` | MMVP pair-metric MCQ eval (vLLM gen + rule MCQ match, **no judge**; single-question + pair accuracy). |
| `baseline/eval/run_vqa_eval.py` | POPE / ChartQA / VQAv2 short-answer eval (vLLM gen + official per-benchmark metric, **no judge**; one engine load, all three). |
| `baseline/eval/vqa_metrics.py` | Pure metric primitives for the above (POPE F1, ChartQA relaxed accuracy, VQAv2 soft accuracy). |
| `baseline/eval/aggregate_suite.py` | Merge the judged + deterministic group summaries into one suite table (POPE per-category +avg, MMMU-Pro per-subscore +avg; optional pass@k/avg@k). |
| `baseline/eval/passk.py` | Unbiased pass@k / avg@k estimator over N samples (Codex estimator), from the per-attempt judge verdicts. |
| `baseline/serve_teacher.py` | vLLM teacher scoring server (`/score_topk`, top-k `prompt_logprobs`). |
| `baseline/teacher_client.py` | HTTP client the trainer uses for the `vllm_server` teacher. |
| `scripts/train_opd_qwen25_3b.sh` | Train launcher (runs `baseline/train_opd.py`); env-var overrides. |
| `scripts/eval_opd.sh` | Eval launcher (runs `baseline/eval/run_opd_eval.py`). |
| `scripts/eval_mmvp.sh` | MMVP eval launcher (runs `baseline/eval/run_mmvp_eval.py`). |
| `scripts/eval_vqa.sh` | POPE/ChartQA/VQAv2 eval launcher (runs `baseline/eval/run_vqa_eval.py`). |
| `scripts/eval_suite.sh` | One-command full suite: judged + deterministic groups → merged `suite_summary.json` table. |
| `scripts/serve_teacher_vllm.sh` | Launch the teacher scoring server. |

ViGOS files under `vigos/` (`train_vigos.py`, `trainer.py`, `data_collator.py`, …) are unchanged.

## Environment

```bash
uv sync --python 3.11   # PyTorch 2.8, Transformers 4.57.1, TRL 0.26, vLLM 0.11
```

`flash-attn` is **not** a dependency. The training script defaults to
`ATTN_IMPLEMENTATION=sdpa` for the HF forward, so it runs without flash-attn;
eval and the `vllm_server` teacher use vLLM and never need it. Set
`ATTN_IMPLEMENTATION=flash_attention_2` (and `TEACHER_ATTN_IMPLEMENTATION=...`)
once flash-attn is installed.

## Data

Any HuggingFace dataset exposing `problem` (text), `images` (PIL), and `answer`.
Default reference dataset:

```bash
export DATASET_NAME=LMMs-Lab-Turtle/Vision-SR1-47K
```

All stages (teacher GRPO, student OPD, eval) share **one unified system prompt**
(paper appendix B.4) — enforcing `<reason></reason>` CoT + a `\boxed{}` final
answer — with the user turn = image + the dataset's raw question. This structural
alignment between teacher and student is what OPD needs. The system prompt lives in
`baseline/opd_data_collator.py::OPD_SYSTEM_PROMPT` (and is mirrored in the GRPO
launcher's `--system`); switching datasets needs no prompt changes.

## Training

```bash
DATASET_NAME=LMMs-Lab-Turtle/Vision-SR1-47K \
TEACHER_MODEL=Qwen/Qwen2.5-VL-7B-Instruct \
bash scripts/train_opd_qwen25_3b.sh
```

`TEACHER_MODEL` may be a base checkpoint or an RL/SFT-tuned one; it must be the
**same model family** as the student (shared tokenizer/vocab is required for
exact full-vocabulary KL).

### Verified multi-GPU launch — read this first

The form above is minimal. For a real run, **pin the GPUs and clear stale
smoke-test env vars explicitly** — on a shared box a fresh shell can default to a
single visible GPU, which silently drops the effective batch to ~1:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NUM_PROCESSES=8   # pin all 8 GPUs
unset  MAX_STEPS MAX_TRAIN_SAMPLES                            # clear smoke-test limits
export WANDB_MODE=online                                      # or offline + `wandb sync` later
M=/path/to/models
MODEL_NAME_OR_PATH=$M/Qwen3-VL-2B-Instruct TEACHER_MODEL=$M/Qwen3-VL-8B-Instruct \
DATASET_NAME=.../Vision-SR1-47K \
RUN_CONFIG=opd_qwen3_8b_to_2b OUTPUT_DIR=runs/opd_qwen3_8b_to_2b \
bash scripts/train_opd_qwen25_3b.sh
```

At startup the run prints its real world size — **confirm it before trusting a run**:

```
[OPD] num_processes(world_size)=8  per_device_bs=1  grad_accum=4  -> effective_batch=32
```

**Why this matters (NaN root cause).** Reverse-KL full-FT NaN'd within 3 steps when
the effective batch was ~1 (single GPU). At batch 1, one image's Qwen2.5-VL ViT
(`visual.patch_embed`) gradient spike (grad_norm ~448) overflows in bf16 and poisons
the optimizer → NaN weights. At a real 8-GPU batch of 32 the spikes average out
(grad_norm ~3–20) and full FT **including the ViT** is stable for both Qwen2.5-VL-3B
and Qwen3-VL-2B. If a run NaNs, **check the effective batch / GPU count first**; the
`[OPD-NaN]` probe in `opd_trainer.py` (fires only on a non-finite step) localizes
forward-vs-weight NaNs. As a small-batch / single-GPU fallback,
`FREEZE_VISION_TOWER=true` freezes the ViT.

### Key knobs (env vars)

| Var | Default | Meaning |
|-----|---------|---------|
| `TEACHER_MODEL` | *(required)* | Frozen teacher checkpoint path/id |
| `MODEL_NAME_OR_PATH` | `Qwen/Qwen2.5-VL-3B-Instruct` | Student base |
| `FINETUNING_MODE` | `full` | `full` (all params) or `lora` |
| `FREEZE_VISION_TOWER` | `false` | Freeze the ViT under full-FT — small-batch / single-GPU NaN fallback (see above) |
| `LEARNING_RATE` | `2e-6` | Full-FT LR (Vision-OPD uses 2e-6) |
| `WARMUP_RATIO` | `0.03` | LR warmup fraction |
| `CUDA_VISIBLE_DEVICES` / `NUM_PROCESSES` | `0..7` / `8` | **Pin explicitly** — a fresh shell may default to 1 GPU |
| `WANDB_MODE` | `online` | `offline` on no-network boxes, then `wandb sync` later |
| `OPD_LOSS_MODE` | `full_kl` | `full_kl` (full vocab) or `topk_kl` |
| `OPD_KL_DIRECTION` | `reverse` | `reverse` / `forward` / `jsd` |
| `OPD_TOP_K` | `32` | Top-k tokens when `topk_kl` (verl=32, thunlp=16) |
| `LAMBDA_OPD` | `1.0` | Distillation loss weight |
| `DISTILL_TEMPERATURE` | `1.0` | KL softmax temperature |
| `TOKEN_LOSS_CLIP` | `0.0` | Per-token KL clip (0 = off) |
| `OPD_PROMPT_SUFFIX` | boxed-answer instruction | Appended to the raw dataset prompt |
| `ACCELERATE_CONFIG` | `configs/accelerate_zero2_gpu_8.yaml` | DeepSpeed ZeRO-2 (full-FT friendly) |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.30` | Lowered to make room for the teacher replica |

### Distillation loss

The OPD divergence is configurable, matching the wider top-k OPD ecosystem
([verl](https://verl.readthedocs.io/en/latest/algo/opd.html) `forward_kl_topk`,
[thunlp/OPD](https://github.com/thunlp/OPD), [Uni-OPD](https://github.com/WenjinHou/Uni-OPD)):

- `OPD_LOSS_MODE=full_kl` (default): exact full-vocabulary KL. With the local
  teacher (full logits) it costs nothing, so it's the default.
- `OPD_LOSS_MODE=topk_kl`: KL over the top-`OPD_TOP_K` tokens only — the point of
  top-k is a remote teacher that returns just top-k logprobs (the vllm_server path);
  for a local teacher it's only an approximation with no speedup.
- `OPD_KL_DIRECTION`: `reverse` (default) = `KL(student‖teacher)`, mode-seeking —
  the canonical OPD objective (Thinking Machines / GKD β=1); `forward` =
  `KL(teacher‖student)`, mass-covering (verl's forward_kl_topk; **required by the
  vllm_server teacher**); `jsd` = Jensen-Shannon.

### Teacher source

`TEACHER_SOURCE` selects how teacher signals are obtained:

- `local_hf` (default): a frozen teacher replica per GPU runs a full-logit HF
  forward; supports all loss modes/directions. Memory cost = a teacher copy on
  every training GPU (so ≤14B).
- `vllm_server` (experimental): a **separate** vLLM server scores
  `prompt_token_ids + completion_ids` (+image) with `prompt_logprobs=top_k` and
  returns the teacher's top-k logprobs. No per-GPU replica, so the teacher can be
  far larger than the student (32B/72B). Only `topk_kl` + `forward` is supported
  (a server returns the teacher's top-k, i.e. forward KL).

```bash
# 1) Start the teacher server on its own GPU(s):
CUDA_VISIBLE_DEVICES=0,1 TEACHER_MODEL=Qwen/Qwen2.5-VL-72B-Instruct \
TENSOR_PARALLEL_SIZE=2 PORT=8200 bash scripts/serve_teacher_vllm.sh

# 2) Train, pointing at the server (no local teacher replica):
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
DATASET_NAME=LMMs-Lab-Turtle/Vision-SR1-47K \
TEACHER_SOURCE=vllm_server TEACHER_SERVER_URL=http://127.0.0.1:8200 \
bash scripts/train_opd_qwen25_3b.sh
```

> Query the teacher at temperature 1.0 (the server does); `DISTILL_TEMPERATURE`
> then scales only the student. The multimodal `prompt_token_ids` +
> `prompt_logprobs` path is experimental and needs GPU validation.

### Memory / teacher scale

Full-FT is heavy: it co-locates a full-parameter student (params + grads +
Adam state, ZeRO-2-sharded), a **replicated frozen teacher** on every GPU, and
the colocate vLLM engine. The default `scripts/train_opd_qwen25_3b.sh` uses
`per_device_train_batch_size=1`, `gradient_accumulation_steps=4`, and
`VLLM_GPU_MEMORY_UTILIZATION=0.30`.

The teacher is replicated (inference-only) on **every** GPU:

- **3B/4B student (full FT) + 7B teacher** — fits on 8×A100-80G under ZeRO-2.
- **7B student (full FT)** — switch to a ZeRO-3 + CPU-offload accelerate config
  (and note: under ZeRO-3 the frozen teacher must be loaded *unpartitioned*,
  which is not yet wired up — validate before relying on it).
- **≥32B teacher** — does not fit per-GPU; needs teacher tensor-parallel or a
  switch to top-k KL. Out of scope for the default script.

vLLM's logprob API returns only top-k, so the teacher must run a **local HF
forward pass** for full-vocabulary KL.

## Evaluation

A **general** multi-benchmark harness (`baseline/eval/`, launched by
`scripts/eval_opd.sh`) that uses the dataset's own prompt — **not** the ViGOS
format — so it works for any checkpoint (OPD / OPSD / base) and any dataset. It
reuses the generic `vigos.eval_utils` / `vigos.eval_benchmarks` helpers (sample
extraction, LLM-judge prompts, scoring) and adds the general OPD prompt.

Pipeline: vLLM generate pass@k → extract `\boxed` answer → OpenAI-compatible
LLM-judge → pass@k / avg@k → `responses/`, `judgments/`, `summary.json`.

```bash
export DEEPSEEK_API_KEY=...                      # judge (or SKIP_JUDGE=true)
# Full FT writes a full checkpoint, so point straight at the run dir:
MODEL_PATH=runs/opd_qwen25_3b_<RUN> bash scripts/eval_opd.sh
# Generation only, no judging:
MODEL_PATH=runs/opd_qwen25_3b_<RUN> SKIP_JUDGE=true bash scripts/eval_opd.sh
```

Knobs (env): `EVAL_DATASETS`, `EVAL_BENCHMARKS` (e.g. `vilp-f,vilp-p,cv-bench`),
`PASS_K`, `LIMIT`, `GRADER` (`llm` default / `rule` = mathruler+option match, no API),
`JUDGE_MODEL`, `TENSOR_PARALLEL_SIZE`, …

For `Acc@1` use `PASS_K=1 GEN_TEMPERATURE=0` (greedy); for `Avg@k`/`Pass@k` use
`PASS_K=k` with sampling (`GEN_TEMPERATURE=1.0`). `summary.json`: `pass_at_k` =
Pass@k, `avg_at_k` = Avg@k.

### Standard benchmarks (MMMU, MMMU-Pro, MathVista, MathVerse, MathVision, MMStar, HallusionBench)

Use the pre-formatted `zli12321/*` datasets (already `problem`/`answer`/`images`).
Download once (behind a proxy: direct HF, no Xet/hf_transfer):

```bash
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
DSROOT=<datasets>/zli12321
for d in mmstar MMMU mmmu_pro_10options mmmu-pro-vision mathvista mathverse mathvision hallusionbench; do
  hf download "zli12321/$d" --repo-type dataset --local-dir "$DSROOT/$d"
done
```

Then run Acc@1 (greedy) and Avg@8 / Pass@8 (sampled). vLLM eval needs the Q5
triton patch on the venv (see the GRPO README); use a free GPU.

```bash
export DEEPSEEK_API_KEY=...
DS="$DSROOT/mmstar,$DSROOT/MMMU,$DSROOT/mmmu_pro_10options,$DSROOT/mmmu-pro-vision,$DSROOT/mathvista,$DSROOT/mathverse,$DSROOT/mathvision,$DSROOT/hallusionbench"

# Acc@1
CUDA_VISIBLE_DEVICES=0 MODEL_PATH=<model> EVAL_DATASETS="$DS" \
  PASS_K=1 GEN_TEMPERATURE=0 OUTPUT_DIR=eval_outputs/<m>_acc1 bash scripts/eval_opd.sh
# Avg@8 / Pass@8
CUDA_VISIBLE_DEVICES=0 MODEL_PATH=<model> EVAL_DATASETS="$DS" \
  PASS_K=8 GEN_TEMPERATURE=1.0 GEN_TOP_P=0.9 OUTPUT_DIR=eval_outputs/<m>_passk8 bash scripts/eval_opd.sh
```

Caveats: HallusionBench here ≈ aAcc only (not fAcc/qAcc); MMMU-Pro = average the
`mmmu_pro_10options` + `mmmu-pro-vision` sub-scores. POPE / ChartQA / VQAv2 have
their own official metrics — use the dedicated `scripts/eval_vqa.sh` below.

### MMVP (pair metric — visual-perception / ViT-unfreeze probe)

MMVP (Tong et al., *Eyes Wide Shut?*, CVPR 2024) = 150 CLIP-blind image **pairs**
→ 300 binary MCQs. The two images in a pair differ by a single fine visual
attribute, so the **pair metric** (a pair scores 1 only if *both* questions are
correct) cannot be gamed by a language prior — a clean check on whether unfreezing
the ViT during OPD genuinely improved general visual perception or catastrophically
degraded it. Dedicated, deterministic, **no LLM judge** (option-letter match), run
under the unified OPD system prompt; separate from the general harness.

```bash
# greedy single-sample (canonical MMVP); writes summary.json with pair_accuracy
CUDA_VISIBLE_DEVICES=0 MODEL_PATH=<model> bash scripts/eval_mmvp.sh
# smoke test on 4 questions first:
CUDA_VISIBLE_DEVICES=0 MODEL_PATH=<model> LIMIT=4 bash scripts/eval_mmvp.sh
```

`summary.json` → `metrics.pair_accuracy` (headline, greedy), plus `pair_pass_at_k`,
`question_accuracy`, `question_avg_at_k`. Knobs (env): `MMVP_REPO` (default
`MMVP/MMVP`), `IMAGE_DIR`, `PAIR_SIZE` (2), `LIMIT`, `PASS_K` / `GEN_TEMPERATURE`
(raise temperature if `PASS_K>1`), `OUTPUT_DIR`. Per-category buckets populate only
if the source CSV carries a category column.

### POPE / ChartQA / VQAv2 (short-answer, official per-benchmark metrics)

Three classic single-image benchmarks, each scored by its **own canonical official
metric**, deterministically and with **no LLM judge** (no API key). Run under the
unified OPD system prompt; one `scripts/eval_vqa.sh` invocation loads the vLLM
engine once and evaluates each requested benchmark in turn.

| Benchmark | Headline metric | Also reported |
|-----------|-----------------|---------------|
| **POPE** (hallucination, yes/no) | **F1** | accuracy / precision / recall / yes-ratio, per category (random/popular/adversarial) |
| **ChartQA** (chart QA) | **relaxed accuracy** (numeric within 5 %, else exact) | human vs augmented split + their mean |
| **VQAv2** (open-ended VQA) | **VQA soft accuracy** (`min(1, agreement/3)` over the 10 human answers, official normalization) | per answer-type (yes/no, number, other) |

Sources default to the canonical **lmms-lab** HF datasets and are auto-downloaded
& cached on first use (the box can connect to HF directly). To pre-fetch (or for an
offline box, then point `*_REPO` at the local dir):

```bash
# POPE — all 3 categories incl. adversarial (~9k yes/no questions, images embedded)
hf download lmms-lab/POPE    --repo-type dataset --local-dir $D/POPE
# ChartQA — test split, human + augmented (~2.5k, images embedded)
hf download lmms-lab/ChartQA --repo-type dataset --local-dir $D/ChartQA
# VQAv2 — validation split (the one you don't have yet; large, ~214k Q + images)
hf download lmms-lab/VQAv2   --repo-type dataset --local-dir $D/VQAv2
```

```bash
# all three, greedy single-sample (canonical); writes summary.json with all metrics
CUDA_VISIBLE_DEVICES=0 MODEL_PATH=<model> bash scripts/eval_vqa.sh
# smoke test (8 questions each) first:
CUDA_VISIBLE_DEVICES=0 MODEL_PATH=<model> LIMIT=8 bash scripts/eval_vqa.sh
# skip VQAv2 (large) / run just POPE adversarial like your local split:
CUDA_VISIBLE_DEVICES=0 MODEL_PATH=<model> BENCHMARKS=pope POPE_CATEGORY=adversarial bash scripts/eval_vqa.sh
# use locally-downloaded copies instead of HF:
CUDA_VISIBLE_DEVICES=0 MODEL_PATH=<model> POPE_REPO=$D/POPE CHARTQA_REPO=$D/ChartQA VQAV2_REPO=$D/VQAv2 \
  bash scripts/eval_vqa.sh
```

`summary.json` → `benchmarks.{pope,chartqa,vqav2}.metrics` (e.g.
`benchmarks.pope.metrics.f1`, `…chartqa.metrics.relaxed_accuracy`,
`…vqav2.metrics.vqa_accuracy`; all reported as 0–1 fractions). Knobs (env):
`BENCHMARKS` (`pope,chartqa,vqav2`), `POPE_REPO` / `POPE_CATEGORY`, `CHARTQA_REPO`,
`VQAV2_REPO` / `VQAV2_SPLIT`, `LIMIT`, `PROMPT_SUFFIX`, `PASS_K` / `GEN_TEMPERATURE`
(raise temperature if `PASS_K>1`), `OUTPUT_DIR`. VQAv2 validation is large — set
`VQAV2_LIMIT` (caps VQAv2 only, leaving POPE/ChartQA full) for a quick read, or
drop `vqav2` from `BENCHMARKS`.

### Full suite in one command (`scripts/eval_suite.sh`)

Runs **everything** and merges it into one table: MathVista, MathVerse,
MathVision, MMMU, MMMU-Pro, MMStar, HallusionBench, POPE, ChartQA, VQAv2 — with
**POPE split into its 3 categories + average** and **MMMU-Pro split into its 2
sub-scores + average**. Internally it dispatches by grading philosophy (you do
*not* LLM-judge a benchmark that has an official deterministic metric):

* **judged group** (`eval_opd.sh`, your LLM judge): MathVista / MathVerse /
  MathVision (free-form math — judge needed), MMMU / MMMU-Pro / MMStar (MCQ —
  judge or `GRADER=rule`), HallusionBench (yes/no).
* **deterministic group** (`eval_vqa.sh`, no judge): POPE (F1), ChartQA (relaxed
  accuracy), VQAv2 (soft accuracy).

`baseline/eval/aggregate_suite.py` then stitches both `summary.json`s into
`suite_summary.json` + a printed table (scores as 0–1 plus a `%` column).

```bash
# point the judged group at your 32B judge (OpenAI-compatible), then one command:
export JUDGE_API_URL=http://<your-32b-host>:<port>/v1 JUDGE_MODEL=<served-name> OPENAI_API_KEY=<key>
CUDA_VISIBLE_DEVICES=0 MODEL_PATH=<model> bash scripts/eval_suite.sh
# quick read (cap VQAv2's ~214k val; greedy Acc@1):
CUDA_VISIBLE_DEVICES=0 MODEL_PATH=<model> VQAV2_LIMIT=2000 bash scripts/eval_suite.sh
# no judge available -> rule-grade the MCQ/yes-no ones (free-form math will be approximate):
CUDA_VISIBLE_DEVICES=0 MODEL_PATH=<model> GRADER=rule bash scripts/eval_suite.sh
```

Knobs (env): `JUDGED_DATASETS`, `DET_BENCHMARKS`, `GRADER` (`llm`/`rule`),
`SKIP_JUDGE`, `PASS_K` / `GEN_TEMPERATURE` (default greedy Acc@1), `VQAV2_LIMIT`,
`TENSOR_PARALLEL_SIZE`, `OUTPUT_ROOT`.

#### Also report pass@k / avg@k (`MULTI_K=true`)

By default the suite reports **greedy Acc@1** (one sample). Set `MULTI_K=true` to
*additionally* report `avg@N` and `pass@8` / `pass@16` for the judged (math/MCQ)
group. It runs one extra **sampled** pass over those datasets (`PASS_K=SAMPLED_K`,
`SAMPLED_TEMPERATURE>0`) and estimates pass@k for every k from the *same* N samples
(unbiased Codex estimator, `baseline/eval/passk.py`) — so pass@8 and pass@16 cost
~N× the greedy generation, **not** N× per k. POPE/ChartQA/VQAv2 stay greedy-only
(pass@k on yes/no / short-answer is not a standard metric; their official greedy
metric is the headline).

```bash
MODEL_PATH=<model> MULTI_K=true SAMPLED_K=16 PASSK_KS=1,8,16 VQAV2_LIMIT=2000 \
  bash scripts/eval_suite.sh
```

Extra knobs: `MULTI_K`, `SAMPLED_K` (N samples = max k, default 16), `PASSK_KS`
(default `1,8,16`), `SAMPLED_TEMPERATURE` (1.0), `SAMPLED_TOP_P` (0.9). Note
`avg@k = c/N` is the per-sample mean (k-independent, so avg@8 ≈ avg@16); the
k-dependent signal lives in `pass@k`. `pass@1 == avg`.

#### Metrics & lmms-eval parity

`lmms-eval` (the de-facto VLM eval) decodes **greedy, single-sample** and reports
each benchmark's official metric — so the number to compare against it is our
**greedy Acc@1**; `pass@k`/`avg@k` is an extra sampling-robustness view, not an
lmms-eval metric. On the metric *math* we match it where it is deterministic:
**POPE** (accuracy/precision/recall/F1/yes-ratio per category), **ChartQA**
(relaxed accuracy), **VQAv2** (official `VQAEval` normalization + soft accuracy)
are ported from the same reference implementations. For **MathVista / MathVerse /
MathVision** lmms-eval uses a GPT model for answer *extraction*; we use an LLM
*judge* for correctness — aligned in spirit, not bit-identical. For **MMMU /
MMMU-Pro / MMStar** lmms-eval uses a rule-based option parser; we default to the
judge (set `GRADER=rule` to get closer). **HallusionBench** here is aAcc-style
only. The biggest intentional difference: we prompt every benchmark with the **OPD
training prompt** (unified system + `\boxed{}`), not lmms-eval's per-task
templates — so absolute numbers differ from the public leaderboard; this harness is
built for *consistent relative* comparison (before/after OPD, student vs teacher),
not for reproducing leaderboard values.

**LoRA mode** (`FINETUNING_MODE=lora`): merge the adapter first, then point
`MODEL_PATH` at the merged dir:

```bash
uv run python scripts/merge_lora.py \
  --adapter runs/opd_qwen25_3b_<RUN>/checkpoint-XXX \
  --output runs/opd_qwen25_3b_merged --overwrite
MODEL_PATH=runs/opd_qwen25_3b_merged bash scripts/eval_opd.sh
```

> Default `EVAL_DATASETS` covers the generic LLM-judged benchmarks (MM-Vet, MMMU,
> MMMU-Pro, MathVerse, MathVista, MMSI, RealWorldQA). The bespoke benchmarks
> (ViLP / CV-Bench) work via `EVAL_BENCHMARKS` but have their own answer-format
> expectations — set `PROMPT_SUFFIX=""` for those.

> The current eval prompt is ViGOS-style. A general OPD/benchmark eval harness is
> on the roadmap.

## Roadmap

- [x] **GPU-validated full-FT training** on 8×H800 — Qwen2.5-VL-3B (←7B teacher)
      and Qwen3-VL-2B (←8B teacher) train stably (reverse-KL, full vocab, unfrozen
      ViT at effective batch 32). See `PROGRESS_OPD.md` for status & learnings.
- [x] General multi-benchmark evaluation harness (`baseline/eval/`, dataset prompt,
      pass@k/avg@k, LLM judge) — generic-dataset path; bespoke benchmarks reused
      from `vigos.eval_benchmarks`.
- [x] MMVP pair-metric eval (`baseline/eval/run_mmvp_eval.py`) — deterministic
      MCQ/pair scorer (no judge); probes whether unfreezing the ViT in OPD helped
      or hurt general visual perception.
- [x] POPE / ChartQA / VQAv2 eval (`baseline/eval/run_vqa_eval.py`) — official
      per-benchmark metrics (F1 / relaxed accuracy / VQA soft accuracy), deterministic
      (no judge), one engine load for all three.
- [ ] Model/architecture experiments (e.g. attention modifications) on the student.
- [ ] Optional completion-sample logging for OPD rollouts.
- [x] Top-k KL loss (`topk_kl`, forward/reverse/jsd) — local HF teacher.
- [x] vLLM-server teacher returning only top-k logprobs (no per-GPU replica;
      enables ≥32B teachers) — reuses the `topk_kl` forward loss. *(experimental,
      needs GPU validation.)*
- [ ] PG/GRPO OPD variant (reverse-KL-as-reward), like verl PG OPD.

## Attribution & license

Built on the ViGOS framework (see [`README.md`](README.md), [`NOTICE`](NOTICE)).
Code under `vigos/`, `scripts/`, `configs/` follows the upstream Apache-2.0 terms.
