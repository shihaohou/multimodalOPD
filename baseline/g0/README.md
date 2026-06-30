# G0 — grounding diagnostic (looking-vs-using)

The one question G0 answers: when the student gets a vision-grounded question
wrong, is it a **looking failure** (its attention/localization points at the
wrong region) or a **using failure** (it localizes the right region but the
answer doesn't draw on it)? That answer gates every attention/map method
(explains why TAM / Saliency-R1 didn't move accuracy) and decides whether to push
explicit grounding or stay at the OPD **output level** (the hidden-hint).

Two faithful probes, deliberately separate (raw attention / TAM are *not* used):

| Probe | File | Measures | "leg" |
|-------|------|----------|-------|
| **LocalizationHeads** | `localization_heads.py` | where the model's grounding heads look → IoU vs GT box | **looking** |
| **GLIMPSE** | `glimpse.py` | what actually drives the answer (image vs text/prior) | **using** |

Both run on `peterant330/saliency-r1-8k` (every row has a GT evidence box), over
three conditions:

| | model | input | CoT |
|---|---|---|---|
| **C1** | teacher (8B) | image + question | natural |
| **C2** | teacher (8B) | image + question + **silent GT-box hint** | natural (box not verbalized) |
| **C3** | student (2B) | image + question | natural |

Per sample × condition we record `IoU_LH` (looking), `IoU_GL` + `vt_ratio`
(using), and answer `correct`.

## How it works

* **LH** (port of arXiv 2503.06287, LLaVA → Qwen): take the first-generation-step
  attention (last-prompt-token row) over the image patch grid
  `(grid_h//merge, grid_w//merge)`, found via `input_ids == image_token_id`.
  *Calibrate* per-head IoU vs GT on a calibration split to pick each model's
  top-k localization heads (8B and 2B separately — no L14-H24 assumption), then
  assemble those heads → smooth → threshold → box → IoU. Eager attention only.
* **GLIMPSE** (arXiv 2506.18985, one-backward variant): a single
  `torch.autograd.grad` of `S = Σ logit(generated token)` over the attention
  maps; per-head `ReLU(grad⊙attn)` (Eq.5), adaptive head weights (Eq.6), layer
  weights (Eq.9–11), confidence weights (Eq.17). We read the **response rows** of
  each layer's fused relevance instead of the full `N×N` rollout (which OOMs at
  thousands of visual tokens) — documented in `glimpse.py`. Yields a visual map
  (→ `IoU_GL`, energy-in-bbox, pointing) and `vt_ratio` = visual / (visual +
  textual-prompt) mass.

## Run

**Single GPU, quick subset** (offline → pass the LOCAL dataset dir, not the HF id):

```bash
export D=/home/web_server/antispam/project/houshihao/datasets   # datasets root
CUDA_VISIBLE_DEVICES=0 \
STUDENT_MODEL=$M/Qwen3-VL-2B-Instruct TEACHER_MODEL=$M/Qwen3-VL-8B-Instruct \
DATASET=$D/saliency-r1-8k SUBSETS=textvqa,docvqa,gqa,openimages LIMIT=80 \
RUN_NAME=run1 bash scripts/g0_diag.sh
# → eval_outputs/g0/run1/{records.jsonl, head_stats_*.json, analysis.json, report.md, figs/, viz/}
```

**All 8 GPUs · all 8k samples · multiple teachers** (the orchestrator): launches
`NUM_SHARDS` data-parallel shards per teacher (one process/GPU, each decoding only
its 1/N of the images), then analyzes each teacher's merged records:

```bash
export D=/home/web_server/antispam/project/houshihao/datasets
export M=/home/web_server/antispam/project/houshihao/models
STUDENT_MODEL=$M/Qwen3-VL-2B-Instruct \
TEACHER_MODELS=$M/Qwen3-VL-8B-Instruct,$M/CapCurriculum-8B \
bash scripts/g0_diag_multi.sh
# defaults: NUM_SHARDS=8, GPUS=0..7, DATASET=$D/saliency-r1-8k, SUBSETS="" (all), LIMIT=0 (full 8k)
# → eval_outputs/g0/{Qwen3-VL-8B-Instruct,CapCurriculum-8B}/report.md  (compare across teachers)
```

Sharding is row-strided (`samples[i::N]`); head calibration uses a fixed unsharded
slice so every shard finds identical localization heads (IoU_LH comparable), and
the **eval set is held disjoint from the calibration set** (so analysis 1 is not
optimistic). Each shard writes `records.shardXofN.jsonl`; `analyze_g0` globs them
all (and tolerates a half-written trailing line for mid-run previews). `LIMIT=0` =
no per-subset cap; `SUBSETS=""` = all subsets (note: the free-form subsets
`flickr30k`/`v7w` need an LLM judge — our rule grader under-scores them, so their
`correct` is noisy; the headline subsets textvqa/docvqa/gqa/openimages/cub/vsr are
graded cleanly).

## Outputs → the four analyses (`report.md`)

1. **Head usability** (8B & 2B) — best per-head mean IoU, the selected heads +
   their layers, assembled `IoU_LH`. Gates the LH-box / label-free plans.
2. **Student looking-vs-using** (KEY) — the verdict is **correlation-driven**
   (threshold-free): `corr(correct, IoU_LH)` vs `corr(correct, vt_ratio)` on C3.
   IoU_LH not predicting correctness while vt_ratio does ⇒ **using failure** (stay
   output-level); IoU_LH tracking correctness ⇒ **looking failure**. Reported with
   THREE 2×2s (absolute IoU≥0.30 / pointing / relative-median) since a median split
   mechanically calls half the samples "high IoU". Computed for both the
   first-gen-step LH and the **answer-span** LH (`iou_lh_answer`).
3. **Hint mechanism** (C1 vs C2, paired) — ΔIoU_LH ≫ 0 with Δacc > 0 ⇒
   *attentional*; Δacc > 0 with ΔIoU_LH ≈ 0 ⇒ *non-attentional / output-level*.
4. **Teacher-vs-student gap** (C1 vs C3) — is the gap mainly localization
   (`IoU_LH`) or attribution (`vt_ratio`)? Tells us what OPD must transfer.

## Comparing teachers / runs

G0 is one-teacher-per-run, so to compare teachers (e.g. Qwen3-VL-8B vs
CapCurriculum-8B) put their run dirs side by side:

```bash
uv run python -m baseline.g0.compare_g0 \
    --run-dirs eval_outputs/g0/Qwen3-VL-8B-Instruct eval_outputs/g0/CapCurriculum-8B \
    --output eval_outputs/g0/compare.md
```

It tabulates the teacher block (head usability + C1 acc/IoU/pointing + the
C1-vs-C2 hint Δacc/ΔIoU_LH/verdict) per run, plus the student (C3) block — the
same 2B model in every run, so it's also a consistency check (works on partial
runs; just shows fewer n).

## Visualizing cases (stratified, criterion-based)

The inline viz (`VIZ_PER_SUBSET`, default 2) saves the first N samples **per
subset** during the run. To *see the failures the numbers point to*, run the
post-hoc selector — it reads `records.jsonl`, picks the most informative cases
per subset, and re-renders only those (cheap; reuses the run's config + heads):

```bash
uv run python -m baseline.g0.viz_g0 --run-dir eval_outputs/g0/Qwen3-VL-8B-Instruct \
    --select using_failure --per-subset 4 --conditions c1,c2,c3
# --select: low_iou_lh | low_iou_gl | low_vt | using_failure | looking_failure | wrong | high_iou_lh | random
# → eval_outputs/.../viz_using_failure/<subset>_<id>_<cond>.png
```

`using_failure` = wrong but high IoU_LH (looked right, answered wrong) — the
"using" smoking guns. `--rank-condition` (default c3) chooses whose records the
criterion ranks; each picked sample is rendered for every `--conditions` so you
can compare C1/C2/C3 on the same image.

## Knobs / gotchas

* `MAX_PIXELS` (default 602112 ≈ 768 visual tokens) is the OOM lever — the
  GLIMPSE grad forward's memory scales ~`S²`. Lower it if the 8B grad OOMs; raise
  it for finer grids. The same cap is used for calibration and eval so head
  indices stay comparable.
* Offline box: pass `DATASET=$D/saliency-r1-8k` (local dir), never the HF id —
  `HF_HUB_OFFLINE=1` makes an id fail with `OfflineModeIsEnabled`.
* `--top-k-heads` (3), `--min-layer` (2, ignore early layers when selecting),
  `--lh-sigma` (1.0) for LH; `--glimpse-lambda` / `--glimpse-lambda-depth` for
  GLIMPSE; `--threshold {mean,top_frac}` for the maps.
* `--glimpse-layers` default `last8` (`all` | `lastN` | comma list) — the memory
  lever; first-round diagnosis doesn't need full all-layer attribution.
* `--answer-tokens` (16) = last-K generated tokens treated as the answer span
  (records `iou_lh_answer` / `vt_ratio_answer` — the verdict prefers these over the
  full-response / first-token variants, which dilute / mis-target the signal).
* `analyze_g0 --abs-iou-threshold` (0.30) sets the "genuinely looked right" bar for
  the 2×2 tables (the verdict itself uses correlations, not this).
* Greedy decoding by default (reproducible); `SAMPLE=1` to sample.
* CPU self-tests for the geometry/aggregation logic:
  `python -m baseline.g0.metrics` (and `python -m baseline.g0.answer_spans`,
  `python -m baseline.g0.eagle_src.regions`).

## Answer-span precision + LLM-judge correctness (all probes)

Two upgrades the verdict depends on:

* **Boxed-span metrics.** Every probe can target the precise final answer — the
  tokens inside `\boxed{...}` (`baseline.g0.answer_spans`, fallback last-K) —
  instead of the noisy last-K proxy. `run_g0` now records `iou_lh_boxed` /
  `lh_boxed_pointing` (LH), `iou_gl_boxed` / `vt_ratio_boxed` (GLIMPSE, computed
  from the *same* single backward via a third β mask) and `boxed_span_mode`.
  `analyze_g0` runs the headline looking-vs-using on the **boxed** span by default
  (`--span boxed|answer|first`), with span-matched pointing+vt keys.
* **LLM-judge correctness.** The rule grader under-scores free-form answers,
  biasing the correctness axis. `baseline.g0.judge_g0` re-grades a run's answers
  with an OpenAI-compatible judge (dummy key + `--judge-no-think` for self-hosted
  Qwen3) into a `judgments.jsonl` **sidecar**; `analyze_g0 --use-judge` overlays
  it. Per-task-type (Visual-CoT subset) breakdown of the verdict is always emitted
  (object/spatial vs OCR/doc often disagree).

```bash
# after a run_g0 / run_eagle_g0 run dir exists:
uv run python -m baseline.g0.judge_g0 --run-dir eval_outputs/g0/run1 \
    --judge-api-url http://localhost:8000/v1 --judge-model Qwen3-30B-A3B --judge-no-think
uv run python -m baseline.g0.analyze_g0 --run-dir eval_outputs/g0/run1 --use-judge --span boxed
```

## EAGLE-G0 — faithful **causal** attribution (looking + reliance)

LH/GLIMPSE read gradients/attention; **EAGLE** (CVPR 2026, arXiv 2509.22496,
vendored under `eagle_src/`) measures what the answer *causally depends on* by
perturbing image sub-regions (submodular insertion/deletion search). It is the
cross-check for "is the attention map even trustworthy?" Per sample/condition
(`eagle_probe.py`):

| metric | leg | meaning |
|--------|-----|---------|
| `iou_eagle` / `pointing_eagle` / `area_eagle` | looking (causal) | important-region map vs GT box |
| `visual_reliance` = org−baseline, `text_reliance` = baseline, `visual_fraction` | using | does the answer need the image (vs prior)? |
| `sufficiency` (insertion AUC) / `necessity` (deletion AUC) | causal sanity | keep-only / delete-region effect on the answer prob |

EAGLE is perturbation-based (~hundreds of forwards/sample) → small budget: the
image is downsized (`EAGLE_IMAGE_SIZE`), the answer is the `\boxed{}` span (last
`K` rows ⇒ `logits_to_keep=K`), and only `~n_regions` superpixels are searched.
`run_eagle_g0` also runs the cheap LH+GLIMPSE on the same rollout (default ON) so
the **EAGLE-vs-LH** comparison falls out of one pass.

```bash
# all 4 models (teacher + base-8B + OPD-2B + hint-OPD-2B), 2 GPUs each, concurrent:
export M=/.../models D=/.../datasets
JUDGE=1 JUDGE_API_URL=http://localhost:8000/v1 JUDGE_MODEL=Qwen3-30B-A3B \
  bash scripts/eagle_g0_multi.sh
# → eval_outputs/eagle_g0/{<name>/...}/  +  eval_outputs/eagle_g0/eagle_report.md
# single model:  MODEL=$M/CapCurriculum-8B GPUS=0,1 bash scripts/eagle_g0.sh
```

`analyze_eagle_g0` emits four tables (region accuracy w/ EAGLE-vs-LH;
looking-vs-using via `corr(correct, iou_eagle)` vs `corr(correct, visual_fraction)`;
hint mechanism plain-vs-hint; OPD/hint training ↑ image-reliance) + per-task-type.
Region division degrades gracefully: SLICO → skimage SLIC → numpy grid, so it runs
without `opencv-contrib`.
