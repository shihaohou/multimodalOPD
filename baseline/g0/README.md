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

```bash
# on the GPU box; needs eager attention + grad → single H800/A100-80G.
STUDENT_MODEL=$M/Qwen3-VL-2B-Instruct \
TEACHER_MODEL=$M/Qwen3-VL-8B-Instruct \
SUBSETS=textvqa,docvqa,gqa,openimages LIMIT=80 CALIB_LIMIT=40 \
RUN_NAME=run1 bash scripts/g0_diag.sh
# → eval_outputs/g0/run1/{records.jsonl, head_stats_*.json, analysis.json, report.md, figs/, viz/}
```

Or the two steps directly:

```bash
uv run python -m baseline.g0.run_g0 --student-model ... --teacher-model ... \
    --output-dir eval_outputs/g0/run1 --subsets textvqa,docvqa,gqa,openimages --limit 80
uv run python -m baseline.g0.analyze_g0 --run-dir eval_outputs/g0/run1
```

## Outputs → the four analyses (`report.md`)

1. **Head usability** (8B & 2B) — best per-head mean IoU, the selected heads +
   their layers, assembled `IoU_LH`. Gates the LH-box / label-free plans.
2. **Student looking-vs-using** (KEY) — the 2×2 of `IoU_LH` (high/low) ×
   correctness on C3, plus `vt_ratio` for right vs wrong. Heavy
   *looked-right-but-wrong + low vt_ratio* ⇒ **using failure** (stay output-level);
   `IoU_LH` that tracks correctness ⇒ **looking failure** (where-to-look has headroom).
3. **Hint mechanism** (C1 vs C2, paired) — ΔIoU_LH ≫ 0 with Δacc > 0 ⇒
   *attentional*; Δacc > 0 with ΔIoU_LH ≈ 0 ⇒ *non-attentional / output-level*.
4. **Teacher-vs-student gap** (C1 vs C3) — is the gap mainly localization
   (`IoU_LH`) or attribution (`vt_ratio`)? Tells us what OPD must transfer.

## Knobs / gotchas

* `MAX_PIXELS` (default 602112 ≈ 768 visual tokens) is the OOM lever — the
  GLIMPSE grad forward's memory scales ~`S²`. Lower it if the 8B grad OOMs; raise
  it for finer grids. The same cap is used for calibration and eval so head
  indices stay comparable.
* `--top-k-heads` (3), `--min-layer` (2, ignore early layers when selecting),
  `--lh-sigma` (1.0) for LH; `--glimpse-lambda` / `--glimpse-lambda-depth` /
  `--glimpse-layers` for GLIMPSE; `--threshold {mean,top_frac}` for the maps.
* Greedy decoding by default (reproducible); `SAMPLE=1` to sample.
* CPU self-tests for the geometry/aggregation logic:
  `python -m baseline.g0.metrics`.
