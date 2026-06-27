# Grounding-Hint Distillation (GHD)

**Privileged-bbox On-Policy Distillation.** Vanilla OPD, but the frozen teacher —
and only the teacher — is handed the GT evidence bounding box as a *text
coordinate hint* appended to the question. The student rolls out from, and is
scored on, the plain `(image, question)` prompt and **never sees the box**, at
train time or at inference.

The image is **not cropped or upsampled** for the teacher. It gets *direction*
("look at region `[x1,y1,x2,y2]`"), not *information* (a sharper view). So any
per-token gap the teacher opens up over the student is attributable to
**grounding** — knowing where the answer lives — which is exactly what we want the
un-hinted student to internalize for visual-search benchmarks (V\*Bench).

## One training step

| stage | what happens |
|-------|--------------|
| 0. prompts | student = `system + user(image, question)`; teacher = `system + user(image, question, ` **bbox hint** `)`. Same image, same resolution. |
| 1. rollout | the **student** samples `y` on `(image, question)` — `no_grad`. On-policy: the loss lands on the tokens the student would actually emit. |
| 2. two forwards | teacher forwards `(image, question, bbox, y)` → grounded `p_T` (`no_grad`); student forwards `(image, question, y)` → `p_S` (with grad). |
| 3. distill | per-token `KL(student‖teacher)` over the completion, pulling the un-hinted student toward the grounded teacher. Backward into the student only. |

The completion is identical in both forwards; `_completion_logits` slices it from
the *end* of each sequence, so the teacher's longer (hint-bearing) prefix never
misaligns the per-token KL.

> **Gating (deferred).** A natural next step is to apply the KL only on tokens
> where `logprob_teacher > logprob_student` — exactly the evidence-dependent
> tokens the teacher answers better *because* it knows the box. Not implemented
> yet (per the plan: ship the spine, add the gate once it shows signal). The
> `rollout/teacher_minus_student_logprob` curve already surfaces that gap.

## Files

| file | role |
|------|------|
| `opd_hint_collator.py` | `OPDHintDataCollator` (+ `HINT_TEMPLATE`, `format_bbox_hint`, `build_hint_teacher_messages`). Builds the student prompt (via `OPDDataCollator`) **and** the privileged `teacher_prompt_*`. |
| `opd_hint_trainer.py`  | `OPDHintTrainer(OPDTrainer)` — overrides `compute_loss` to score the teacher on the privileged prompt; everything else inherited. `local_hf` teacher only. |
| `../train_opd_hint.py` | entry point (`OPDHintScriptArguments` adds `--bbox_field` / `--filter_no_bbox` / `--hint_template` / `--hint_coord_decimals`). |
| `../../scripts/train_opd_hint_qwen3_2b.sh` | launcher (same env-var knobs as `train_opd.sh`). |
| `sanity_check.py` | collation sanity check (hint lands on the teacher only; same image both sides). |

Nothing in `vigos/` or the vanilla OPD files is modified — GHD is purely additive.

## Data

Needs an evidence-box column. **`peterant330/saliency-r1-8k`** ships
`problem` / `solution` / `bbox` / `image`, where `bbox` is a string
`"[x1, y1, x2, y2]"` normalized to `[0,1]` (reused via
`baseline.probe.saliency_data.parse_bbox_norm`, which order-normalizes, clamps,
and drops degenerate boxes). Set `ANSWER_FIELD=solution`. Rows without a parseable
box are dropped by default (`--filter_no_bbox true`); the `hint_coverage` W&B
curve reports the fraction of each batch that was actually privileged.

## Run

```bash
export M=/path/to/models D=/path/to/datasets
PER_DEVICE_TRAIN_BATCH_SIZE=8 GRADIENT_ACCUMULATION_STEPS=8 FREEZE_VISION_TOWER=false \
MODEL_NAME_OR_PATH=$M/Qwen3-VL-2B-Instruct TEACHER_MODEL=$M/Vero-Qwen3I-8B \
DATASET_NAME=$D/saliency-r1-8k ANSWER_FIELD=solution \
RUN_CONFIG=opd_hint_qwen3_vero_8b2b_fullft_saliency-r1-8k \
bash scripts/train_opd_hint_qwen3_2b.sh
```

`saliency-r1-8k` is small (~8k boxed rows). At eff-batch 512 that is ~16
steps/epoch, so `NUM_TRAIN_EPOCHS` defaults to 3 here — raise it (or lower the
batch) for a longer curve.

## The A/B that makes the point

GHD's claim is "the privileged *where-to-look* hint on the teacher buys student
grounding." Test it against vanilla OPD on the **same data and schedule** — the
only difference is whether the teacher sees the box:

```bash
# GHD (privileged teacher)
DATASET_NAME=$D/saliency-r1-8k ANSWER_FIELD=solution TEACHER_MODEL=$M/Vero-Qwen3I-8B \
  bash scripts/train_opd_hint_qwen3_2b.sh
# vanilla OPD (same teacher, no box) — the control
DATASET_NAME=$D/saliency-r1-8k ANSWER_FIELD=solution TEACHER_MODEL=$M/Vero-Qwen3I-8B \
  MODEL_NAME_OR_PATH=$M/Qwen3-VL-2B-Instruct bash scripts/train_opd.sh
```

Then eval both (and the untrained base) on **V\*Bench** with the deterministic
harness, plus the general suite to check for regressions:

```bash
MODEL_PATH=runs/<ghd_run>  bash scripts/eval_vstar.sh
MODEL_PATH=runs/<opd_run>  bash scripts/eval_vstar.sh
```

GHD works iff it beats vanilla OPD on V\*Bench without regressing the suite.

## Sanity check

```bash
uv run python -m baseline.hint.sanity_check                       # text-only
uv run python -m baseline.hint.sanity_check --model $M/Qwen3-VL-2B-Instruct  # full
```
