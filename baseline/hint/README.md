# Grounding-Hint Distillation (GHD)

**Privileged-bbox On-Policy Distillation.** Vanilla OPD, but the frozen teacher —
and only the teacher — is privileged with the GT evidence bounding box. The student
rolls out from, and is scored on, the plain `(image, question)` prompt and **never
sees the box**, at train time or inference. Two privilege channels
(`TEACHER_PRIVILEGE_MODE`):

| mode | teacher input | privilege | knob |
|------|---------------|-----------|------|
| `hint` (default) | full image + box as **text coords** appended to the question | *direction* — where to look (not a sharper view) | — |
| `crop` | image **cropped to the box** (no text) | *zoom* — a higher-res view of the evidence region; real extra detail on high-res inputs | `CROP_PADDING` |

Both use the **GT** box, so the crop/hint is fixed and the student forward backprops
normally — **no RL needed** (that caveat only applies to a *student-generated* box).
The per-token gap the teacher opens up is attributable to grounding, which is what we
want the un-privileged student to internalize for visual-search (V\*Bench).
`hint` is the conservative spine (closest to what the student can do); `crop` is the
stronger-privilege sibling (bigger student↔teacher gap, more upside and more risk of
overshooting what the student can resolve in the full image).

## One training step

| stage | what happens |
|-------|--------------|
| 0. prompts | student = `system + user(image, question)`; teacher = `system + user(`**privileged image**`, question` [`+ hint`]`)`. `hint`: same image + text coords. `crop`: image cropped to the box. |
| 1. rollout | the **student** samples `y` on `(image, question)` — `no_grad`. On-policy: the loss lands on the tokens the student would actually emit. |
| 2. two forwards | teacher forwards `(privileged image, question, y)` → grounded `p_T` (`no_grad`); student forwards `(image, question, y)` → `p_S` (with grad). |
| 3. distill | per-token `KL(student‖teacher)` over the completion, pulling the un-privileged student toward the grounded teacher. Backward into the student only. |

The completion is identical in both forwards; `_completion_logits` slices it from
the *end* of each sequence, so the teacher's different prefix (longer hint text, or a
different-size cropped image → different #visual tokens) never misaligns the per-token KL.

> **Gating (deferred).** A natural next step is to apply the KL only on tokens
> where `logprob_teacher > logprob_student` — exactly the evidence-dependent
> tokens the teacher answers better *because* it knows the box. Not implemented
> yet (per the plan: ship the spine, add the gate once it shows signal). The
> `rollout/teacher_minus_student_logprob` curve already surfaces that gap.

## Files

| file | role |
|------|------|
| `opd_hint_collator.py` | `OPDHintDataCollator` (+ `HINT_TEMPLATE`, `format_bbox_hint`, `build_hint_teacher_messages`, `crop_to_bbox`). Builds the student prompt (via `OPDDataCollator`) **and** the privileged `teacher_prompt_*` (text hint or cropped image per `teacher_privilege_mode`). |
| `opd_hint_trainer.py`  | `OPDHintTrainer(OPDTrainer)` — overrides `compute_loss` to score the teacher on the privileged prompt; everything else inherited. Mode-agnostic. `local_hf` teacher only. |
| `../train_opd_hint.py` | entry point (`OPDHintScriptArguments` adds `--teacher_privilege_mode` / `--bbox_field` / `--filter_no_bbox` / `--hint_template` / `--hint_coord_decimals` / `--crop_padding`). |
| `../../scripts/train_opd_hint_qwen3_2b.sh` | launcher (same env-var knobs as `train_opd.sh` + `TEACHER_PRIVILEGE_MODE` / `CROP_PADDING`). |
| `inspect_dataset.py` | schema sniffer — columns, bbox shape (normalized vs pixel), image storage; suggests `ANSWER_FIELD` / `BBOX_FIELD`. |
| `sanity_check.py` | collation sanity check (hint/crop lands on the teacher only; crop geometry). |

Nothing in `vigos/` or the vanilla OPD files is modified — GHD is purely additive.

## Data

Needs an evidence-box column. Two datasets are wired in:

**`saliency-r1-8k`** (`peterant330/saliency-r1-8k`) — `problem` / `solution` /
`bbox` / `image`, where `bbox` is a string `"[x1,y1,x2,y2]"` normalized to `[0,1]`.
Set `ANSWER_FIELD=solution`. Loads as-is.

**`Visual-CoT`** (`deepcs233/Visual-CoT`) — per-domain `metadata/*.jsonl` with
`question` / `answer` / `bboxs` (nested **pixel** box) / `image` (basename) /
`width` / `height` / `dataset`. `baseline.opd_dataset.load_viscot_dataset` (auto-
detected by `load_opd_dataset`) folds the JSONLs into one dataset, renames
`question→problem`, normalizes `bboxs[0]` by `width`/`height` → the `bbox` string,
and resolves each image via a cached basename index over the **extracted** image
root (`VISCOT_IMAGE_ROOT`, default the dataset dir). Set `ANSWER_FIELD=answer`.
Extract the image tars first: `cd $D/Visual-CoT && cat cot_images_tar_split/* >
cot_images.tar && tar xf cot_images.tar` (then optionally delete the tar). Pre-flight
with `python -c "from baseline.opd_dataset import load_viscot_dataset as L;
print(L('$D/Visual-CoT')[0])"`.

`bbox` parsing is `baseline.probe.saliency_data.parse_bbox_norm` (order-normalizes,
clamps, drops degenerate boxes). `--filter_no_bbox true` (default) drops boxless
rows so every GHD step is privileged; set **`false` to keep parity** with a vanilla-OPD
baseline on the same dataset (then unboxed rows fall back to plain OPD). The
`hint_coverage` W&B curve reports the privileged fraction of each batch.

## Run

```bash
export M=/path/to/models D=/path/to/datasets
# hint mode (default)
PER_DEVICE_TRAIN_BATCH_SIZE=8 GRADIENT_ACCUMULATION_STEPS=8 FREEZE_VISION_TOWER=false \
MODEL_NAME_OR_PATH=$M/Qwen3-VL-2B-Instruct TEACHER_MODEL=$M/Vero-Qwen3I-8B \
DATASET_NAME=$D/saliency-r1-8k ANSWER_FIELD=solution \
bash scripts/train_opd_hint_qwen3_2b.sh
# crop mode (same command + one knob)
TEACHER_PRIVILEGE_MODE=crop ...same env... bash scripts/train_opd_hint_qwen3_2b.sh
```

The run tag auto-encodes the mode (`opd_hint_…` / `opd_crop_…`) so the two never
collide. Defaults (epochs/batch/lr/gen) **match `scripts/train_opd.sh`**, so GHD and
the vanilla-OPD baseline run the same number of steps — a clean A/B. `saliency-r1-8k`
is small (~8k boxed rows → ~16 steps/epoch at eff-batch 512); if you raise
`NUM_TRAIN_EPOCHS`, raise it on the OPD baseline too.

## The A/B/C that makes the point

The claim is "privileging the teacher with the box buys student grounding." Test it
against vanilla OPD on the **same data, schedule and teacher** — the only difference
is the teacher's privilege:

```bash
# vanilla OPD (control: same teacher, no box)
DATASET_NAME=$D/saliency-r1-8k ANSWER_FIELD=solution TEACHER_MODEL=$M/Vero-Qwen3I-8B \
  MODEL_NAME_OR_PATH=$M/Qwen3-VL-2B-Instruct bash scripts/train_opd.sh
# GHD hint (direction)
DATASET_NAME=$D/saliency-r1-8k ANSWER_FIELD=solution TEACHER_MODEL=$M/Vero-Qwen3I-8B \
  bash scripts/train_opd_hint_qwen3_2b.sh
# GHD crop (zoom)
DATASET_NAME=$D/saliency-r1-8k ANSWER_FIELD=solution TEACHER_MODEL=$M/Vero-Qwen3I-8B \
  TEACHER_PRIVILEGE_MODE=crop bash scripts/train_opd_hint_qwen3_2b.sh
```

Then eval all (and the untrained base) on **V\*Bench** with the deterministic
harness, plus the general suite to check for regressions:

```bash
MODEL_PATH=runs/<run>  bash scripts/eval_vstar.sh
```

GHD works iff hint and/or crop beat vanilla OPD on V\*Bench without regressing the suite.

## Sanity check

```bash
uv run python -m baseline.hint.sanity_check                       # text-only
uv run python -m baseline.hint.sanity_check --model $M/Qwen3-VL-2B-Instruct  # full
```
