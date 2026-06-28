# Locate-Once Grounding (LOG) — Fork A v2

**Hidden-hint OPD + a student box RL term.** The verified hidden-hint distillation
spine ([`baseline/hint`](../hint/README.md), the B1 result) plus an *explicit,
student-generated* evidence box trained by RL. The student opens its `<think>` with a
single `<box>[x1,y1,x2,y2]</box>` (**no crop** — same pixels, same resolution), then
reasons and answers. The box is a *commitment to where to look*; RL pushes the
model's internal attention to honor it.

Why no crop / why the box is RL-only (the empirical priors this is built on):

* hidden-hint-OPD **>** hint-OPD **>** plain OPD, and verbalized-hint OPD *collapses*
  → the gain is from *knowing the region*, not from zooming, and **verbalizing
  coordinates is harmful**. So the **teacher** uses the box silently (hidden hint) and
  the **student's** box is only an RL handle — it never enters the OPD target.

## One training step — two span-decoupled gradients on one student forward

```
L = lambda_opd * L_OPD(answer/reasoning span, box span MASKED)   # how to answer
  + lambda_rl  * L_RL(box coordinate span)                       # where to look
```

| stage | what happens |
|-------|--------------|
| prompts | **student** = locate-once (`<box>` then reason then `\boxed{}`); **teacher** = plain think prompt + the GT box as a *silent* hint (forbidden from verbalizing it). Asymmetric: the student is *asked* to locate, the teacher is *handed* the box. |
| rollout | the student samples **G** completions per prompt (`group_size`, the GRPO group) — `no_grad`, vLLM. |
| OPD | per-token `KL(student‖teacher)` over the completion **with the `<box>…</box>` span removed from the mask** — the hidden-hint teacher emits no box, so scoring the student's box tokens under it would push the student to stop emitting boxes, fighting the RL term. |
| RL | reward = `IoU(student_box, GT_box)` **gated by answer correctness** (DeepEyes-style: only a correct rollout earns localization credit) → group-normalized advantage → `-A·logπ` on the box coordinate tokens. |

The two **supervised output positions** are disjoint (OPD on non-box tokens, RL on
box-coordinate tokens; the literal `<box>`/`</box>` tags get neither), so no token gets
a direct OPD *and* RL gradient — that decoupling is the core design decision. (They are
not fully independent: the answer tokens still attend back to the box text in context,
so OPD's later-token loss is implicitly conditioned on the student's box — the box stays
in context and is only removed from the *loss*.)

A box that appears at/after `\boxed{}` ("answer then locate") is masked from OPD but
earns **no RL/reward** (`late_box_rate` logs it). RL trains the *coordinates* of an
emitted box, not whether to emit one — emission relies on the prompt and, if
`box_coverage`/`box_present_rate` come up low, a short SFT cold-start (Option β); the RL
group baseline is mean-centered, so it cannot by itself bootstrap box emission.

> **Position gate (deferred, off by default).** `--kl_position_gate true` applies the
> OPD KL only where the teacher gives the sampled token higher logprob than the student
> (the evidence-dependent tokens). Per the plan, ship the spine first; gate later.

## Files

| file | role |
|------|------|
| `prompts.py` | `LOCATE_SYSTEM_PROMPT` (import-light, so CPU sanity checks need no training stack). |
| `locate_rl.py` | Pure RL math (no model): `parse_student_box`, `iou_norm`, `group_normalize_advantage`. |
| `opd_locate_collator.py` | `OPDLocateDataCollator(OPDHintDataCollator)` — group expansion + the locate-once student prompt; teacher stays on the hidden-hint prompt (decoupled via `teacher_system_prompt`). Emits `group_ids` + `locate_gt_boxes`. |
| `opd_locate_trainer.py` | `OPDLocateTrainer(OPDHintTrainer)` — overrides `compute_loss`: box-span masking on OPD + the GRPO box RL term. `local_hf` teacher only. |
| `../train_opd_locate.py` | entry point (`OPDLocateScriptArguments` adds `--group_size` / `--lambda_rl` / `--rl_reward` / `--rl_ungated_weight` / `--rl_normalize_adv` / `--kl_position_gate` / `--locate_system_prompt`). |
| `../../scripts/train_opd_locate_qwen3_2b.sh` | launcher (GHD knobs + `GROUP_SIZE` / `LAMBDA_RL` / `RL_REWARD`). |
| `sanity_check.py` | CPU-only RL math + (with `--model`) the collator group-expansion / prompt-asymmetry check. |

The only change outside this package is a backward-compatible `teacher_system_prompt`
field on `OPDHintDataCollator` (default `None` → unchanged GHD behaviour) so the
student and teacher turns can carry different system prompts. `vigos/` is untouched.

## Data

Same evidence-box datasets as GHD (`--bbox_field`, default `bbox`): **`saliency-r1-8k`**
(`ANSWER_FIELD=solution`) or **`Visual-CoT`** (`ANSWER_FIELD=answer`; extract the image
tars and set `VISCOT_IMAGE_ROOT` — see the [GHD README](../hint/README.md#data)). The
GT box is used **twice**: silently by the teacher (the hidden hint) and as the IoU
target for the RL reward. `--filter_no_bbox true` (default) keeps only boxed rows — RL
needs a target.

## Run (B2)

```bash
export M=/path/to/models D=/path/to/datasets
PER_DEVICE_TRAIN_BATCH_SIZE=1 GROUP_SIZE=8 GRADIENT_ACCUMULATION_STEPS=8 \
MODEL_NAME_OR_PATH=$M/Qwen3-VL-2B-Instruct TEACHER_MODEL=$M/Vero-Qwen3I-8B \
DATASET_NAME=$D/saliency-r1-8k ANSWER_FIELD=solution \
bash scripts/train_opd_locate_qwen3_2b.sh
```

`per_device_train_batch_size` counts **prompts**; the collator expands each into
`GROUP_SIZE` rollouts (the GRPO group). Effective rollouts/step = `per_device ×
GROUP_SIZE × grad_accum × world`. Keep per-device small and let `GROUP_SIZE` batch the
rollout. The v2 doc's models are MMR1-3B-SFT (student) / MMR1-7B-RL (teacher) — set
`MODEL_NAME_OR_PATH` / `TEACHER_MODEL` to use them (any same-vocab family works).

Smoke test (one box, a couple of steps):

```bash
MAX_STEPS=2 SAVE_STEPS=2 REPORT_TO=none NUM_PROCESSES=1 CUDA_VISIBLE_DEVICES=0 \
MAX_TRAIN_SAMPLES=8 PER_DEVICE_TRAIN_BATCH_SIZE=1 GROUP_SIZE=4 \
MODEL_NAME_OR_PATH=$M/Qwen3-VL-2B-Instruct TEACHER_MODEL=$M/Qwen3-VL-8B-Instruct \
DATASET_NAME=$D/saliency-r1-8k ANSWER_FIELD=solution \
bash scripts/train_opd_locate_qwen3_2b.sh
```

Then inspect `runs/<run>/completion_samples/*.md`: does the student actually emit
`<box>[…]</box>` at the head of `<think>`? W&B curves to watch: `box_present_rate` (any
`<box>`) and `box_coverage` (a parsed valid box) → near 1.0; `late_box_rate` → ~0;
`iou_correct_mean` (the gated signal) should rise; `nonzero_adv_rate` (RL has signal);
`mean_box_area` (collapse monitor); `loss_rl`, `loss_opd`. If `box_coverage` is low,
cold-start (Option β) or raise `RL_UNGATED_WEIGHT` (warmup: rewards a well-placed box
even when the answer is wrong, default 0).

## Baselines / decision (why this fork exists)

| arm | config | role |
|-----|--------|------|
| B0 | vanilla OPD (no hint) | floor — `scripts/train_opd.sh` |
| **B1** | hidden-hint OPD, box-free | the verified spine — `scripts/train_opd_hint_qwen3_2b.sh` |
| **B2** | B1 + locate-once + box RL | **this package** |

Same data / teacher / schedule across arms. Ablations via knobs: `LAMBDA_RL=0`
(box emitted but unrewarded), `RL_REWARD=iou` (ungated), `RL_UNGATED_WEIGHT>0`
(gated + warmup mix), `KL_POSITION_GATE=true`.

**Go/No-Go** (eval on V\*Bench with `scripts/eval_vstar.sh`, + the suite for regressions):

| gate | pass criterion | else |
|------|----------------|------|
| G0 | the student's box has vision-conditioning (probe: coords move with the image) | retreat to B1 |
| G1 | B2 > B0 | mechanism broken |
| **G2** | **B2 > B1 on V\*Bench** | **ship B1, drop the RL/box** |
| G3 | the gain is grounding not guessing (IoU↑ **and** acc↑, image-dependence holds) | leaderboard artifact |

## Sanity check

```bash
uv run python -m baseline.locate.sanity_check                         # text-only RL math
uv run python -m baseline.locate.sanity_check --model $M/Qwen3-VL-2B-Instruct  # + collator
```
