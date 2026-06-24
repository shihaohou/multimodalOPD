# OPD Progress

Living status doc for the **On-Policy Distillation (OPD)** baseline (the
`baseline/` package on top of the ViGOS framework). Read this first when picking
up planning. See `README_OPD.md` for usage and `CLAUDE.md` for the architecture.

_Last updated: 2026-06-24._

## Current status: training validated on GPU ✅

OPD full fine-tuning runs **stably on the 8×H800 box**. Reverse-KL
`KL(student‖teacher)`, full vocabulary, separate frozen same-family teacher, vLLM
colocate rollout, DeepSpeed ZeRO-2, bf16.

Two model lines, both confirmed stable (no NaN, loss finite & trending down):

| Line | Student (full-FT) | Teacher (frozen) | Vocab | Notes |
|------|-------------------|------------------|-------|-------|
| Qwen2.5-VL | 3B-Instruct | 7B-Instruct | 151936 / 152064 → min | ViT stable at batch 32 |
| Qwen3-VL | 2B-Instruct | 8B-Instruct | shared | redesigned ViT, more bf16-stable |

**Running now:** Qwen3-VL `8B → 2B` OPD, full-FT, ViT unfrozen, 8 GPU, reverse-KL,
WandB online. Output → `runs/opd_qwen3_8b_to_2b/`.

**Config (paper Vision-OPD/VGS Table 4 aligned + throughput-tuned, all script
defaults):** AdamW lr 1e-6, weight_decay 1e-2, constant schedule (no warmup),
**global batch 512 = per_device 8 × grad_accum 8 × 8 GPU** (per_device 8 batches
the vLLM rollout for throughput; `VLLM_GPU_MEMORY_UTILIZATION=0.25` since 0.30 OOMs
at per_device 8), rollout temp/top_p 1.0 (no top-k), max prompt/response
16384/2048, SAVE_STEPS 5 (keep all ckpts for the acc curve). WandB metrics:
`loss_opd` (= reverse KL), `completion_length` (mean response tokens),
`answer_accuracy`, `completion_token_ratio`.

## The NaN saga — root cause & resolution

Reverse-KL full-FT NaN'd `loss_opd` within 3 steps. After several wrong
hypotheses, the **real root cause was an effective batch of ~1** (the run was on a
single GPU because `CUDA_VISIBLE_DEVICES`/`NUM_PROCESSES` weren't pinned and a
fresh shell on the box defaults to one visible GPU). At batch 1, a single image's
Qwen2.5-VL ViT (`visual.patch_embed`) gradient spike (grad_norm ~448) overflows in
bf16 → poisons the Adam state → NaN weights, cascading to the LLM. **At a real
8-GPU batch of 32 the spikes average out (grad_norm ~3–20) and full FT incl. the
ViT is stable** — which is why Vision-OPD (verl, batch 96) never needs to freeze.

How it was found: the `_report_opd_nan` probe (student-forward NaN → param scan →
weights non-finite, `visual.patch_embed.proj.weight` first) + noticing
`answer_accuracy` read 0.0/1.0 (denominator 1, i.e. one sample) instead of n/32.

**Lesson: when an on-policy / full-FT run NaNs, check the effective batch / actual
GPU count FIRST.** A few code hypotheses (fp32 KL, diff-clamp, valid-vocab mask)
were spent before finding it was a run-config issue.

## Operational gotcha: stale shell env vars (bit us repeatedly)

The box's shells carry stale exports between sessions/venvs. Each silently
degraded a run:
- `CUDA_VISIBLE_DEVICES` unpinned / single GPU → effective batch 1 → NaN.
- `WANDB_MODE=offline` → no live logs (script's `${WANDB_MODE:-online}` only
  defaults when *unset*).
- `MAX_STEPS`, `MAX_TRAIN_SAMPLES` → run stops after 2 steps / dataset capped.
- `VIRTUAL_ENV=.../swift-env/.venv` → wrong venv (uv ignores it, but noisy).

**Always launch a real run with the explicit block** (see `README_OPD.md`
"Verified multi-GPU launch"): pin `CUDA_VISIBLE_DEVICES=0..7 NUM_PROCESSES=8`,
`unset MAX_STEPS MAX_TRAIN_SAMPLES`, set `WANDB_MODE`, and **confirm the
`[OPD] num_processes(world_size)=8 ... effective_batch=32` startup print**.

## Code changes this cycle (all on `main`)

Kept (earned their keep):
- **`_report_opd_nan` probe** — fires only on a non-finite step; localizes
  student-forward / teacher-forward / KL-math NaN + scans params. Permanent net.
- **`[OPD] num_processes=...` startup print** — self-documents the real world size.
- **`del teacher_logits` bug fix** — was unconditional, crashed `vllm_server` mode.
- Cheap hardening: fp32 KL, ±20 logprob-diff clamp, symmetric token clip, LR warmup.

Defaults / cleanup:
- **`freeze_vision_tower=False`** (full FT incl. ViT, like Vision-OPD); knob kept
  as a small-batch / single-GPU fallback.
- **Removed the valid-vocab mask** — GPT's hypothesis; a no-op on a trained model.
- **`WANDB_MODE=online`** default in the launcher.

## Next / open items (for new planning)

- [ ] **Finish + eval the Qwen3 8B→2B run** (`baseline/eval/`, `scripts/eval_opd.sh`:
      pass@k / avg@k, LLM judge). Compare vs base 2B (no-train) baseline.
- [ ] **GRPO teacher**: swap `TEACHER_MODEL` to the GRPO-trained (merged) Qwen3-VL-8B
      for the full-method result; contrast vs base-8B teacher.
- [ ] **Qwen2.5 3B←7B** full run for the cross-line comparison.
- [ ] Model / attention architecture experiments on the student (keep in `baseline/`).
- [ ] PG/GRPO OPD variant (reverse-KL-as-reward), like verl PG OPD.
- [ ] vllm_server teacher path — still needs GPU validation (multimodal
      `prompt_token_ids`), enables ≥32B teachers.

## Pointers

- Usage / knobs: `README_OPD.md`
- Architecture / method: `CLAUDE.md`, `docs/OPD_BASELINE_PLAN.md`
- Launcher: `scripts/train_opd_qwen25_3b.sh` (generic; set model paths per line)
- Box paths / run commands: see agent memory `opd-run-paths-h800`
