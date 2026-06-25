# CLAUDE.md

Guidance for AI agents working in this repository.

## What this repo is

This is **ViGOS** (paper: *Seeing Before Reasoning*), a training + evaluation
package for **multimodal On-Policy Self-Distillation (OPSD)** of vision-language
models (Qwen2.5-VL / Qwen3-VL). The local directory is named `multimodalOPD`
because the goal here is to extend it with an **On-Policy Distillation (OPD)**
baseline (see "OPD extension" below).

LoRA post-training only. Paper models: Qwen2.5-VL-3B/7B-Instruct.

## Core mechanism (read this before touching training code)

The method is **self-distillation**: the teacher and the student are the **same
model weights**. The only thing that differs between "student" and "teacher" is
the prompt (the teacher prompts carry privileged info). There is **no external
teacher model** anywhere in the codebase.

`ViGOSTrainer.compute_loss` (`vigos/trainer.py`) does the following per step:

1. **On-policy rollout**: the student samples one completion from the *student
   prompt* (image + question + format instructions, with a `<description>`
   assistant prefill). vLLM colocate is the default generation backend; the
   sampled tokens are then re-run through the train model to get gradients.
2. The completion is parsed into spans:
   `<description>…</description> <think>…</think> \boxed{…}`.
3. Three teacher forward passes, **all using the same `model`** but different
   prompts (built in `vigos/data_collator.py`):
   | Teacher    | Prompt context                              | Supervised tokens          | KL direction |
   |------------|---------------------------------------------|----------------------------|--------------|
   | perception | **image only** (no question, no answer)     | description span           | KL(teacher‖student) |
   | reasoning  | image + question + **reference answer**     | think + answer span        | KL(teacher‖student) |
   | reference  | image + question + **reference answer**, full format | malformed/fallback rows only | reverse KL(student‖teacher) |
4. Total loss = `λ_perception·L_perc + λ_reasoning·L_reas + λ_ref·L_ref`.

KL is **full-vocabulary, exact, per-token** (`vigos/losses.py::masked_kl_loss`),
with optional per-token clipping (`token_loss_clip`) and boundary-token clips
(`description_last_token_clip`, `reasoning_first_token_clip`). Loss is
re-normalized across DDP ranks by global active-token count
(`_distributed_masked_loss_with_stats`) so sparse masks aren't underweighted.

The "privileged answer leakage" the paper criticizes is exactly the reasoning
teacher seeing the reference answer in its prompt.

## File map

| File | Role |
|------|------|
| `vigos/train_vigos.py`   | Entry point. `ViGOSScriptArguments` dataclass = all CLI knobs; loads model + LoRA, builds collator + trainer. |
| `vigos/trainer.py`       | `ViGOSTrainer` (subclass of HF `Trainer`). Rollout, span parsing, mask building, teacher forwards, loss. ~2200 lines. |
| `vigos/data_collator.py` | Builds the 4 prompts (student / perception / reasoning / reference). Prompt strings live here. |
| `vigos/losses.py`        | `masked_kl_loss`, full-vocab KL, JS divergence. |
| `vigos/dataset_utils.py` | HF dataset loading / filtering (tiny-image filter, sample cap). |
| `vigos/answer_utils.py`  | `extract_boxed_content`, answer normalization for correctness metrics. |
| `vigos/eval_*.py`, `scripts/eval_vigos.py` | vLLM generation + LLM-judge (DeepSeek) eval over benchmark suite. |
| `scripts/train_vigos_qwen25_{3b,7b}.sh` | Accelerate+DeepSpeed launchers; every hyperparameter is an env-var override. |
| `scripts/merge_lora.py`  | Merge LoRA adapter into base model for standalone eval. |
| `configs/accelerate_zero2_lora_gpu_8.yaml` | ZeRO-2, 8 GPU. |

## How the paper's baselines map to the code

There is **no `--mode` flag**. The result-table rows are reproduced via the
lambda knobs / model choice:

- **Baseline** = the untrained base Qwen2.5-VL checkpoint (eval only, no training).
- **OPSD** = this trainer with the perception split effectively off — i.e. the
  privileged reasoning self-teacher only. Approximate it with
  `LAMBDA_PERCEPTION=0 LAMBDA_REF=0` (and the student prompt still emits the
  format; the distinguishing ViGOS ingredient is the image-only perception
  teacher, which `λ_perception=0` removes).
- **ViGOS** = full default (`λ_perception=1, λ_reasoning=1, λ_ref=2`).

## OPD extension (the current task)

**OPD = On-Policy Distillation** (Agarwal et al. GKD 2023; Thinking Machines
2025). The student samples on-policy rollouts and a **separate, frozen, stronger
teacher model** grades each token via **per-token reverse KL** `KL(student‖teacher)`.
Crucially the teacher is a *different model* and does **not** see a privileged
reference answer — that is the whole point of contrasting it with OPSD.

Difference from what this repo does today:

| | OPSD (current) | OPD (to add) |
|---|---|---|
| Teacher weights | same as student | separate frozen model |
| Teacher prompt | privileged (has answer) | same non-privileged student prompt |
| Supervised tokens | description / think / answer spans | full completion (typically) |
| Loss terms | 3 (λ_perc, λ_reas, λ_ref) | 1 reverse-KL term |

**Implemented in a separate top-level `baseline/` package (ViGOS code untouched;
`vigos/` is imported as a library):**

- `baseline/opd_data_collator.py` — `OPDDataCollator` + `OPD_SYSTEM_PROMPT`: the
  **unified system prompt** (paper appendix B.4: `<reason></reason>` CoT + `\boxed{}`)
  used across teacher GRPO / student OPD / eval for structural alignment. Student
  prompt = system + user(image + raw question), no privileged answer.
- `baseline/opd_trainer.py` — `OPDTrainer(ViGOSTrainer)`: overrides only
  `compute_loss`. Reuses inherited rollout / `_batched_teacher_completion_logits`
  / `masked_kl_loss` / `_distributed_masked_loss_with_stats` / answer-accuracy
  helpers. Teacher is a separate frozen module (`requires_grad_(False)`, `eval()`,
  moved to `accelerator.device`, never synced into vLLM).
- `baseline/opd_losses.py` — `masked_topk_kl_loss`: top-k (or full-vocab) KL,
  `direction` ∈ forward/reverse/jsd. Loss is configurable via
  `--opd_loss_mode {topk_kl,full_kl}` / `--opd_kl_direction` / `--opd_top_k`.
  **Default = `topk_kl` (top-100) + `reverse`** (top-k reverse KL `KL(student‖teacher)`;
  the OPD-ecosystem standard — verl/Uni-OPD/thunlp-OPD — capturing ~99% of the mass,
  and it avoids the full-vocab `exp`/`diff` that OOMs at micro-batch 8 on long
  completions. Set `--opd_loss_mode full_kl` for exact full-vocab KL via vigos
  `masked_kl_loss` — canonical but heavier). The `vllm_server` path must use
  `topk_kl`+`forward`, since the server returns only the teacher's top-k logprobs.
- `baseline/train_opd.py` — standalone entry point (`OPDScriptArguments`,
  `--teacher_model_name_or_path` required). Imports `vigos.*` as a library.
- `scripts/train_opd_qwen25_3b.sh` — launcher (runs `baseline/train_opd.py`;
  `TEACHER_MODEL` env required).
- `README_OPD.md` — the project README for this OPD work.
- `baseline/eval/` + `scripts/eval_opd.sh` — **general** multi-benchmark eval
  harness. Uses the dataset's own prompt (not the ViGOS format) so it evaluates
  any checkpoint; reuses generic `vigos.eval_utils` / `vigos.eval_benchmarks`
  helpers (sample extraction, LLM-judge prompts, scoring) + the OPD eval prompt.
  Pipeline: vLLM gen pass@k → `\boxed` extract → OpenAI-compatible judge →
  pass@k/avg@k → `responses/`,`judgments/`,`summary.json`. Full-FT writes a full
  checkpoint, so eval points straight at the run dir (no merge).
- Dedicated **deterministic** evals (no LLM judge; official per-benchmark metric;
  reuse `run_opd_eval`'s `generate_records`/`make_engine` under the same OPD prompt):
  `baseline/eval/run_mmvp_eval.py` + `scripts/eval_mmvp.sh` (MMVP pair accuracy);
  `baseline/eval/run_vqa_eval.py` + `baseline/eval/vqa_metrics.py` +
  `scripts/eval_vqa.sh` (POPE F1 / ChartQA relaxed accuracy / VQAv2 soft accuracy —
  `BENCHMARKS=pope,chartqa,vqav2`, one engine load for all three).
- `scripts/eval_suite.sh` + `baseline/eval/aggregate_suite.py` — **one-command full
  suite**: runs the LLM-judged group (`eval_opd.sh`: MathVista/MathVerse/MathVision/
  MMMU/MMMU-Pro×2/MMStar/HallusionBench) and the deterministic group (`eval_vqa.sh`:
  POPE/ChartQA/VQAv2), then merges both `summary.json`s into one table (POPE split by
  category +avg, MMMU-Pro split by sub-score +avg). Deterministic metrics are NOT
  judged; only the math/MCQ group uses the judge. `MULTI_K=true` adds a sampled pass
  for `pass@k`/`avg@k` on the judged group (`baseline/eval/passk.py`, unbiased Codex
  estimator — pass@8/16 from the same N samples). Note: greedy Acc@1 is the
  lmms-eval-comparable number; we prompt with the OPD training prompt (not lmms-eval's
  per-task templates), so absolute numbers differ — the harness is for relative compare.
- `baseline/teacher_grpo/` — recipe to GRPO-train a stronger **Qwen3-VL** teacher
  on Vision-SR1 via **ms-swift** (separate venv `/root/shihao_project/swift-env`,
  not the OPD env). `prepare_vision_sr1.py` (HF→ms-swift JSONL+images),
  `reward_accuracy.py` (self-contained `vqa_accuracy`/`vqa_format` ORM via
  `swift.rewards`), `train_teacher_grpo.sh` (Qwen3-VL-8B GRPO; transformers
  rollout default, vLLM optional). Output (full ckpt, or merged LoRA) becomes the
  OPD `TEACHER_MODEL`. Qwen3-VL line: student Qwen3-VL-2B + teacher Qwen3-VL-8B
  (both `qwen3_vl`, same vocab).
- `baseline/serve_teacher.py` + `baseline/teacher_client.py` +
  `scripts/serve_teacher_vllm.sh` — **`teacher_source=vllm_server`** (experimental):
  a separate vLLM server scores `prompt_token_ids+completion` with
  `prompt_logprobs=top_k` and returns top-k logprobs; the trainer computes forward
  top-k KL via `masked_topk_kl_loss_from_teacher_topk`. No per-GPU teacher replica
  → enables 32B/72B teachers. Only `topk_kl`+`forward`. `teacher_source=local_hf`
  stays the default. Needs GPU validation (multimodal `prompt_token_ids` path).

Key reuse trick: `_batched_teacher_completion_logits(model, jobs)` already runs
under `no_grad()/eval` and its adapter-disable context degrades to a no-op for a
non-PEFT module — so `OPDTrainer` passes `self.teacher_model` to it directly.

The student uses **full fine-tuning by default** (`--finetuning_mode full`, like
Vision-OPD; `lora` still available). Full-FT runs on DeepSpeed ZeRO-2
(`configs/accelerate_zero2_gpu_8.yaml`) and `save_model` writes a full checkpoint
(no LoRA merge before eval). Default script uses LR `2e-6`, micro-batch 1, grad
accum 4.

Constraints: teacher must share the student's tokenizer/vocab (same family); the
teacher must run a **local HF forward** (vLLM logprob is top-k only); the frozen
teacher is replicated per GPU, so budget memory (3B/4B student + 7B teacher fits
on 8×A100-80G under ZeRO-2; 7B full-FT student needs ZeRO-3 offload + an
unpartitioned-teacher fix; ≥32B teacher needs TP/top-k KL).
Run: `DATASET_NAME=... TEACHER_MODEL=... bash scripts/train_opd_qwen25_3b.sh`.

Roadmap (per user): general multi-benchmark eval framework, and model/attention
architecture changes on the student — keep new work in OPD-specific files.

## Environment & commands

```bash
uv sync --python 3.11          # PyTorch 2.8, Transformers 4.57.1, TRL 0.26, vLLM 0.11
# Train (8×A100 assumed):
DATASET_NAME=LMMs-Lab-Turtle/Vision-SR1-47K bash scripts/train_vigos_qwen25_3b.sh
# Quick smoke test:
DATASET_NAME=... MAX_STEPS=5 SAVE_STEPS=5 REPORT_TO=none bash scripts/train_vigos_qwen25_3b.sh
# Merge + eval:
uv run python scripts/merge_lora.py --adapter runs/<run>/checkpoint-XXX --output runs/<run>_merged --overwrite
MODEL_PATH=runs/<run>_merged SKIP_JUDGE=true bash scripts/eval_vigos.sh
```

Training data field requirements: `problem`, `images`, `answer` (+ optional
`problem_id`). Dataset: `LMMs-Lab-Turtle/Vision-SR1-47K`.

## Conventions / gotchas

- Every training hyperparameter is an env-var override in the shell scripts;
  prefer adding knobs there rather than hardcoding.
- Tokenizer padding is **left** (set in the collator) — required for rollout.
- `use_cache` must stay `False` while gradient checkpointing is on.
- When changing the loss, preserve `_distributed_masked_loss_with_stats`
  normalization or sparse-mask gradients will be wrong under DDP.
- vLLM weights are resynced to the train policy via `ViGOSVLLMSyncCallback`
  every `vllm_sync_frequency` steps — a new teacher model must NOT be synced.
