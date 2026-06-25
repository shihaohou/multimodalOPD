# GRPO teacher (ms-swift)

GRPO-train a **Qwen3-VL** teacher on Vision-SR1-47K to use as the stronger OPD
teacher. Uses [ms-swift](https://github.com/modelscope/ms-swift) (installed in a
**separate** venv, not the OPD env). These files are version-controlled here but
run against ms-swift.

## Why
OPD wants a stronger, same-family teacher. GRPO (verifiable answer-accuracy
reward) on the 8B base produces one. GRPO does not change the vocab, so the result
stays compatible with the 2B student for the top-k/full-vocab KL.

## One-time env (separate from OPD)
```bash
uv venv /root/shihao_project/swift-env/.venv --python 3.11
cd /home/web_server/antispam/project/houshihao && git clone https://github.com/modelscope/ms-swift.git
source /root/shihao_project/swift-env/.venv/bin/activate
cd ms-swift
uv pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install -e .
# vLLM only if you want fast rollout (verify Qwen3-VL support first):
# uv pip install vllm==0.11.0
```

## 1. Build the dataset (HF arrow -> ms-swift JSONL + images)
```bash
python baseline/teacher_grpo/prepare_vision_sr1.py \
  --output-jsonl /home/web_server/antispam/project/houshihao/datasets/vision_sr1_swift/train.jsonl \
  --image-dir   /home/web_server/antispam/project/houshihao/datasets/vision_sr1_swift/images \
  --limit 200     # smoke; omit for the full 47k
```

## 2. Smoke (a few steps, transformers rollout, LoRA)
```bash
source /root/shihao_project/swift-env/.venv/bin/activate
DATA=/home/web_server/antispam/project/houshihao/datasets/vision_sr1_swift/train.jsonl \
MODEL=/home/web_server/antispam/project/houshihao/models/Qwen3-VL-8B-Instruct \
TUNER_TYPE=lora USE_VLLM=false NPROC_PER_NODE=2 MAX_STEPS=4 NUM_GENERATIONS=4 \
OUTPUT_DIR=/home/web_server/antispam/project/houshihao/runs/teacher_grpo_smoke \
bash baseline/teacher_grpo/train_teacher_grpo.sh
```

## 3. Full run (2 epochs) — background
```bash
tmux new -s grpo
source /root/shihao_project/swift-env/.venv/bin/activate
# full-param (paper-faithful) needs zero3; LoRA is much cheaper:
TUNER_TYPE=lora NUM_TRAIN_EPOCHS=2 NPROC_PER_NODE=8 \
bash baseline/teacher_grpo/train_teacher_grpo.sh 2>&1 | tee grpo.log
```

## 4. Produce an HF checkpoint for the OPD teacher
- `TUNER_TYPE=full`: the output dir is already a full HF checkpoint → point OPD at it.
- `TUNER_TYPE=lora`: merge first:
  ```bash
  swift export --adapters <OUTPUT_DIR>/checkpoint-XXX --merge_lora true
  # -> <...>-merged ; use that as TEACHER_MODEL in OPD
  ```

Then in OPD: `TEACHER_MODEL=<merged-or-full-ckpt> bash scripts/train_opd.sh`
(student = Qwen3-VL-2B; both qwen3_vl, same vocab).

## Verified working setup (8×H800, driver 560.35.03 / CUDA 12.9, 2026-06-23)

swift env (separate from OPD), pinned versions that work together:
- **`transformers==4.57.1`** — NOT the 5.12.1 that ms-swift pulls by default.
  vLLM 0.11.0 crashes loading Qwen3-VL on transformers 5
  (`'Qwen3VLTextConfig' object has no attribute 'tie_word_embeddings'`). Downgrade:
  `uv pip install transformers==4.57.1` (ms-swift 4.4 still runs; Qwen3-VL still supported).
- `vllm==0.11.0`, `torch==2.8.0+cu128` (CUDA 13 / newer vLLM need driver ≥580 — we
  only have 560).
- `deepspeed==0.18.2`, `qwen_vl_utils==0.0.14`, `decord` (av).
- **Q5 triton patch** (HPC-X box) — vLLM JIT crashes with `ldconfig`
  `UnicodeDecodeError`; patch once (re-apply if triton is reinstalled):
  ```bash
  sed -i 's/decode()/decode("utf-8", errors="ignore")/' \
    /root/shihao_project/swift-env/.venv/lib/python3.11/site-packages/triton/backends/nvidia/driver.py
  ```

Confirmed: ms-swift auto-recognized the Qwen3-VL local path (no `--model_type`);
`vqa_accuracy`/`vqa_format` rewards fire; `USE_VLLM=true` step_time ≈16s vs ≈49s
for transformers rollout (2 GPUs, LoRA).

## Gotchas
- **`NPROC_PER_NODE` must equal the number of visible GPUs**, else ms-swift errors
  `DeepSpeed is not compatible with device_map`. Set `CUDA_VISIBLE_DEVICES`
  accordingly (e.g. `0,1` with `NPROC_PER_NODE=2`; all 8 with `NPROC_PER_NODE=8`).
- `reward_accuracy.py` is self-contained (no math_verify). Vision-SR1 is mixed
  (math + short VQA + options); tune `_is_match` if accuracy looks off.
- The GRPO `--system` prompt asks for `\boxed{}`; keep it consistent with how the
  teacher is later queried in OPD.
- **Benign exit crash**: after `End time of running main` + the checkpoint is
  saved, the process may abort with `Trying to free a pointer not allocated here`
  / SIGABRT during Python finalization (vLLM CUDA allocator + process-group
  teardown). The checkpoint is already written — this is cosmetic, but the exit
  code is non-zero, so don't gate follow-up steps on it.
