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

Then in OPD: `TEACHER_MODEL=<merged-or-full-ckpt> bash scripts/train_opd_qwen25_3b.sh`
(student = Qwen3-VL-2B; both qwen3_vl, same vocab).

## Notes / to verify on GPU
- ms-swift must recognize the **Qwen3-VL** local path; if auto-detect fails add
  `--model_type` (check `swift rlhf --help` / ms-swift model registry for the name).
- `reward_accuracy.py` is self-contained (no math_verify). Vision-SR1 is mixed
  (math + short VQA + options); tune `_is_match` if accuracy looks off.
- vLLM rollout (`USE_VLLM=true`) needs vLLM with Qwen3-VL support; otherwise keep
  transformers rollout.
- The GRPO `--system` prompt asks for `\boxed{}`; keep it consistent with how the
  teacher is later queried in OPD.
