# Evidence-Reliance Probe (the go/no-go "命门" experiment)

Tests one **dissociation**: standard OPD transfers the teacher's *output
behavior* — does it also transfer the teacher's *use of visual evidence*? This is
the cheap, no-train diagnostic that gives a clear **GO / STOP** before committing
to the method, and (if STOP) saves the training run.

It uses [`peterant330/saliency-r1-8k`](https://huggingface.co/datasets/peterant330/saliency-r1-8k):
each sample has a question, an answer (`solution`), and a GT **evidence bbox**
(field `bbox`, a string `"[x1,y1,x2,y2]"` **normalized 0–1**). The local copy has
**10 subsets** (field `dataset`); results are reported **per subset**:

| Tier | Subsets | Why |
|------|---------|-----|
| **Clean (headline)** | `textvqa` `textcap` `docvqa` `infographicsvqa` `gqa` `openimages` | answer is a specific token literally inside a **small** box (median area 0.005–0.15) → masking it cleanly removes the answer, floor ≈ 0, random control easy. The strongest Reliance signal. |
| Secondary | `cub` `vsr` (yes/no, ~50% floor) `flickr30k` `v7w` | yes/no compresses Reliance; `flickr30k` answers are free-form sentences (needs `--grader llm`, rule grading ≈ 0). Report separately. |

Two data hygiene knobs follow from the box-size distribution:
- **`--max-bbox-area` (default 0.5)** drops near-whole-image boxes (e.g. some
  `gqa`/`openimages`/`v7w` boxes ≈ 1.0) where an equal-area random mask can't be
  placed disjointly — those would dilute Reliance toward 0.
- For `flickr30k` (and phrase answers like v7w "On the table.") use `--grader llm`.

## Metrics (all pure pixel-space; no model-internal hooks)

For each model, every sample is answered under 4 image conditions — `full`,
`mask_evidence` (occlude the evidence box), `mask_random` (occlude an
**equal-shape** box elsewhere — cancels the generic "image corrupted" artifact),
`crop@pad` (crop to the evidence box). Then:

```
Reliance  = (Acc_full − Acc_mask_evidence) − (Acc_full − Acc_mask_random)
          =  Acc_mask_random − Acc_mask_evidence
Delta_RG  =  Acc_crop − Acc_full
```

- **Reliance ≫ 0** → the model causally uses *that* region. **Reliance ≈ 0** →
  shortcut / prior. (Only the *differential* drop counts, so it is robust to the
  occlusion being out-of-distribution.)
- **Delta_RG small** → the model already focuses on the evidence in the full
  image; **large** → it needs the crop handed to it.

Every number gets a **paired percentile bootstrap CI** over samples.

### Go / No-Go gate
- **GO** — teacher `Reliance` significantly > 0 (CI low > 0). Corroborated by
  teacher `Delta_RG` < student `Delta_RG`. → the teacher's edge is (at least
  partly) visual; proceed to Stage 1.
- **STOP** — teacher `Reliance ≈ 0` everywhere. → the edge is not visual; the
  framing is wrong; do not train.

## Pipeline

```bash
M=/home/web_server/antispam/project/houshihao/models     # box models dir
D=/home/web_server/antispam/project/houshihao/datasets   # box datasets dir
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# NOTE: with offline set, pass the LOCAL dataset dir (not the HF id), or it tries
# to reach the Hub and fails. saliency-r1-8k is at $D/saliency-r1-8k on the box.

# 0) CPU-only sanity: schema stats + bbox→image overlay montages. LOOK at these:
#    the red box must sit on the answer evidence before you trust any number.
uv run python baseline/probe/inspect_saliency.py \
    --dataset $D/saliency-r1-8k --num-sheets 16 --output-dir probe_outputs/inspect

# 1) One probe run per model (greedy, rule-graded, no API). ~hundreds of samples
#    × 8 conditions = minutes on one H800. Run candidates on different GPUs.
#    SUBSETS picks the clean tier; MAX_BBOX_AREA=0.5 (default) drops huge boxes.
SUB=textvqa,textcap,docvqa,infographicsvqa,gqa,openimages
CUDA_VISIBLE_DEVICES=0 DATASET=$D/saliency-r1-8k SUBSETS=$SUB MODEL_PATH=$M/MMR1-7B-RL  MODEL_NAME=MMR1-7B-RL  bash scripts/probe_stage0.sh
CUDA_VISIBLE_DEVICES=1 DATASET=$D/saliency-r1-8k SUBSETS=$SUB MODEL_PATH=$M/MMR1-3B-SFT MODEL_NAME=MMR1-3B-SFT bash scripts/probe_stage0.sh

# Check the printed Acc_full per model is sane (not ~0). If a reasoning model's
# Acc_full looks suppressed, give it its native prompt: --system-prompt "..." or
# --no-system-prompt (env: pass through the script).

# 2) Aggregate -> Reliance / Delta_RG + bootstrap CIs + GO/STOP verdict.
uv run python baseline/probe/analyze_stage0.py \
    --model teacher=probe_outputs/stage0/MMR1-7B-RL \
    --model student=probe_outputs/stage0/MMR1-3B-SFT \
    --teacher teacher --student student \
    --output probe_outputs/stage0/summary.json
```

Add more teacher candidates (Saliency-R1-7B, Qwen2.5-VL-7B, Qwen3-VL-8B) as extra
`--model label=...` inputs to compare the pattern across teachers.

## Files
| File | Role |
|------|------|
| `saliency_data.py`   | Load + parse the normalized `bbox` string; per-subset cap. |
| `image_ops.py`       | `mask_box` / area-matched `random_box_same_shape` / `crop_box` / overlay montage. |
| `inspect_saliency.py`| CPU sanity: schema stats + alignment montages. |
| `run_stage0.py`      | Per-model vLLM generation over all conditions + grading. |
| `analyze_stage0.py`  | Acc / Reliance / Delta_RG + paired bootstrap + GO/STOP. |

Minimum critical path for go/no-go = `full` + `mask_*` (Reliance). `crop`/Delta_RG
is corroborating. Stage 1 (1a output-convergence + 1b re-probe on a short
token-KL OPD checkpoint) and the mechanism probes (1c attention, 1d linear probe)
build on this harness.
