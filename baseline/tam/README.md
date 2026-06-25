# TAM → OPD: Token Activation Map as the visual-evidence channel

`baseline/tam/` adds a **visual-space supervision channel** on top of vanilla
OPD's token-KL. It is the implementation of the *TAM → OPD* migration doc.

> **One line.** Token-KL teaches the student *what token to generate* (behavior).
> TAM-align teaches *where in the image the evidence for that token lives* (visual
> evidence). Two channels, added: `L = L_OPD + λ·L_TAM`.

This is a sibling of `baseline/evidence/` (the Saliency-R1 → OPD port) but a
**different, much cheaper** visual channel — see the comparison below.

## Why TAM is a near-free visual channel for OPD

The base map is the **logit-lens** of the last-layer hidden state at the visual
token positions, read along the *generated token's* unembedding row (TAM paper
Eq.1):

```
a_i = ReLU( F^v · W[y_i]^T )      # [n_v]  per rolled-out token i
```

* `F^v` = `hidden_states[-1]` at the image-placeholder positions `[n_v, d]`.
* `W[y_i]` = the `lm_head` row for the rolled-out token id `y_i`.

Four properties that make it ideal for OPD:

1. **Free cross-size bridge — no PCA.** The 8B teacher (`d=3584`) and 2B student
   (`d=2048`) both collapse to the *same* `n_v`-dim per-position scalar map. We
   only ever compare those maps, never `F^v` directly. The shared vocabulary +
   `lm_head` *are* the fixed interface. Preconditions: same **tokenizer** + same
   **patch grid** (the Qwen3-VL family satisfies both → 8B→2B is the natural pair;
   a cross-family teacher would break the bridge).
2. **Gradient flows into the visual representation.** `a_i = ReLU(F^v_θ · W[y_i]^T)`
   is differentiable in `F^v_θ`. With `lm_head` **detached** (`tam_detach_lm_head`,
   doc §3) the gradient lands cleanly on `F^v → LLM → projector/ViT` — exactly the
   "act on the visual end" signal token-KL can't give.
3. **Aligns in image space**, not hidden-dim space — the target is semantically
   clear ("look here") and maps to grounding metrics (Obj-IoU).
4. **Almost zero extra cost — forward only.** The logit-lens needs **no attention
   weights** (it is *not* the attention-based TAM variant), so it runs under
   **FlashAttention/SDPA** with no eager switch and no hooks. On top of the OPD
   forward it is a few `n_v`-sized matmuls.

## What we use from TAM (`/Users/houshihao/project/code/TAM-main`)

| TAM module | role | in the loss? | here |
|---|---|---|---|
| **Base map** (Eq.1) | logit-lens visual relevance | ✅ core, differentiable | student grad path |
| **ECI** (Eq.2,4,5) | subtract context-token interference | ✅ but **stop-grad** correction | closed-form `s = <a,E>/<E,E>`, ECI detached |
| **RGF** (Eq.6,7) | rank-Gaussian denoise | ❌ ranking non-differentiable | replaced by a fixed **Gaussian blur**; RGF left to offline viz |

## Loss (`tam_losses.py`)

```
ã_i = ECI(a_i)                         # cleaned base map (ReLU)
p_i = normalize( gaussian_blur(ã_i) )  # blur both sides equally; L2 (cosine) or sum-to-1 (js/l1)
L_TAM = mean_i  g_i · d( p^stu_i , sg[p^tea_i] )
```

* `d` = `cosine` (`1-cos`, default — robust on non-negative maps), `js`
  (Jensen-Shannon, the doc's theoretical default), or `l1` (total variation).
* `g_i` = **teacher-map concentration gate** (`sigmoid((h0 − H_norm)/τ)`):
  down-weights tokens whose teacher map is diffuse (function words point nowhere).
  This is the doc's position gate #1 — cheap, no extra forward. The full loss is
  `L = λ_opd·L_OPD + λ_tam·L_TAM`; **token-KL is never removed**.

## Files

| file | role |
|---|---|
| `tam_engine.py` | `compute_tam_token_maps` (differentiable base map + ECI), `resolve_tam_parts`. No attention, no decoder-layer poking — only `hidden_states[-1]` + `lm_head.weight`. |
| `tam_losses.py` | `gaussian_blur_maps`, `concentration_gate`, `cosine/js/l1` divergences, `tam_alignment_loss`. |
| `tam_trainer.py` | `TAMTrainer(OPDTrainer)` — one student grad forward (OPD logits **and** TAM hidden states), one no-grad teacher forward, `L_opd + λ·L_tam`. |
| `sanity_check.py` | standalone engine check: grad reaches the **vision tower**, teacher detached, runs under **SDPA** (no attention), grid match, peak memory. |
| `../train_opd_tam.py` | entry point (`OPDTAMScriptArguments`, `--tam_*` knobs). |
| `../../scripts/train_opd_tam_qwen3_8b_to_2b.sh` | launcher (Qwen3-VL 8B→2B, ViT **trained** to match the full-FT OPD baseline). |

Nothing in `vigos/` or the vanilla OPD files is modified.

## TAM vs the saliency (evidence) channel

| | `baseline/evidence/` (Saliency-R1) | `baseline/tam/` (this) |
|---|---|---|
| map | two-hop `answer→reason→visual` attention routing + OV circuit | logit-lens `ReLU(F^v·W[y])` |
| needs attention weights | **yes** → eager + forward hooks → `O(S²)` memory wall | **no** → SDPA/Flash, `output_hidden_states` only |
| per-step cost | one eager `output_attentions` forward; `per_device≈1` | a few matmuls on top of OPD; cheap |
| gradient target | attention / value projections | visual representation `F^v` (→ ViT) |

They are complementary perspectives; TAM is the cheaper default to get a signal.

## Run

Validate the engine first (1 GPU):

```bash
uv run python -m baseline.tam.sanity_check \
    --student_model Qwen/Qwen3-VL-2B-Instruct \
    --teacher_model Qwen/Qwen3-VL-8B-Instruct --attn sdpa
```

Then train (8×A100/H800; full ViT, OPD+TAM):

```bash
M=/path/to/models \
DATASET_NAME=/path/to/Vision-SR1-47K \
bash scripts/train_opd_tam_qwen3_8b_to_2b.sh
# smoke: prepend MAX_STEPS=5 SAVE_STEPS=5 REPORT_TO=none WANDB_MODE=offline
# ablation: LAMBDA_TAM=0 recovers vanilla OPD exactly (clean OPD-vs-OPD+TAM comparison).
```

Watch `loss_opd` (behavior), `loss_tam` / `tam_div` (visual evidence — should fall
without hurting `answer_accuracy`), and `tam_gate_mean` (fraction of tokens the
gate keeps). Ablation knobs (doc §8): `TAM_DIVERGENCE`, `TAM_USE_ECI`,
`TAM_ALIGN_SPAN`, `TAM_MAX_TOKENS`, `LAMBDA_TAM ∈ {0,0.5,1,5,10}`,
`FREEZE_VISION_TOWER`.
