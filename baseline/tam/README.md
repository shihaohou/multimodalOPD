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
| **RGF** (Eq.6,7) | rank-Gaussian denoise | ✅ **opt-in** (`TAM_DENOISE=rgf`) | `rank_gaussian_filter_maps`: vectorized + value-differentiable via `torch.sort` (sort permutation held constant → "hard" rank grad). A fixed **Gaussian blur** is the smooth default; RGF reproduces the paper exactly (`TAM-MSE-RGF` ablation) |

## Loss (`tam_losses.py`)

```
ã_i = ECI(a_i)                         # cleaned base map (ReLU)
ā_i = filter(ã_i)                      # filter = gaussian_blur (default) | RGF (paper) | none
p_i = normalize(ā_i)                   # L2 (cosine) or Laplace sum-to-1 (js/l1/mse)
L_TAM = mean_i  g_i · d( p^stu_i , sg[p^tea_i] )    # both sides filtered equally; teacher detached
```

* `d` (`TAM_DIVERGENCE`) = `cosine` (`1-cos`, default — robust on non-negative
  maps), `js` (Jensen-Shannon, the doc's theoretical default), `l1` (total
  variation), or `mse` (normalized heatmap-regression `Σ_j (p^S−p^T)²`).
* `filter` (`TAM_DENOISE`) = `gaussian` (fixed blur, default), `rgf` (the paper's
  Rank-Gaussian Filter, value-differentiable), or `none`.
* `g_i` = **teacher-map concentration gate** (`sigmoid((h0 − H_norm)/τ)`):
  down-weights tokens whose teacher map is diffuse (function words point nowhere).
  Set `TAM_GATE=false` to disable it — then every aligned token gets weight 1 and
  `L_TAM` is the plain `1/|P|` mean (the "align all tokens" step). The full loss is
  `L = λ_opd·L_OPD + λ_tam·L_TAM`; **token-KL is never removed**.

### Paper-faithful ablation: **OPD + TAM-MSE-RGF**

`TAM_DIVERGENCE=mse TAM_DENOISE=rgf TAM_GATE=false` reproduces the paper's visual
objective exactly — the denoised **visual** activation map `Ā_i^a = RGF(ECI(a_i))`,
normalized to a spatial distribution, matched by per-patch MSE against the detached
teacher's. (We use only `Ā_i^a`, **not** the multimodal viz map `M_i = N(Ā_i^a ⊥ r_i)`.)
"Version A": hard RGF on both sides; the student's `torch.sort` is differentiable
w.r.t. the map values but not the rank, so the gradient is "hard" — the known risk
of this ablation. If it destabilizes, fall back to `TAM_DENOISE=gaussian` (the
smooth stand-in) with the same `mse` divergence.

## Files

| file | role |
|---|---|
| `tam_engine.py` | `compute_tam_token_maps` (differentiable base map + ECI), `resolve_tam_parts`. No attention, no decoder-layer poking — only `hidden_states[-1]` + `lm_head.weight`. |
| `tam_losses.py` | `gaussian_blur_maps`, `rank_gaussian_filter_maps` (RGF), `apply_spatial_filter`, `concentration_gate`, `cosine/js/l1/mse` divergences, `tam_alignment_loss`. |
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

**OPD + TAM-MSE-RGF ablation** (paper-faithful denoised-map MSE, no gate, all tokens):

```bash
M=/path/to/models DATASET_NAME=/path/to/Vision-SR1-47K \
TAM_DIVERGENCE=mse TAM_DENOISE=rgf TAM_GATE=false \
bash scripts/train_opd_tam_qwen3_8b_to_2b.sh
# auto-named runs/opd_tam_qwen3_8b_to_2b_ltam1.0_mse_rgf_nogate_fullft_<date>
```

Watch `loss_opd` (behavior), `loss_tam` / `tam_div` (visual evidence — should fall
without hurting `answer_accuracy`), and `tam_gate_mean` (fraction of tokens the
gate keeps; `=1.0` when `TAM_GATE=false`). Ablation knobs (doc §8): `TAM_DIVERGENCE`,
`TAM_DENOISE`, `TAM_GATE`, `TAM_USE_ECI`, `TAM_ALIGN_SPAN`, `TAM_MAX_TOKENS`,
`LAMBDA_TAM ∈ {0,0.5,1,5,10}`, `FREEZE_VISION_TOWER`.
