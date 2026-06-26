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

## Readout direction: emitted token vs OPD correction (`TAM_DIRECTION`)

The base map can be read along **any** vocab-space direction. Two choices:

```
token       a_i = ReLU( F^v · W[y_i]^T )              # emitted-token evidence (original TAM)
correction  a_i = ReLU( F^v · (W^T u_i)^T ),  u_i = sg(top-k(p_T - p_S))   # OPD residual evidence
```

* **`token`** (default): "which patches support the token the student *emitted*." On
  the (majority) tokens where teacher≈student this map is easy to match — both look
  at the obvious region — so the loss bottoms out fast and **decouples from what OPD
  is still correcting** (the observed `loss_tam` plateau).
* **`correction`**: "which patches support the teacher's *intended correction*."
  `u_i = p_T - p_S` is the **same residual the OPD reverse-KL is built on**; reading
  the map along it makes the visual channel and the behavior channel share one
  signal. On agreement tokens `u_i ≈ 0` → empty map → the term **auto-zeros**, so the
  loss concentrates on exactly the disagreement set OPD works on. This is the
  Saliency-R1-style *correction-direction* alignment.

Mechanics (`sparse_correction_topk` + `project_correction` in `tam_engine.py`):

1. **sparse top-k by `|p_T − p_S|`** (`TAM_CORR_TOP_K`, default 100) — keeps the few
   high-disagreement tokens, drops the ~150k-wide noise tail (and makes the
   projection a cheap gather). Picks *disagreement*, not the union of the argmaxes.
2. **L1-normalize** the kept correction (`TAM_CORR_NORMALIZE`, default on): `û = u/Σ|u|`
   so the map *shape* is decoupled from the disagreement *magnitude*.
3. **`corr_mass = Σ|p_T − p_S|`** is re-injected as a per-token **loss weight**
   (`TAM_CORR_GATE`, default on → the **A′** variant): `L = Σ_i m_i·d_i / Σ_i m_i`, so
   near-agreement (empty-correction) tokens don't dilute the mean. Off → plain mean.
4. **No extra forward.** `p_S`/`p_T` are the OPD completion logits already computed,
   indexed to line up with the aligned tokens. `u` **and** both `lm_head`s are
   detached, so the evidence gradient still reaches `F^v` only — `W^T u` is a
   constant readout direction, exactly like `W[y_i]` was.
5. **ECI off** with correction (`TAM_USE_ECI=false`): ECI's context interference is
   defined around the one-hot token identity, and prompt context tokens have no
   rollout residual — a fully-consistent correction-ECI isn't well-defined, so keep
   it off for a clean isolation.

`hybrid` (fallback): `u_i = onehot(y_i) + α·sg(p_T − p_S)` (`TAM_CORR_ALPHA`) — keeps
the emitted-token evidence and adds the residual. Run **only if** pure `correction`
turns out too sparse; it re-mixes the two signals so the science is less clean.

### Recommended ablation ladder (isolate the direction variable)

Current run = `mse + rgf(hard) + nogate`. Change **one** variable at a time:

| run | knobs (on top of `TAM_DIVERGENCE=mse TAM_DENOISE=rgf TAM_GATE=false`) | question |
|---|---|---|
| **A0**  | `TAM_DIRECTION=token` (current) | baseline |
| **A0′** | `TAM_DIRECTION=token TAM_USE_ECI=false` | isolate the ECI-off effect |
| **A**   | `TAM_DIRECTION=correction TAM_USE_ECI=false TAM_CORR_GATE=false` | does the correction direction help at all? |
| **A′**  | `TAM_DIRECTION=correction TAM_USE_ECI=false` (corr-mass weight on) | + fix near-agreement dilution |

Headline comparison is **A′ vs A0′** (both ECI-off), not A′ vs A0 — else a gain could
just be the ECI-off change. Watch the new **`tam_corr_mass`** metric (mean
disagreement of the aligned tokens): if it collapses toward 0, the rollouts already
agree with the teacher and the correction signal is genuinely sparse → consider
`hybrid`. The forward adds no model passes, so A′ costs the same as A0.

```bash
# A' (the recommended next run):
M=/path/to/models DATASET_NAME=/path/to/Vision-SR1-47K \
TAM_DIRECTION=correction TAM_USE_ECI=false \
TAM_DIVERGENCE=mse TAM_DENOISE=rgf TAM_GATE=false \
bash scripts/train_opd_tam_qwen3_8b_to_2b.sh
# auto-named ..._mse_rgf_nogate_correctioncorrgate_ecioff_fullft_<date>
```

### Paper-faithful ablation: **OPD + TAM-MSE-RGF**

`TAM_DIVERGENCE=mse TAM_DENOISE=rgf TAM_GATE=false` reproduces the paper's visual
objective exactly — the denoised **visual** activation map `Ā_i^a = RGF(ECI(a_i))`,
normalized to a spatial distribution, matched by per-patch MSE against the detached
teacher's. (We use only `Ā_i^a`, **not** the multimodal viz map `M_i = N(Ā_i^a ⊥ r_i)`.)
"Version A": hard RGF on both sides; the student's `torch.sort` is differentiable
w.r.t. the map values but not the rank, so the gradient is "hard" — the known risk
of this ablation. If it destabilizes, fall back to `TAM_DENOISE=gaussian` (the
smooth stand-in) with the same `mse` divergence, or use a gradient surrogate below.

#### `TAM_RGF_GRAD` — student-side gradient surrogate (forward stays exact RGF)

The forward is **always** exact RGF on both sides (so the loss *value* is identical
across all modes — only the student's backward changes):

| mode | forward | backward | notes |
|---|---|---|---|
| `hard` (default) | RGF | true RGF grad | paper-faithful; same class as max-pool / median grad — try this first, the "hard" grad is rarely actually broken |
| `detach_sigma` | RGF (exact) | RGF grad **minus** the `σ=std/mean` term | **recommended hedge**: forward unchanged, drops only the one term that explodes on sparse windows; grad stays RGF-shaped & bounded |
| `gaussian` | RGF | Gaussian-blur grad (`G(ã)+sg(RGF(ã)−G(ã))`) | smooth spatial diffusion (GPT's suggestion); forward≠backward family |
| `identity` | RGF | straight-through to the raw map | bluntest; ignores RGF's rank redistribution entirely |

Faithfulness `hard > detach_sigma > gaussian > identity`. Suggested order: run `hard`
as the scientific baseline; if loss/grad-norm misbehaves, switch to `detach_sigma`,
then `gaussian`. (A *separate* experiment — teacher target RGF, student forward
Gaussian — is **not** a gradient trick; set `TAM_DENOISE=gaussian` and compare to a
teacher-only-RGF run rather than conflating it here.)

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
# correction direction (the A' path): add --direction correction --no_eci
#   --direction correction --no_eci --divergence mse --denoise rgf --no_gate
# asserts the correction direction is detached and the grad still reaches the ViT.
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
# stability hedge (forward still exact RGF, bounded gradient):
#   ... TAM_RGF_GRAD=detach_sigma ...   -> ..._mse_rgf_detach_sigmagrad_nogate_...
```

Watch `loss_opd` (behavior), `loss_tam` / `tam_div` (visual evidence — should fall
without hurting `answer_accuracy`), `tam_gate_mean` (fraction of tokens the gate
keeps; `=1.0` when `TAM_GATE=false`), and — for `TAM_DIRECTION=correction` — the new
**`tam_corr_mass`** (mean teacher↔student disagreement of the aligned tokens; a
collapse toward 0 means the rollouts already agree → the correction signal is
sparse). Ablation knobs (doc §8): `TAM_DIRECTION`, `TAM_CORR_TOP_K`,
`TAM_CORR_NORMALIZE`, `TAM_CORR_GATE`, `TAM_CORR_ALPHA`, `TAM_DIVERGENCE`,
`TAM_DENOISE`, `TAM_RGF_GRAD`, `TAM_GATE`, `TAM_USE_ECI`, `TAM_ALIGN_SPAN`,
`TAM_MAX_TOKENS`, `LAMBDA_TAM ∈ {0,0.5,1,5,10}`, `FREEZE_VISION_TOWER`.
