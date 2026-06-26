# OPD + Evidence Alignment (Saliency-R1 → OPD migration)

Adds a **differentiable evidence-alignment loss** on top of vanilla OPD: the
student's per-token *saliency map* (where its answer logit draws support from the
image) is pulled toward the frozen teacher's map for the same token. The student
then learns not just *what* the teacher answers (the OPD token-KL) but *where it
looks* to answer it.

```
loss = λ_opd · L_opd(reverse-KL token distillation)  +  λ_evidence · L_evidence
```

`L_evidence = mean over selected answer tokens t of  g_t · (1 − corr(S_S_t, sg[S_T_t]))`,
normalized by `Σ g_t`. `S` = signed saliency map, `g_t` = concentration gate on
`|S_T|`, `sg` = stop-grad (teacher map is a constant target).

This is **additive**: `vigos/` and the vanilla OPD files (`baseline/opd_*.py`,
`baseline/train_opd.py`) are untouched. Everything lives in `baseline/evidence/`.

## Files

| File | Role |
|------|------|
| `saliency_engine.py` | Differentiable port of Saliency_R1's logit-decomposition saliency. Two-hop `answer → reason → visual` attention routing, OV circuit `o_proj(α·V)` summed over layers, norm-rescale, unembed onto the generated answer token → per-patch scalar map. Config-driven (Qwen2.5-VL **and** Qwen3-VL). |
| `span_utils.py` | Parse an OPD completion (`<reason></reason>` + `\boxed{}`) into reason / answer token spans (id-aligned char→token offsets; malformed rows flagged). |
| `evidence_loss.py` | Signed-Pearson divergence, `|S_T|` concentration gate (+ optional kl/mass triple gate), high-KL token selection, gated aggregate. |
| `opd_evidence_trainer.py` | `OPDEvidenceTrainer(OPDTrainer)` — shares the OPD rollout, adds the evidence forward + loss. |
| `sanity_check.py` | **Step 1** standalone backward / peak-memory / grid check. Run this first. |
| `../train_opd_evidence.py` | Entry point (`--evidence_*` knobs). |
| `../../scripts/train_opd_evidence_qwen25_3b.sh` | Launcher. |

## Faithfulness to Saliency_R1

The engine reproduces `peterant330/Saliency_R1` (`trl/grpo_trainer.py` ~1815-1847)
exactly, with three adaptations needed to make it **differentiable** and per-token:

1. **Value states recomputed** as `v_proj(input_layernorm(hidden_states[l]))`
   instead of read from the no-grad generation KV-cache (values carry no RoPE, so
   this is byte-identical — but now has a gradient). Qwen3's QK-norm touches only
   q/k, so this holds for Qwen3-VL too.
2. **Per-answer-token maps** `[n_ans, H, W]` (Saliency_R1 sums them into one map
   per sample to score against a bbox; the evidence loss needs per-token maps).
3. **Direction-only unembed** — gather just the generated token's row of `lm_head`
   instead of materializing `[n_ans, P, vocab]` under grad.

`signed=True` keeps the sign (negative = the image argues *against* the token);
`signed=False` reproduces Saliency_R1's positive-only ReLU.

## Staging (do these in order)

**Step 1 — standalone sanity (run FIRST, on the box):**

```bash
uv run python -m baseline.evidence.sanity_check \
    --student_model Qwen/Qwen2.5-VL-3B-Instruct \
    --teacher_model Qwen/Qwen2.5-VL-7B-Instruct \
    --attn eager --max_new_tokens 64
```

Confirms: `S_S.requires_grad` / `not S_T.requires_grad`, `L_ev.backward()` puts a
**non-zero** gradient on the student's `v_proj`, peak CUDA memory, and — with a
teacher — that **teacher and student share the patch grid** (same `#visual
tokens` and `(H,W)`). The grid check is the go/no-go for a cross-size pair.

**Step 3 — wired training:**

```bash
DATASET_NAME=LMMs-Lab-Turtle/Vision-SR1-47K \
TEACHER_MODEL=Qwen/Qwen2.5-VL-7B-Instruct \
bash scripts/train_opd_evidence_qwen25_3b.sh
```

Watch WandB: `loss_opd` (the OPD KL, should match the vanilla run), `loss_ev`,
`ev_corr` (teacher-student saliency correlation — should rise), `ev_gate_mean`,
`ev_n_selected`. If `loss_ev` never appears, the eager forward returned no
attentions (see caveat) — the run keeps training OPD-only.

## Which model line? (the Qwen3-VL question)

The evidence loss compares teacher vs student saliency **per patch**, so they must
share a patch grid for the same image. The per-patch scalar is **independent of
the hidden dim** (it is contracted inside the engine), so a *cross-size* pair
works **iff the grids match** — purely an empirical check, not a theoretical bar.

- **Qwen2.5-VL 3B ← 7B**: shares the ViT → same grid, guaranteed safe.
- **Qwen3-VL 8B → 2B** (the current OPD line): the ViT differs across sizes and
  DeepStack/visual-token layout may shift the grid. The engine itself runs
  (config-driven, QK-norm-safe); whether the *pair* aligns is decided by the
  sanity-check grid assertion. If it fails, use the Qwen2.5-VL line.

## Attention capture — `recompute` (default) vs `eager`

The saliency engine needs the model's attention *in-graph*. `EVIDENCE_ATTN_MODE`
picks how it is obtained:

- **`recompute` (default, the Stage-2 fix).** The forward stays on the fast kernel
  (**SDPA/Flash**). `capture_qkv_attention` temporarily overrides that kernel's
  registry entry with a wrapper that delegates to the real kernel (so the OPD
  logits are byte-for-byte unchanged) and stashes the model's own post-RoPE /
  post-QK-norm `q/k/v` for the evidence layers. The engine then redoes the softmax
  for **only the rows the two-hop routing needs** (`answer_query` ≤8 rows,
  `reason_query` T rows) — never the full `[H, S, S]`. RoPE / QK-norm / GQA / the
  causal+padding mask are all the model's own (the config is left on its real
  kernel, so `create_causal_mask` still builds the correct mask). Cost drops from
  eager's `L·H·S²` (retained for backward) to `K·H·(n_ans+T)·S` transient, and the
  other ~30 layers pay no eager tax → `per_device` can usually go to 2-4.
  Numerically identical to eager — proven on CPU by
  `python -m baseline.evidence.test_recompute_equiv` (`max|Δ|≈0`, incl. a
  left-padded batch).

- **`eager` (legacy / numerical reference).** Forces eager attention and captures
  full `[H, S, S]` matrices via forward hooks. Equivalent result, but retains every
  layer's attention matrix for backward — over thousands of visual tokens this is
  the dominant cost, so it wants a small `per_device_train_batch_size` (1-2). Kept
  as the reference the recompute path is validated against, and as a fallback if a
  new model family's attention dispatch ever breaks the capture.

Shared knobs that bound the work either way:

- `EVIDENCE_MAX_SAMPLES` (default 1) — rows of the micro-batch scored.
- `EVIDENCE_LAYERS` / `EVIDENCE_NUM_LAYERS` — which / how many decoder layers to sum.
- `EVIDENCE_TOP_RATIO` / `EVIDENCE_MAX_TOKENS` — cap the answer tokens scored.
- In `eager` mode only, **gradient checkpointing** can swallow the attentions; if
  `loss_ev` never logs, set `GRADIENT_CHECKPOINTING=false` or lower the batch. The
  trainer logs a warning and skips evidence rather than crashing.

## Knobs (env vars → CLI)

| Env | Default | Meaning |
|-----|---------|---------|
| `LAMBDA_EVIDENCE` | 1.0 | weight of the evidence term |
| `EVIDENCE_ATTN_MODE` | recompute | `recompute` (SDPA + selected-row recompute, fast) or `eager` (legacy full `[H,S,S]`) |
| `EVIDENCE_MAX_SAMPLES` | 1 | batch rows scored by the evidence forward |
| `EVIDENCE_LAYERS` | all | comma list of decoder layers to sum |
| `EVIDENCE_TOP_RATIO` | 0.2 | top-KL fraction of answer tokens kept |
| `EVIDENCE_MIN/MAX_TOKENS` | 1 / 8 | floor/cap on selected tokens per sample |
| `EVIDENCE_SIGNED` | true | signed map (vs positive-only ReLU) |
| `EVIDENCE_KL_DIRECTION` | forward | token-selection / kl-gate KL direction |
| `EVIDENCE_GATE_H0` / `_TAU` / `_TEMP` | 0.9 / 0.1 / 1.0 | concentration gate |
| `EVIDENCE_KL_THRESHOLD` / `_MASS_THRESHOLD` | 0 / 0 | enable the triple gate when > 0 |

See the migration doc for the method derivation and the experiment matrix
(raw-attention vs logit-decomposition, positive-only vs signed, gated vs ungated,
generated-token vs correction direction).
