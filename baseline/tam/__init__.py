"""Token Activation Map (TAM) visual-evidence alignment for OPD.

Adds a **visual-space** supervision channel alongside the vanilla OPD token-KL.
Token-KL teaches the student *what token to generate* (behavior); TAM alignment
teaches *where in the image the evidence for that token lives* (visual evidence),
by pulling the student's per-token Token Activation Map toward the frozen
teacher's on the visual-dependent tokens — on-policy, every step.

The TAM here is the **logit-lens** map ``ReLU(F^v @ W[y]^T)`` (last-layer hidden
state at the visual tokens, read along the rolled-out token's unembedding row),
made **differentiable** so the alignment gradient flows into the student's visual
representation. It needs only ``output_hidden_states`` — no attention weights — so
it runs under FlashAttention/SDPA and costs only a few matmuls on top of the OPD
forward. The shared vocabulary + ``lm_head`` give a free cross-size bridge between
the 8B teacher and 2B student (no PCA), provided they share a tokenizer + patch
grid (the Qwen3-VL family does).

Nothing in ``vigos/`` or the vanilla OPD files is modified — this is an additive
:class:`baseline.opd_trainer.OPDTrainer` subclass plus standalone library
modules. See ``baseline/tam/README.md`` and the migration doc.
"""

from baseline.tam.tam_engine import (
    TAMModelParts,
    compute_tam_token_maps,
    resolve_tam_parts,
)
from baseline.tam.tam_losses import (
    apply_spatial_filter,
    concentration_gate,
    cosine_divergence,
    gaussian_blur_maps,
    js_divergence,
    l1_divergence,
    mse_divergence,
    normalized_entropy,
    rank_gaussian_filter_maps,
    tam_alignment_loss,
)
from baseline.tam.tam_trainer import TAMTrainer

__all__ = [
    "TAMModelParts",
    "resolve_tam_parts",
    "compute_tam_token_maps",
    "gaussian_blur_maps",
    "rank_gaussian_filter_maps",
    "apply_spatial_filter",
    "normalized_entropy",
    "concentration_gate",
    "cosine_divergence",
    "js_divergence",
    "l1_divergence",
    "mse_divergence",
    "tam_alignment_loss",
    "TAMTrainer",
]
