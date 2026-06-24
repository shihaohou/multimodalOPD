"""Evidence-alignment extension for OPD (the Saliency-R1 -> OPD migration).

This package adds a **differentiable evidence-alignment loss** alongside the
vanilla OPD token-distillation loss. The student's per-token *saliency map*
(where, spatially, the model's answer logit draws support from the image) is
pulled toward the frozen teacher's saliency map for the same token, so the
student learns not just *what* the teacher answers (the OPD token-KL) but *where
it looks* to answer it.

The saliency engine (:mod:`baseline.evidence.saliency_engine`) is a faithful,
**differentiable** port of peterant330/Saliency_R1's logit-decomposition
saliency (two-hop ``answer -> reason -> visual`` attention routing, OV-circuit
projection, unembedding onto the generated answer token). Saliency_R1 used it as
a non-differentiable GRPO *reward*; here it runs inside the training forward so
gradients flow into the student's attention/value projections.

Nothing in ``vigos/`` or the vanilla OPD files is modified — this is an additive
``OPDTrainer`` subclass plus standalone library modules.

See ``baseline/evidence/README.md`` and the migration doc for the method, the
staging (Step 1 standalone sanity -> Step 3 wired training), and the
eager-attention memory caveat.
"""

from baseline.evidence.evidence_loss import (
    concentration_gate_abs,
    evidence_alignment_loss,
    normalized_abs_entropy,
    per_token_kl,
    signed_corr_loss,
    signed_pearson_corr,
    top_indices_by_score,
)
from baseline.evidence.saliency_engine import (
    SaliencyModelParts,
    compute_token_saliency_maps,
    resolve_model_parts,
)

__all__ = [
    "SaliencyModelParts",
    "resolve_model_parts",
    "compute_token_saliency_maps",
    "signed_corr_loss",
    "signed_pearson_corr",
    "normalized_abs_entropy",
    "concentration_gate_abs",
    "per_token_kl",
    "top_indices_by_score",
    "evidence_alignment_loss",
]
