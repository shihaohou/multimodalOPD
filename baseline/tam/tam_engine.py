"""Differentiable Token Activation Map (TAM) engine for OPD evidence alignment.

This is a **differentiable** re-implementation of the *logit-lens* Token
Activation Map from ``TAM-main`` (``tam.py``; paper *Token Activation Map ...*).
The original ``TAM`` runs under ``no_grad`` on a finished ``generate`` trace and
draws a heat-map per generated token for *visualization / evaluation*. We compute
the **same quantity** inside the training forward so the map carries gradients
into the student's visual representation — that is what lets a *visual-evidence
alignment* loss move *where the student looks* (see ``baseline/tam/README.md``).

Why TAM is a near-free visual channel for OPD (migration doc §0):

* The base map is the **logit-lens** of the last-layer hidden state at the visual
  token positions, read out along the *generated token's* unembedding row:

      a_i = ReLU( F^v @ W[y_i]^T )           # [n_v]  per answer token i

  ``F^v`` = ``hidden_states[-1]`` at the image-placeholder positions ``[n_v, d]``;
  ``W[y_i]`` = the ``lm_head`` row for the rolled-out token id ``y_i``.
* It needs **only** ``output_hidden_states`` — **no attention weights**. So it is
  compatible with FlashAttention / SDPA (no eager, no hooks) and adds only a few
  ``n_v``-sized matmuls on top of the forward the OPD loss already runs.
* The shared vocabulary + ``lm_head`` give a **free cross-size bridge**: the 8B
  teacher (``d=3584``) and the 2B student (``d=2048``) both collapse to the same
  ``n_v``-dim per-position scalar map, so they are directly comparable without any
  PCA / projection — provided the two share a **tokenizer** and a **patch grid**
  (asserted by the trainer; holds for the Qwen3-VL family).
* The gradient flows ``a_i -> F^v -> LLM layers -> projector/ViT``. With the
  ``lm_head`` **detached** (``detach_lm_head=True``, migration doc §3) the map is
  linear in ``F^v`` and the gradient lands cleanly on the visual representation
  rather than smearing into the unembedding.

ECI (Estimated Causal Inference, ``tam.py`` ~ lines 577-593): the raw base map of
an answer token is contaminated by *context* tokens that also activate the same
unembedding direction. ECI subtracts a least-squares-scaled, text-relevance-
weighted sum of the **context tokens' own** base maps:

    r^k_i = ReLU( <h_text_k, W[y_i]^T> )                 # text relevance of ctx k
    E_i   = sum_k ( r^k_i / sum_k' r^k'_i ) * a_k        # interference map
    s_i   = <a_i, E_i> / <E_i, E_i>                      # closed-form LS scale
    ã_i   = ReLU( a_i - s_i * E_i )

(``r^k_i = 0`` when the context token equals ``y_i`` — don't subtract an object
from itself.) On the **student** side ``E_i`` and ``s_i`` are treated as constants
(``stop-grad``, migration doc §1/§3): the cleaned map stays differentiable through
``a_i`` (and the ReLU gate) only, so ECI is an exact correction, not a new grad
path. The closed-form ``s`` replaces ``tam.py``'s SciPy ``minimize_scalar`` (same
optimum, ``argmin_x ||a - x E||^2 = <a,E>/<E,E>``) so the whole thing is a few
tensor ops.

RGF (the rank-Gaussian denoiser in ``tam.py``) is **not** used here: ranking is
non-differentiable. The trainer applies a fixed Gaussian blur to both maps
instead (``baseline/tam/tam_losses.py``); RGF is left to offline visualization.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class TAMModelParts:
    """Handles + dims the TAM engine needs from a VLM.

    Deliberately minimal: the logit-lens map only reads ``hidden_states[-1]`` (the
    post-final-norm hidden state ``lm_head`` consumes) and the ``lm_head`` weight,
    so — unlike the attention-routing saliency engine — it needs **no** decoder
    layer / value-projection / head-count handles.
    """

    lm_head: nn.Module
    image_token_id: int
    spatial_merge_size: int


def _first_present(obj: object, names: tuple[str, ...]):
    for name in names:
        if obj is not None and hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return None


_PARTS_CACHE: dict[int, TAMModelParts] = {}


def resolve_tam_parts(
    model: nn.Module,
    *,
    image_token_id: int | None = None,
    spatial_merge_size: int | None = None,
) -> TAMModelParts:
    """Resolve (and cache) the ``lm_head`` + visual-grid config for ``model``.

    ``image_token_id`` / ``spatial_merge_size`` override the config lookup (e.g.
    when the processor and config disagree). Robust to tied embeddings
    (``get_output_embeddings`` returns the unembedding either way).
    """
    key = id(model)
    cached = _PARTS_CACHE.get(key)
    if cached is not None:
        return cached

    lm_head = model.get_output_embeddings()
    if lm_head is None:
        lm_head = getattr(model, "lm_head", None)
    if lm_head is None or getattr(lm_head, "weight", None) is None:
        raise AttributeError(
            f"No output embedding / lm_head weight on {type(model).__name__}; "
            "the TAM logit-lens needs the unembedding matrix."
        )

    cfg = model.config
    if image_token_id is None:
        image_token_id = _first_present(cfg, ("image_token_id", "image_token_index"))
    if image_token_id is None:
        raise ValueError(
            "image_token_id not found on model.config; pass it explicitly to "
            "resolve_tam_parts()."
        )
    if spatial_merge_size is None:
        vision_cfg = getattr(cfg, "vision_config", None)
        spatial_merge_size = _first_present(vision_cfg or cfg, ("spatial_merge_size",))
    if spatial_merge_size is None:
        spatial_merge_size = 2  # Qwen2.5-VL / Qwen3-VL default.

    parts = TAMModelParts(
        lm_head=lm_head,
        image_token_id=int(image_token_id),
        spatial_merge_size=int(spatial_merge_size),
    )
    _PARTS_CACHE[key] = parts
    return parts


def compute_tam_token_maps(
    hidden_last: torch.Tensor,
    lm_head_weight: torch.Tensor,
    *,
    visual_positions: torch.Tensor,
    token_ids: torch.Tensor,
    token_positions: torch.Tensor | None = None,
    context_positions: torch.Tensor | None = None,
    context_ids: torch.Tensor | None = None,
    use_eci: bool = True,
    detach_lm_head: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-token TAM activation maps for one sample.

    Args:
        hidden_last: ``[S, hidden]`` the **last-layer (post-norm)** hidden states
            for the full ``prompt+completion`` sequence — i.e. exactly what
            ``lm_head`` consumes (``model(..., output_hidden_states=True)
            .hidden_states[-1][b]``). Carries grad for the student, detached for
            the teacher.
        lm_head_weight: ``[V, hidden]`` the unembedding matrix
            (``parts.lm_head.weight``). Detached internally when ``detach_lm_head``.
        visual_positions: ``[n_v]`` absolute positions of the image-placeholder
            tokens (``input_ids == image_token_id``).
        token_ids: ``[n_cand]`` vocab ids of the tokens to explain (the rolled-out
            completion tokens). Each picks the ``lm_head`` row ``W[y_i]`` the map is
            read along.
        token_positions: ``[n_cand]`` absolute sequence positions of those tokens —
            only needed for the ECI causal mask (a context token must precede the
            token it interferes with). Required when ``use_eci``.
        context_positions: ``[n_ctx]`` absolute positions of the **text** tokens
            usable as ECI context (non-visual, non-pad). ``None`` disables ECI.
        context_ids: ``[n_ctx]`` vocab ids at ``context_positions`` (each context
            token's *own* base map is read along its *own* unembedding row).
        use_eci: apply the ECI interference subtraction (migration doc §1).
        detach_lm_head: read the map along a **detached** ``lm_head`` row so the
            gradient lands on ``F^v`` only (migration doc §3). Strongly recommended.
        eps: denominator floor for the relevance normalization / LS scale.

    Returns:
        ``[n_cand, n_v]`` non-negative maps, one per token, with a live gradient
        through ``hidden_last`` at the visual positions when ``hidden_last``
        requires grad. ``n_v == visual_positions.numel()``.
    """
    device = hidden_last.device
    visual_positions = visual_positions.to(device)
    token_ids = token_ids.to(device=device, dtype=torch.long)

    weight = lm_head_weight.detach() if detach_lm_head else lm_head_weight

    # F^v in fp32: the maps are bf16 off the hidden states and the dot products are
    # numerically touchy; .float() keeps the student grad path into hidden_last.
    f_vis = hidden_last.index_select(0, visual_positions).float()      # [n_v, hidden]
    w_tok = weight.index_select(0, token_ids).float()                  # [n_cand, hidden]
    base = torch.relu(f_vis @ w_tok.t()).t()                           # [n_cand, n_v]  (grad)

    if (
        not use_eci
        or context_positions is None
        or context_ids is None
        or context_positions.numel() == 0
        or token_positions is None
    ):
        return base

    # --- ECI: subtract the context's interference (all stop-grad on the student) --
    context_positions = context_positions.to(device)
    context_ids = context_ids.to(device=device, dtype=torch.long)
    token_positions = token_positions.to(device)

    f_vis_d = f_vis.detach()
    w_ctx = weight.index_select(0, context_ids).float()               # [n_ctx, hidden]
    a_ctx = torch.relu(f_vis_d @ w_ctx.t()).t()                        # [n_ctx, n_v] (detached)

    h_text = hidden_last.index_select(0, context_positions).detach().float()  # [n_ctx, hidden]
    relevance = torch.relu(h_text @ w_tok.t())                        # [n_ctx, n_cand] (detached)

    # A context token only interferes if it (a) precedes the explained token and
    # (b) is a *different* token id (don't subtract an object from itself).
    causal = context_positions.unsqueeze(1) < token_positions.unsqueeze(0)   # [n_ctx, n_cand]
    non_repeat = context_ids.unsqueeze(1) != token_ids.unsqueeze(0)
    relevance = relevance * (causal & non_repeat).to(relevance.dtype)
    weights = relevance / (relevance.sum(dim=0, keepdim=True) + eps)  # [n_ctx, n_cand]

    interference = weights.t() @ a_ctx                                # [n_cand, n_v] (detached)
    # Closed-form least-squares scale argmin_x ||a - x*E||^2 = <a,E>/<E,E>.
    numerator = (base.detach() * interference).sum(dim=-1)            # [n_cand]
    denominator = (interference * interference).sum(dim=-1) + eps
    scale = (numerator / denominator).unsqueeze(-1)                   # [n_cand, 1]
    # Subtract the (detached) interference; grad flows through `base` and the ReLU
    # gate only — ECI is an exact correction, not a second gradient path.
    return torch.relu(base - scale * interference)
