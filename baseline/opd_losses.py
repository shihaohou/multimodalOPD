"""Top-k / full-vocab distillation losses for OPD.

The wider OPD ecosystem (verl `forward_kl_topk`, thunlp/OPD, Uni-OPD) computes the
distillation KL over a small **top-k** token set instead of the full vocabulary:
the top-k tokens already carry ~97-99% of the probability mass, and — crucially —
a top-k objective is what lets the teacher be served remotely (only top-k logprobs
+ token ids need to be transferred). This module implements a masked top-k KL that
matches the full-vocab `vigos.losses.masked_kl_loss` normalization contract so it
drops into ``OPDTrainer`` and ``_distributed_masked_loss_with_stats`` unchanged.

``top_k=None`` recovers exact full-vocabulary KL.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vigos.losses import js_tokens


def masked_topk_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    token_mask: torch.Tensor,
    *,
    top_k: int | None = None,
    direction: str = "reverse",
    temperature: float = 1.0,
    token_clip: float | None = None,
) -> torch.Tensor:
    """Masked active-token mean of a (top-k) distillation divergence.

    direction:
      - ``reverse``  -> KL(student || teacher), support = student's top-k
      - ``forward``  -> KL(teacher || student), support = teacher's top-k
      - ``jsd``      -> Jensen-Shannon (top-k via :func:`vigos.losses.js_tokens`)
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "student_logits and teacher_logits must have the same shape, got "
            f"{tuple(student_logits.shape)} and {tuple(teacher_logits.shape)}."
        )
    if token_mask.shape != student_logits.shape[:-1]:
        raise ValueError(
            "token_mask must match logits without the vocabulary dimension, got "
            f"{tuple(token_mask.shape)} for logits {tuple(student_logits.shape)}."
        )

    mask = token_mask.to(device=student_logits.device, dtype=torch.bool)
    if not bool(mask.any()):
        # Keep both tensors in the autograd graph so DDP sees a gradient.
        return (student_logits.sum() + teacher_logits.sum()) * 0.0

    student_active = student_logits[mask]
    teacher_active = teacher_logits[mask]
    per_token = _topk_divergence(
        student_active,
        teacher_active,
        top_k=top_k,
        direction=direction,
        temperature=temperature,
    )
    if token_clip is not None and token_clip > 0:
        per_token = per_token.clamp(max=token_clip)
    return per_token.sum() / mask.sum().to(dtype=per_token.dtype).clamp_min(1.0)


def _topk_divergence(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    top_k: int | None,
    direction: str,
    temperature: float,
) -> torch.Tensor:
    if direction == "jsd":
        # js_tokens selects the top-k support from its first (student) argument.
        return js_tokens(
            student_logits,
            teacher_logits,
            temperature=temperature,
            top_k=top_k,
        )

    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
    if direction == "reverse":
        # KL(student || teacher); concentrate on the student's own modes.
        p_log_probs, q_log_probs, selector = (
            student_log_probs,
            teacher_log_probs,
            student_logits,
        )
    elif direction == "forward":
        # KL(teacher || student); concentrate on the teacher's modes.
        p_log_probs, q_log_probs, selector = (
            teacher_log_probs,
            student_log_probs,
            teacher_logits,
        )
    else:
        raise ValueError(
            f"Unknown direction {direction!r}; use 'reverse', 'forward', or 'jsd'."
        )

    if top_k is not None and 0 < top_k < selector.shape[-1]:
        indices = torch.topk(selector, k=top_k, dim=-1).indices
        p_log_probs = torch.gather(p_log_probs, dim=-1, index=indices)
        q_log_probs = torch.gather(q_log_probs, dim=-1, index=indices)

    return (p_log_probs.exp() * (p_log_probs - q_log_probs)).sum(dim=-1)
