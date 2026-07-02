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

import os

import torch
import torch.nn.functional as F

from vigos.losses import js_tokens

# Clamp the per-element log-prob difference (log p - log q) before weighting, like
# verl. Bounds the reverse-KL gradient so an outlier token (teacher ~0 prob where
# the student has mass) can't explode it; only affects extreme outliers.
_LOGPROB_DIFF_CLAMP = 20.0


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

    chunk_size = _kl_chunk_size()
    active_positions = mask.nonzero(as_tuple=False)
    if chunk_size <= 0 or int(active_positions.shape[0]) <= chunk_size:
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
            # Symmetric clamp: the diff-clamped surrogate (see _topk_divergence) can dip
            # slightly negative, so bound both tails — verl `loss_max_clamp` semantics.
            per_token = per_token.clamp(min=-token_clip, max=token_clip)
        return per_token.sum() / mask.sum().to(dtype=per_token.dtype).clamp_min(1.0)

    # Avoid materializing [all_active_tokens, vocab] log-prob tensors. With long
    # rollouts (e.g. mb=8, C=2048, V~150k), the unchunked top-k KL path needs
    # multiple extra 8GB tensors per rank even though it only keeps top-k terms.
    total = student_logits.new_zeros((), dtype=torch.float32)
    for start in range(0, int(active_positions.shape[0]), chunk_size):
        pos = active_positions[start : start + chunk_size]
        index = tuple(pos[:, dim] for dim in range(pos.shape[1]))
        per_token = _topk_divergence(
            student_logits[index],
            teacher_logits[index],
            top_k=top_k,
            direction=direction,
            temperature=temperature,
        )
        if token_clip is not None and token_clip > 0:
            per_token = per_token.clamp(min=-token_clip, max=token_clip)
        total = total + per_token.sum().float()
    return total / mask.sum().to(device=student_logits.device, dtype=total.dtype).clamp_min(1.0)


def masked_topk_kl_loss_from_teacher_topk(
    student_logits: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    teacher_topk_logprobs: torch.Tensor,
    token_mask: torch.Tensor,
    *,
    temperature: float = 1.0,
    token_clip: float | None = None,
) -> torch.Tensor:
    """Forward top-k KL ``KL(teacher||student)`` from a server teacher's top-k.

    Used by the vLLM-server teacher path, where only the teacher's top-k token ids
    and (full-vocab-normalized) log-probs are available per completion position —
    not the full teacher logits. The student still has full logits locally, so its
    log-probs are gathered at the teacher's top-k ids.

    Shapes: ``student_logits`` [B, C, V]; ``teacher_topk_ids`` /
    ``teacher_topk_logprobs`` [B, C, k] (pad unused slots with id ``-1``);
    ``token_mask`` [B, C]. ``temperature`` scales the **student** only — query the
    teacher server at temperature 1.0 so the provided log-probs match.
    """
    if teacher_topk_ids.shape != teacher_topk_logprobs.shape:
        raise ValueError(
            "teacher_topk_ids and teacher_topk_logprobs must share shape, got "
            f"{tuple(teacher_topk_ids.shape)} and {tuple(teacher_topk_logprobs.shape)}."
        )
    if token_mask.shape != student_logits.shape[:-1]:
        raise ValueError(
            "token_mask must match student_logits without the vocab dim, got "
            f"{tuple(token_mask.shape)} for logits {tuple(student_logits.shape)}."
        )

    mask = token_mask.to(device=student_logits.device, dtype=torch.bool)
    if not bool(mask.any()):
        return student_logits.sum() * 0.0

    student_log_probs = F.log_softmax(student_logits[mask] / temperature, dim=-1)
    ids = teacher_topk_ids[mask].to(device=student_logits.device, dtype=torch.long)
    teacher_lp = teacher_topk_logprobs[mask].to(
        device=student_logits.device, dtype=student_log_probs.dtype
    )
    valid = ids >= 0
    student_lp = torch.gather(student_log_probs, dim=-1, index=ids.clamp_min(0))
    diff = (teacher_lp - student_lp).clamp(
        min=-_LOGPROB_DIFF_CLAMP, max=_LOGPROB_DIFF_CLAMP
    )
    summand = torch.where(
        valid,
        teacher_lp.exp() * diff,
        torch.zeros_like(student_lp),
    )
    per_token = summand.sum(dim=-1)
    if token_clip is not None and token_clip > 0:
        # Symmetric clamp (see masked_topk_kl_loss) — verl `loss_max_clamp`.
        per_token = per_token.clamp(min=-token_clip, max=token_clip)
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

    diff = (p_log_probs - q_log_probs).clamp(
        min=-_LOGPROB_DIFF_CLAMP, max=_LOGPROB_DIFF_CLAMP
    )
    return (p_log_probs.exp() * diff).sum(dim=-1)


def _kl_chunk_size() -> int:
    raw = os.environ.get("OPD_KL_CHUNK_SIZE", "256")
    try:
        return int(raw)
    except ValueError:
        return 256
