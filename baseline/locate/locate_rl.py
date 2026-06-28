"""Pure RL math for Locate-Once Grounding — no model / processor / GPU needed.

Everything here is exercised CPU-only by ``baseline.locate.sanity_check`` so the
reward / advantage / box-extraction logic is testable without a training run:

* :func:`parse_student_box` — pull the student's first ``<box>[x1,y1,x2,y2]</box>``
  out of a decoded completion and normalize it (reusing the saliency bbox parser, so
  brackets-or-not, x2<x1 swaps, out-of-range clamping and degenerate-drop all match
  the GT side exactly).
* :func:`iou_norm` — IoU of two normalized ``[x1,y1,x2,y2]`` boxes (the reward).
* :func:`group_normalize_advantage` — GRPO advantage ``(r - mean)/(std + eps)`` per
  group of rollouts (verl/DeepEyes-style; singleton groups get advantage 0).

The trainer (:mod:`baseline.locate.opd_locate_trainer`) owns the token-span finding
and the policy-gradient term; this module stays framework-light on purpose.
"""

from __future__ import annotations

import re
from typing import Optional

import torch

from baseline.probe.saliency_data import BoxNorm, parse_bbox_norm

# The student is prompted to emit exactly one ``<box>...</box>`` at the head of its
# <think>. We key on the FIRST occurrence so a stray later mention can't hijack the
# reward. DOTALL: tolerate a coordinate list that the model split across newlines.
BOX_OPEN = "<box>"
BOX_CLOSE = "</box>"
_BOX_RE = re.compile(re.escape(BOX_OPEN) + r"(.*?)" + re.escape(BOX_CLOSE), re.DOTALL)


def extract_box_text(completion_text: str) -> Optional[str]:
    """Inner text of the FIRST ``<box>...</box>`` in ``completion_text``, else None."""
    match = _BOX_RE.search(completion_text)
    return match.group(1).strip() if match else None


def parse_student_box(completion_text: str) -> Optional[BoxNorm]:
    """Parse the student's first ``<box>[x1,y1,x2,y2]</box>`` into a normalized box.

    Returns None if there is no box, or the coordinates are unparseable / degenerate.
    Uses :func:`baseline.probe.saliency_data.parse_bbox_norm`, the SAME parser the GT
    boxes go through, so a student box and the GT box are directly IoU-comparable.
    """
    inner = extract_box_text(completion_text)
    if inner is None:
        return None
    return parse_bbox_norm(inner)


def first_box_span(text: str):
    """Char offsets of the FIRST ``<box>...</box>`` in ``text``.

    Returns ``(inner_text, open_start, close_end, inner_start, inner_end)`` (char
    indices into ``text``) or None. ``open_start..close_end`` spans the whole
    ``<box>...</box>``; ``inner_start..inner_end`` spans the coordinate text between the
    tags. Text-based on purpose: ``<box>``/``</box>`` are NOT single tokens (unlike
    Qwen's ``<think>``), so token-subsequence matching is unreliable — find them in the
    decoded string, then map back to tokens with :func:`char_span_to_token_span`.
    """
    match = _BOX_RE.search(text)
    if match is None:
        return None
    return match.group(1), match.start(), match.end(), match.start(1), match.end(1)


def char_span_to_token_span(
    pieces: list[str], char_start: int, char_end: int
) -> tuple[int, int]:
    """Map ``[char_start, char_end)`` in ``''.join(pieces)`` to a ``[tok_start, tok_end)``
    token range. ``pieces[i]`` = the decoded text of token ``i``. Exact for byte-level
    BPE over the (ASCII) box region. Pure mirror of the trainer's per-token decode loop,
    so the mapping logic is CPU-testable without a tokenizer.
    """
    tok_start: int | None = None
    cumulative = 0
    for index, piece in enumerate(pieces):
        nxt = cumulative + len(piece)
        if tok_start is None and nxt > char_start:
            tok_start = index
        if nxt >= char_end:
            return (tok_start if tok_start is not None else index, index + 1)
        cumulative = nxt
    total = len(pieces)
    return (tok_start if tok_start is not None else total, total)


def iou_norm(box_a: BoxNorm, box_b: BoxNorm) -> float:
    """IoU of two normalized ``(x1, y1, x2, y2)`` boxes (top-left origin).

    0.0 if the boxes are disjoint or either is degenerate. Inputs are assumed already
    order-normalized (x1<=x2, y1<=y2), which :func:`parse_bbox_norm` guarantees.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def sampled_token_logprobs(
    logits: torch.Tensor, token_ids: torch.Tensor
) -> torch.Tensor:
    """``log pi(token) = logit[token] - logsumexp(logits)`` over the FULL given vocab.

    ``logits`` ``[..., V]`` (use the STUDENT's full logits — the action is the
    student's own sample, so the normalizer must be the student's whole vocab, never a
    teacher-truncated slice); ``token_ids`` ``[...]`` (long). Returns ``[...]``. Pure
    so the RL gradient direction is unit-testable on CPU (see sanity_check).
    """
    gathered = logits.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
    return gathered - logits.logsumexp(dim=-1)


def group_normalize_advantage(
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    normalize_std: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """GRPO group-normalized advantage. Returns a DETACHED float tensor (a weight).

    ``rewards`` / ``group_ids`` are 1-D, same length (one entry per rollout). For each
    group the advantage is ``r - mean`` (``normalize_std=False``, Dr.GRPO) or
    ``(r - mean) / (std + eps)`` (``normalize_std=True``, the GRPO default). Population
    std (``unbiased=False``) for stability on small groups; a constant group →
    advantage 0. A singleton group has no baseline → advantage 0 (it cannot tell the
    policy whether its one sample was good or bad). Detached on purpose: the advantage
    multiplies ``log pi`` in the PG loss but must not carry gradient itself.
    """
    rewards = rewards.detach().to(dtype=torch.float32)
    group_ids = group_ids.detach().to(device=rewards.device, dtype=torch.long)
    advantage = torch.zeros_like(rewards)
    for gid in torch.unique(group_ids):
        idx = (group_ids == gid).nonzero(as_tuple=True)[0]
        if idx.numel() <= 1:
            continue  # singleton group: no group baseline available
        group_rewards = rewards[idx]
        centered = group_rewards - group_rewards.mean()
        if normalize_std:
            centered = centered / (group_rewards.std(unbiased=False) + eps)
        advantage[idx] = centered
    return advantage
