"""Evidence-alignment loss pieces (pure tensor ops, model-free).

Implements §2-§3 of the migration doc:

* **signed Pearson** divergence ``D = 1 - corr(S_S, sg[S_T])`` — subtracts the
  per-map mean so a global offset between the student and teacher maps does not
  count as disagreement (more stable than raw cosine).
* **concentration gate** ``g_t`` on ``|S_T|`` — down-weights answer tokens whose
  *teacher* saliency is spatially diffuse (high normalized entropy), so the loss
  only pulls on tokens where the teacher actually points somewhere. Optional
  ``kl`` / ``mass`` gates (the doc's triple gate) drop tokens with no
  teacher-student disagreement or near-zero saliency mass.
* **token selection** — rank candidate (answer-span) tokens by teacher/student KL
  and keep the top fraction, so the (expensive) saliency engine and the loss run
  only on the tokens that matter.

The teacher map is always detached (``sg``); the gate carries no gradient.
"""

from __future__ import annotations

import math

import torch


def signed_pearson_corr(
    student_maps: torch.Tensor, teacher_maps: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Per-row signed Pearson correlation of flattened maps. ``[n, P] -> [n]``."""
    a = student_maps - student_maps.mean(dim=-1, keepdim=True)
    b = teacher_maps - teacher_maps.mean(dim=-1, keepdim=True)
    a = a / (a.norm(dim=-1, keepdim=True) + eps)
    b = b / (b.norm(dim=-1, keepdim=True) + eps)
    return (a * b).sum(dim=-1)


def signed_corr_loss(
    student_maps: torch.Tensor, teacher_maps: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """``1 - signed Pearson`` per row. ``[n, P] -> [n]``."""
    return 1.0 - signed_pearson_corr(student_maps, teacher_maps, eps)


def normalized_abs_entropy(
    teacher_maps: torch.Tensor, temp: float = 1.0, eps: float = 1e-9
) -> torch.Tensor:
    """Normalized spatial entropy of ``softmax(|S_T|/temp)``. ``[n, P] -> [n]`` in [0,1].

    Low = the teacher's (absolute) saliency is concentrated on a few patches;
    high = diffuse. Uses ``|S_T|`` so a spatially focused **negative** evidence
    token (image argues against the token) is not mistaken for diffuse.
    """
    weights = torch.softmax(teacher_maps.abs() / temp, dim=-1)
    entropy = -(weights * (weights + eps).log()).sum(dim=-1)
    num_patches = teacher_maps.shape[-1]
    return entropy / math.log(max(num_patches, 2))


def concentration_gate_abs(
    teacher_maps: torch.Tensor,
    *,
    temp: float = 1.0,
    h0: float = 0.9,
    tau: float = 0.1,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Soft concentration gate ``g_t = sigmoid((h0 - H_norm(|S_T|)) / tau)``.

    ``[n, P] -> [n]`` in (0,1); ~1 for concentrated teacher maps, ~0 for diffuse.
    Detached (teacher_maps is already a constant), returned without grad.
    """
    h = normalized_abs_entropy(teacher_maps, temp=temp, eps=eps)
    return torch.sigmoid((h0 - h) / max(tau, 1e-6)).detach()


def per_token_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    direction: str = "forward",
) -> torch.Tensor:
    """Per-token KL for ranking. ``[m, V], [m, V] -> [m]`` (caller detaches).

    ``forward`` = KL(teacher || student) (where the teacher has mass the student
    lacks — "the teacher knows something here"); ``reverse`` = KL(student ||
    teacher).
    """
    s_lp = torch.log_softmax(student_logits.float() / temperature, dim=-1)
    t_lp = torch.log_softmax(teacher_logits.float() / temperature, dim=-1)
    if direction == "reverse":
        p_lp, q_lp = s_lp, t_lp
    else:
        p_lp, q_lp = t_lp, s_lp
    return (p_lp.exp() * (p_lp - q_lp)).sum(dim=-1)


def top_indices_by_score(
    scores: torch.Tensor,
    ratio: float,
    *,
    min_keep: int = 1,
    max_keep: int | None = None,
) -> torch.Tensor:
    """Indices of the top ``ratio`` fraction of ``scores`` (1-D). ``-> [k]`` Long."""
    m = int(scores.shape[0])
    if m == 0:
        return scores.new_zeros(0, dtype=torch.long)
    k = max(min_keep, int(math.ceil(ratio * m)))
    if max_keep is not None:
        k = min(k, max_keep)
    k = min(k, m)
    return torch.topk(scores, k=k, dim=0).indices


def evidence_alignment_loss(
    student_maps: torch.Tensor,
    teacher_maps: torch.Tensor,
    *,
    gate_temp: float = 1.0,
    gate_h0: float = 0.9,
    gate_tau: float = 0.1,
    kl_scores: torch.Tensor | None = None,
    kl_threshold: float = 0.0,
    mass_threshold: float = 0.0,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
    """Gated mean of ``1 - corr(S_S, sg[S_T])`` over selected answer tokens.

    Args:
        student_maps: ``[n, H, W]`` or ``[n, P]`` — student saliency (grad).
        teacher_maps: ``[n, H, W]`` or ``[n, P]`` — teacher saliency (detached).
        gate_*: concentration-gate parameters.
        kl_scores: optional ``[n]`` per-token teacher/student KL for the ``kl``
            gate (dropped when ``<= kl_threshold``).
        kl_threshold / mass_threshold: enable the doc's triple gate when > 0.

    Returns ``(loss, stats)``. ``loss`` is the gated mean (normalized by the gate
    sum, floored by ``eps`` so small gates are not washed out).
    """
    n = student_maps.shape[0]
    if n == 0:
        zero = student_maps.sum() * 0.0
        return zero, {"ev_corr": zero.detach(), "ev_gate_mean": zero.detach(), "ev_n": 0}

    # fp32 for the correlation: the maps are bf16 off lm_head and the centered
    # dot product is numerically touchy. .float() keeps the student grad path.
    s_s = student_maps.reshape(n, -1).float()
    s_t = teacher_maps.reshape(n, -1).detach().float()

    gate = concentration_gate_abs(
        s_t, temp=gate_temp, h0=gate_h0, tau=gate_tau
    )  # [n], no grad
    if kl_scores is not None and kl_threshold > 0:
        gate = gate * (kl_scores.detach() > kl_threshold).to(gate.dtype)
    if mass_threshold > 0:
        gate = gate * (s_t.abs().mean(dim=-1) > mass_threshold).to(gate.dtype)

    corr = signed_pearson_corr(s_s, s_t, eps)  # [n], grad through s_s
    divergence = 1.0 - corr
    gate_sum = gate.sum().clamp_min(eps)
    loss = (gate * divergence).sum() / gate_sum

    stats = {
        "ev_corr": corr.detach().mean(),
        "ev_gate_mean": gate.detach().mean(),
        "ev_n": int(n),
    }
    return loss, stats
