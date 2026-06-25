"""TAM visual-evidence alignment loss pieces (pure tensor ops, model-free).

Implements §2 of the migration doc: blur -> normalize -> divergence on the
per-token TAM maps from :mod:`baseline.tam.tam_engine`, gated to the
*visual-dependent* tokens.

    L_tam = mean_{i in P} g_i * d( p^stu_i , sg[ p^tea_i ] )

* **Gaussian blur** ``g(.)`` (the differentiable stand-in for TAM's rank-Gaussian
  filter — migration doc §1) is applied to **both** maps so the student is not
  penalized for noise the teacher map was denoised out of.
* **normalization**: sum-to-1 (a spatial distribution) for ``js`` / ``l1`` / ``mse``;
  L2 for ``cosine``. Never min-max (migration doc §2/§3 — non-smooth and breaks
  cross-model comparability).
* **divergence** ``d``: ``cosine`` (``1 - cos``, the MVP default — simplest and
  most robust on the non-negative TAM maps), ``js`` (Jensen-Shannon, the doc's
  theoretical default — symmetric, bounded), ``l1`` (total variation), or ``mse``
  (normalized heatmap-regression — squared L2 between the two spatial
  distributions; the most direct "match the teacher's map" objective).
* **concentration gate** ``g_i`` on the **teacher** map (migration doc §2, gate 1):
  down-weights tokens whose teacher map is spatially diffuse (high normalized
  entropy) — function words point nowhere, so the loss only pulls on tokens where
  the teacher actually grounds somewhere. Carries no gradient (teacher-only).

The teacher map is always detached (``sg``).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

_KERNEL_CACHE: dict[tuple, torch.Tensor] = {}


def gaussian_kernel2d(
    kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Cached separable 2-D Gaussian kernel ``[1, 1, k, k]`` (sum-normalized)."""
    key = (int(kernel_size), round(float(sigma), 4), device, dtype)
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return cached
    ax = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    g1 = torch.exp(-(ax**2) / (2.0 * max(sigma, 1e-6) ** 2))
    g1 = g1 / g1.sum()
    kernel = (g1[:, None] * g1[None, :]).reshape(1, 1, kernel_size, kernel_size)
    _KERNEL_CACHE[key] = kernel
    return kernel


def gaussian_blur_maps(
    maps: torch.Tensor,
    grid_thw: tuple[int, int, int],
    *,
    kernel_size: int = 3,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Blur each ``[n, n_v]`` map on its ``(t, H, W)`` patch grid. ``-> [n, n_v]``.

    Differentiable (a fixed-weight ``conv2d``); preserves the student gradient.
    Frames (``t``) are folded into the batch and blurred independently. Edge
    padding is ``reflect`` when the grid is large enough, else ``replicate`` — a
    Gaussian denoiser, so the mode is a minor edge detail.
    """
    t, h, w = (int(x) for x in grid_thw)
    n, n_v = maps.shape
    if t * h * w != n_v:
        raise ValueError(
            f"grid {t}x{h}x{w}={t * h * w} != #visual tokens {n_v}; cannot reshape "
            "TAM maps for the Gaussian blur (check spatial_merge_size / image_grid_thw)."
        )
    if n == 0 or kernel_size <= 1 or min(h, w) < 2:
        return maps  # nothing to blur (no tokens / degenerate grid).

    kernel = gaussian_kernel2d(kernel_size, sigma, maps.device, maps.dtype)
    pad = kernel_size // 2
    grid = maps.reshape(n * t, 1, h, w)
    mode = "reflect" if min(h, w) > pad else "replicate"
    grid = F.pad(grid, (pad, pad, pad, pad), mode=mode)
    blurred = F.conv2d(grid, kernel)
    return blurred.reshape(n, n_v)


def normalized_entropy(maps: torch.Tensor, temp: float = 1.0, eps: float = 1e-9) -> torch.Tensor:
    """Normalized spatial entropy of ``softmax(|maps|/temp)``. ``[n, P] -> [n]`` in [0,1].

    Low = the map is concentrated on a few patches; high = diffuse.
    """
    weights = torch.softmax(maps.abs() / max(temp, 1e-6), dim=-1)
    entropy = -(weights * (weights + eps).log()).sum(dim=-1)
    num_patches = maps.shape[-1]
    return entropy / math.log(max(num_patches, 2))


def concentration_gate(
    teacher_maps: torch.Tensor,
    *,
    temp: float = 1.0,
    h0: float = 0.9,
    tau: float = 0.1,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Soft concentration gate ``g = sigmoid((h0 - H_norm)/tau)``. ``[n, P] -> [n]``.

    ~1 for concentrated (object-like) teacher maps, ~0 for diffuse (function-word)
    ones. Detached — the teacher map is a constant.
    """
    h = normalized_entropy(teacher_maps, temp=temp, eps=eps)
    return torch.sigmoid((h0 - h) / max(tau, 1e-6)).detach()


def cosine_divergence(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """``1 - cos(a, b)`` per row of L2-normalized maps. ``[n, P] -> [n]``."""
    a = a / (a.norm(dim=-1, keepdim=True) + eps)
    b = b / (b.norm(dim=-1, keepdim=True) + eps)
    return 1.0 - (a * b).sum(dim=-1)


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Jensen-Shannon divergence of two sum-normalized maps. ``[n, P] -> [n]`` in [0, ln2]."""
    m = 0.5 * (p + q)
    kl_pm = (p * ((p + eps).log() - (m + eps).log())).sum(dim=-1)
    kl_qm = (q * ((q + eps).log() - (m + eps).log())).sum(dim=-1)
    return 0.5 * kl_pm + 0.5 * kl_qm


def l1_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Total variation ``0.5 * |p - q|_1`` of two sum-normalized maps. ``[n, P] -> [n]`` in [0,1]."""
    return 0.5 * (p - q).abs().sum(dim=-1)


def mse_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Squared L2 ``sum_j (p - q)^2`` of two sum-normalized maps. ``[n, P] -> [n]`` in [0,2].

    The **normalized MSE** (migration doc / user note): both maps are first turned
    into spatial distributions (sum-to-1) so the loss pulls the student toward the
    teacher's *spatial shape* (where the evidence is) and **not** its raw activation
    *scale* — teacher/student differ in hidden dim and ``lm_head`` norm, so a raw-map
    MSE would chase that scale gap instead of "look here". Summed (not meaned) over
    patches, so the value lands in the same ``[0, 2]`` band as ``cosine``/``js``/``l1``
    rather than the ~``1/n_v`` of a per-patch mean — i.e. ``n_v * mean_j (p-q)^2``.
    The teacher is detached by the caller (``sg``)."""
    return (p - q).pow(2).sum(dim=-1)


def _sum_normalize(maps: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return maps / (maps.sum(dim=-1, keepdim=True) + eps)


def tam_alignment_loss(
    student_maps: torch.Tensor,
    teacher_maps: torch.Tensor,
    *,
    grid_thw: tuple[int, int, int],
    divergence: str = "cosine",
    blur: bool = True,
    blur_kernel: int = 3,
    blur_sigma: float = 1.0,
    gate_temp: float = 1.0,
    gate_h0: float = 0.9,
    gate_tau: float = 0.1,
    mass_threshold: float = 0.0,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
    """Concentration-gated mean of ``d(p^stu, sg[p^tea])`` over the given tokens.

    Args:
        student_maps: ``[n, n_v]`` student TAM maps (grad).
        teacher_maps: ``[n, n_v]`` teacher TAM maps (detached inside).
        grid_thw: ``(t, H, W)`` patch grid for the blur (``t*H*W == n_v``).
        divergence: ``cosine`` | ``js`` | ``l1``.
        blur / blur_kernel / blur_sigma: fixed Gaussian blur applied to both maps.
        gate_*: teacher-map concentration gate parameters.
        mass_threshold: drop tokens whose (blurred) teacher map sums below this —
            the teacher grounds nowhere, so there is nothing to align to.

    Returns ``(loss, stats)``; ``loss`` is normalized by the gate sum (floored by
    ``eps`` so a batch of weakly-gated tokens is not washed out).
    """
    n = int(student_maps.shape[0])
    if n == 0:
        zero = student_maps.sum() * 0.0
        return zero, {"tam_div": zero.detach(), "tam_gate_mean": zero.detach(), "tam_n": 0}

    s = student_maps.float()
    t = teacher_maps.detach().float()
    if blur:
        s = gaussian_blur_maps(s, grid_thw, kernel_size=blur_kernel, sigma=blur_sigma)
        t = gaussian_blur_maps(t, grid_thw, kernel_size=blur_kernel, sigma=blur_sigma)

    gate = concentration_gate(t, temp=gate_temp, h0=gate_h0, tau=gate_tau)  # [n], no grad
    if mass_threshold > 0:
        gate = gate * (t.sum(dim=-1) > mass_threshold).to(gate.dtype)

    if divergence == "cosine":
        divergence_per_token = cosine_divergence(s, t, eps)
    elif divergence == "js":
        divergence_per_token = js_divergence(_sum_normalize(s), _sum_normalize(t))
    elif divergence == "l1":
        divergence_per_token = l1_divergence(_sum_normalize(s), _sum_normalize(t))
    elif divergence == "mse":
        divergence_per_token = mse_divergence(_sum_normalize(s), _sum_normalize(t))
    else:
        raise ValueError(
            f"Unknown divergence {divergence!r}; use 'cosine', 'js', 'l1', or 'mse'."
        )

    gate_sum = gate.sum().clamp_min(eps)
    loss = (gate * divergence_per_token).sum() / gate_sum
    stats = {
        "tam_div": divergence_per_token.detach().mean(),
        "tam_gate_mean": gate.detach().mean(),
        "tam_n": n,
    }
    return loss, stats
