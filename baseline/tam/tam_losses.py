"""TAM visual-evidence alignment loss pieces (pure tensor ops, model-free).

Implements §2 of the migration doc: blur -> normalize -> divergence on the
per-token TAM maps from :mod:`baseline.tam.tam_engine`, gated to the
*visual-dependent* tokens.

    L_tam = mean_{i in P} g_i * d( p^stu_i , sg[ p^tea_i ] )

* **spatial filter** ``g(.)`` is applied to **both** maps so the student is not
  penalized for noise the teacher map was denoised out of. Two kinds:
  ``gaussian`` (a fixed-weight ``conv2d`` blur — the smooth default) and ``rgf``
  (:func:`rank_gaussian_filter_maps`, the paper's Rank-Gaussian Filter, made
  value-differentiable for the ``OPD + TAM-MSE-RGF`` ablation). ``apply_spatial_filter``
  dispatches; ``none`` skips filtering.
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


def rank_gaussian_filter_maps(
    maps: torch.Tensor,
    grid_thw: tuple[int, int, int],
    *,
    kernel_size: int = 3,
    detach_sigma: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Differentiable, vectorized **Rank-Gaussian Filter (RGF)** — the TAM paper's
    denoiser (``TAM-main/tam.py::rank_guassian_filter``) on each ``[n, n_v]`` map
    over its ``(t, H, W)`` patch grid. ``-> [n, n_v]``.

    For every ``k*k`` window the paper: sorts the window values ascending, sets
    ``sigma = std/mean`` (the coefficient of variation; **population** std, ddof=0),
    builds a Gaussian over the **rank axis** ``ax = [0..k^2-1] - k^2//2`` (the
    median rank gets the peak weight), sum-normalizes it, and returns the
    rank-weighted sum of the sorted values. So RGF is an *adaptive rank filter*:
    a low-variation window (small CoV → narrow kernel) collapses to its **median**;
    a high-variation window (large CoV → wide kernel) toward its **mean** — a
    robust, edge-preserving denoiser, unlike the fixed Gaussian blur which smears a
    constant kernel regardless of local structure. Windows with non-positive mean
    output ``0`` (matches the paper; for the non-negative TAM maps that means an
    all-zero window stays zero).

    **Differentiability.** Implemented with ``F.unfold`` + ``torch.sort``. The
    gradient is exact w.r.t. the map **values** (``sort`` back-props through the
    gather; ``mean``/``std``/``exp`` are smooth), but the sort **permutation** is
    treated as constant, so the gradient is non-smooth at rank-swap boundaries —
    this is the "hard RGF" used by the ``OPD + TAM-MSE-RGF`` ablation. (The paper's
    NumPy version is for offline visualization and carries no gradient.)

    ``detach_sigma``: stop-grad the per-window bandwidth ``sigma = std/mean`` so the
    rank kernel is a **constant** in the backward pass. The forward value is
    **unchanged** (exact RGF), but the gradient drops the ``∂sigma`` term — the one
    that blows up on sparse windows (small ``mean`` -> large ``1/mean``) — leaving
    only the bounded, RGF-shaped rank-routed term ``sum_r kernel_r * sorted_v_r``.
    Used by ``rgf_with_surrogate_grad(grad="detach_sigma")``; the most faithful
    gradient surrogate (forward is still exact RGF, unlike the gaussian surrogate).

    Note: divisors are floored by ``eps`` (so a uniform window → its value, and an
    all-zero window → 0, with no ``0/0`` NaN that the raw NumPy reference would hit).
    """
    t, h, w = (int(x) for x in grid_thw)
    n, n_v = maps.shape
    if t * h * w != n_v:
        raise ValueError(
            f"grid {t}x{h}x{w}={t * h * w} != #visual tokens {n_v}; cannot reshape "
            "TAM maps for the rank-Gaussian filter (check spatial_merge_size / image_grid_thw)."
        )
    k = int(kernel_size)
    if n == 0 or k <= 1 or min(h, w) < 2:
        return maps  # nothing to filter (no tokens / degenerate grid).

    pad = k // 2
    grid = maps.reshape(n * t, 1, h, w)
    mode = "reflect" if min(h, w) > pad else "replicate"
    grid = F.pad(grid, (pad, pad, pad, pad), mode=mode)
    # Every k*k window as a column: [N, k^2, L], L = H*W (im2col; differentiable).
    windows = F.unfold(grid, kernel_size=k)                       # [N, k^2, L]
    ksq = k * k
    sorted_w, _ = torch.sort(windows, dim=1)                      # ascending along rank
    mean = sorted_w.mean(dim=1, keepdim=True)                     # [N, 1, L]
    std = sorted_w.var(dim=1, unbiased=False, keepdim=True).clamp_min(0).sqrt()
    sigma = std / mean.clamp_min(eps)                             # coefficient of variation
    if detach_sigma:
        sigma = sigma.detach()  # constant rank kernel in backward; forward unchanged.
    ax = (
        torch.arange(ksq, device=maps.device, dtype=sorted_w.dtype) - ksq // 2
    ).view(1, ksq, 1)                                            # rank offset from median
    kernel = torch.exp(-(ax * ax) / (2.0 * (sigma * sigma)).clamp_min(eps))  # [N, k^2, L]
    kernel = kernel / kernel.sum(dim=1, keepdim=True).clamp_min(eps)
    value = (sorted_w * kernel).sum(dim=1)                        # [N, L]
    # Paper: a window whose mean is <= 0 contributes 0 (kills all-zero windows).
    value = torch.where(mean.squeeze(1) > 0, value, torch.zeros_like(value))
    return value.reshape(n, n_v)


def apply_spatial_filter(
    maps: torch.Tensor,
    grid_thw: tuple[int, int, int],
    *,
    kind: str,
    kernel_size: int = 3,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Dispatch the spatial denoiser applied to a ``[n, n_v]`` map before the
    divergence: ``"none"`` (identity), ``"gaussian"`` (fixed blur — the smooth
    default), or ``"rgf"`` (the paper's Rank-Gaussian Filter; ``sigma`` is ignored,
    RGF derives its own per-window bandwidth)."""
    if kind == "none":
        return maps
    if kind == "gaussian":
        return gaussian_blur_maps(maps, grid_thw, kernel_size=kernel_size, sigma=sigma)
    if kind == "rgf":
        return rank_gaussian_filter_maps(maps, grid_thw, kernel_size=kernel_size)
    raise ValueError(f"Unknown spatial filter kind {kind!r}; use 'none', 'gaussian', or 'rgf'.")


def rgf_with_surrogate_grad(
    maps: torch.Tensor,
    grid_thw: tuple[int, int, int],
    *,
    grad: str = "hard",
    kernel_size: int = 3,
    blur_sigma: float = 1.0,
) -> torch.Tensor:
    """RGF with a **straight-through gradient surrogate** — forward is always exact
    ``RGF(maps)``; only the backward path changes (apply on the *student* map; the
    teacher is detached so its grad mode is moot). ``grad``:

    * ``"hard"``        — the true RGF gradient (value-differentiable; rank held
      constant). The paper-faithful default; same class as max-pool / median grad.
    * ``"detach_sigma"``— exact-RGF forward, but the per-window bandwidth ``sigma``
      is stop-grad'd so the backward drops the ``∂sigma`` term that explodes on
      sparse windows. Most faithful surrogate (forward unchanged, RGF-shaped grad).
    * ``"gaussian"``    — STE ``G(maps) + sg(RGF(maps) - G(maps))``: forward RGF,
      backward = Gaussian-blur gradient (smooth spatial diffusion, same family).
    * ``"identity"``    — STE ``maps + sg(RGF(maps) - maps)``: forward RGF, backward
      identity (gradient straight to the raw map, no spatial mixing — the bluntest).

    All four agree in the forward pass (== ``RGF(maps)``); they differ only in what
    gradient reaches ``maps``. ``"hard"``/``"detach_sigma"`` have NO forward/backward
    mismatch in value; ``"gaussian"``/``"identity"`` trade a mismatch for a smoother
    update. See the ``TAM_RGF_GRAD`` knob and ``baseline/tam/README.md``.
    """
    if grad == "hard":
        return rank_gaussian_filter_maps(maps, grid_thw, kernel_size=kernel_size)
    if grad == "detach_sigma":
        return rank_gaussian_filter_maps(
            maps, grid_thw, kernel_size=kernel_size, detach_sigma=True
        )
    rgf = rank_gaussian_filter_maps(maps, grid_thw, kernel_size=kernel_size)
    if grad == "identity":
        surrogate = maps
    elif grad == "gaussian":
        surrogate = gaussian_blur_maps(
            maps, grid_thw, kernel_size=kernel_size, sigma=blur_sigma
        )
    else:
        raise ValueError(
            f"Unknown rgf grad surrogate {grad!r}; use 'hard', 'detach_sigma', "
            "'gaussian', or 'identity'."
        )
    return surrogate + (rgf - surrogate).detach()  # forward == rgf; backward == surrogate


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
    """Squared L2 ``sum_j (p - q)^2`` of two (Laplace-smoothed) sum-to-1 maps. ``[n, P] -> [n]`` in [0,2].

    The **normalized MSE** (migration doc / user note): both maps are first turned
    into spatial distributions (sum-to-1) so the loss pulls the student toward the
    teacher's *spatial shape* (where the evidence is) and **not** its raw activation
    *scale* — teacher/student differ in hidden dim and ``lm_head`` norm, so a raw-map
    MSE would chase that scale gap instead of "look here". Summed (not meaned) over
    patches, so the value lands in the same ``[0, 2]`` band as ``cosine``/``js``/``l1``
    rather than the ~``1/n_v`` of a per-patch mean — i.e. ``n_v * mean_j (p-q)^2``.
    The teacher is detached by the caller (``sg``)."""
    return (p - q).pow(2).sum(dim=-1)


def _prob_normalize(maps: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Laplace-smoothed sum-to-1 over patches: ``(a + eps) / (sum_j a_j + n_v*eps)``.

    A *true* spatial distribution (``sum_j p_j == 1`` exactly) even for a near-empty
    map: an all-zero teacher map normalizes to **uniform** ``1/n_v`` instead of a
    near-zero vector, so a distribution MSE/JS against it pulls the student toward
    "look everywhere / neutral" rather than the wrong "look nowhere" target. Matches
    the migration-doc normalizer; ``eps`` is tiny so a responsive map is unchanged.
    """
    n_v = maps.shape[-1]
    return (maps + eps) / (maps.sum(dim=-1, keepdim=True) + n_v * eps)


def tam_alignment_loss(
    student_maps: torch.Tensor,
    teacher_maps: torch.Tensor,
    *,
    grid_thw: tuple[int, int, int],
    divergence: str = "cosine",
    blur: bool = True,
    denoise: str | None = None,
    rgf_grad: str = "hard",
    blur_kernel: int = 3,
    blur_sigma: float = 1.0,
    use_gate: bool = True,
    gate_temp: float = 1.0,
    gate_h0: float = 0.9,
    gate_tau: float = 0.1,
    mass_threshold: float = 0.0,
    token_weights: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
    """Concentration-gated mean of ``d(p^stu, sg[p^tea])`` over the given tokens.

    Args:
        student_maps: ``[n, n_v]`` student TAM maps (grad).
        teacher_maps: ``[n, n_v]`` teacher TAM maps (detached inside).
        grid_thw: ``(t, H, W)`` patch grid for the blur (``t*H*W == n_v``).
        divergence: ``cosine`` | ``js`` | ``l1`` | ``mse``. The distribution family
            (``js`` / ``l1`` / ``mse``) runs on Laplace-smoothed sum-to-1 maps;
            ``cosine`` runs on L2-normalized maps.
        denoise: spatial filter applied to both maps — ``"gaussian"`` (fixed blur),
            ``"rgf"`` (the paper's Rank-Gaussian Filter — the ``TAM-MSE-RGF``
            ablation), or ``"none"``. ``None`` (default) derives it from ``blur``
            (``True`` -> gaussian) for back-compat. ``blur_kernel`` is the filter
            size for both; ``blur_sigma`` only applies to gaussian.
        rgf_grad: when ``denoise == "rgf"``, the student-side gradient surrogate —
            ``"hard"`` (true RGF grad, default), ``"detach_sigma"``, ``"gaussian"``,
            or ``"identity"`` (see :func:`rgf_with_surrogate_grad`). Forward is always
            exact RGF on both sides; this only changes how the student back-props.
            Ignored for ``gaussian`` / ``none``.
        blur / blur_kernel / blur_sigma: legacy gaussian-blur toggle + params; kept
            for back-compat (superseded by ``denoise`` when that is given).
        use_gate: apply the teacher-map concentration gate (and the ``mass_threshold``
            hard drop). ``False`` -> every token gets weight 1, so the loss is the
            plain ``1/|P|`` mean over all aligned tokens (the "no gate, align all
            tokens" ablation). Default ``True`` preserves current behavior.
        gate_*: teacher-map concentration gate parameters (ignored if not ``use_gate``).
        mass_threshold: hard-drop tokens whose (blurred) teacher map barely responds.
            In ``(0, 1)`` it is RELATIVE to the sample's mean teacher mass (portable);
            ``>= 1`` is an absolute sum threshold. ``0`` disables it (the soft gate
            still down-weights diffuse tokens).
        token_weights: optional ``[n]`` external per-token importance, multiplied into
            the gate (so it composes with ``use_gate``). For the OPD **correction**
            direction this is ``corr_mass = Σ_k |p_T - p_S|`` — weight each token's
            alignment by how much the teacher wants to correct it, so the loss is
            ``Σ_i m_i·d_i / Σ_i m_i`` and near-agreement tokens (empty correction
            map) don't dilute the mean. Detached. ``None`` -> weight 1.

    Returns ``(loss, stats)``; ``loss`` is normalized by the gate sum (floored by
    ``eps`` so a batch of weakly-gated tokens is not washed out).
    """
    n = int(student_maps.shape[0])
    if n == 0:
        zero = student_maps.sum() * 0.0
        z = zero.detach()
        return zero, {
            "tam_div": z, "tam_js": z, "tam_gate_mean": z, "tam_mass_kept": z,
            "tam_corr_mass": z, "tam_n": 0,
        }

    s = student_maps.float()
    t = teacher_maps.detach().float()
    kind = denoise if denoise is not None else ("gaussian" if blur else "none")
    if kind == "rgf":
        # Student: RGF forward + the chosen backward surrogate (rgf_grad). Teacher:
        # plain RGF (detached anyway). Both sides see the SAME RGF-denoised view in
        # forward; only the student's gradient path is shaped by rgf_grad.
        s = rgf_with_surrogate_grad(
            s, grid_thw, grad=rgf_grad, kernel_size=blur_kernel, blur_sigma=blur_sigma
        )
        t = rank_gaussian_filter_maps(t, grid_thw, kernel_size=blur_kernel)
    elif kind != "none":
        s = apply_spatial_filter(s, grid_thw, kind=kind, kernel_size=blur_kernel, sigma=blur_sigma)
        t = apply_spatial_filter(t, grid_thw, kind=kind, kernel_size=blur_kernel, sigma=blur_sigma)

    if use_gate:
        gate = concentration_gate(t, temp=gate_temp, h0=gate_h0, tau=gate_tau)  # [n], no grad
    else:
        # No gate: every token contributes with weight 1, so the loss is the plain
        # 1/|P| mean over all aligned tokens (the "align all tokens" ablation).
        gate = torch.ones(n, device=s.device, dtype=s.dtype)
    # External per-token importance (OPD correction mass): weight each token by how
    # much the teacher wants to correct it, composing multiplicatively with the gate.
    # Detached — a constant reduction weight, no gradient.
    corr_mass_mean = s.new_zeros(())
    if token_weights is not None:
        tw = token_weights.detach().to(device=s.device, dtype=s.dtype)
        corr_mass_mean = tw.mean()
        gate = gate * tw
    # Mass filter: hard-drop tokens whose (blurred) teacher map barely responds
    # anywhere — function words / non-visual tokens have nothing to ground to, and
    # under Laplace smoothing their teacher map is ~uniform, so aligning to it just
    # pushes the student toward uniform (noise). The raw map-sum scale is model-
    # dependent (teacher d != student d), so a value in (0, 1) is read as RELATIVE
    # to this sample's mean teacher mass (a portable "small" cutoff); a value >= 1
    # is an absolute sum threshold (back-compat). `tam_mass_kept` logs the surviving
    # fraction so the drop is never silent.
    teacher_mass = t.sum(dim=-1)  # [n] raw filtered teacher response strength, no grad
    mass_kept = torch.ones_like(teacher_mass)
    if use_gate and mass_threshold > 0:
        cutoff = (
            mass_threshold * teacher_mass.mean()
            if mass_threshold < 1.0
            else mass_threshold
        )
        mass_kept = (teacher_mass > cutoff).to(gate.dtype)
        gate = gate * mass_kept

    if divergence == "cosine":
        divergence_per_token = cosine_divergence(s, t, eps)
    elif divergence == "js":
        divergence_per_token = js_divergence(_prob_normalize(s), _prob_normalize(t))
    elif divergence == "l1":
        divergence_per_token = l1_divergence(_prob_normalize(s), _prob_normalize(t))
    elif divergence == "mse":
        divergence_per_token = mse_divergence(_prob_normalize(s), _prob_normalize(t))
    else:
        raise ValueError(
            f"Unknown divergence {divergence!r}; use 'cosine', 'js', 'l1', or 'mse'."
        )

    gate_sum = gate.sum().clamp_min(eps)
    loss = (gate * divergence_per_token).sum() / gate_sum
    # Monitor-only (no grad): gate-weighted JS over the same blurred/gated tokens —
    # a divergence-agnostic convergence signal logged to W&B no matter which
    # `divergence` drives the loss. JS is symmetric + bounded ([0, ln2]) so it reads
    # cleanly as "are the student & teacher maps actually agreeing on WHERE", even
    # while the loss optimizes cosine / mse / l1. Never affects training.
    with torch.no_grad():
        js_per_token = js_divergence(_prob_normalize(s), _prob_normalize(t))
        js_monitor = (gate * js_per_token).sum() / gate_sum
    stats = {
        "tam_div": divergence_per_token.detach().mean(),
        "tam_js": js_monitor.detach(),
        "tam_gate_mean": gate.detach().mean(),
        "tam_mass_kept": mass_kept.detach().mean(),
        "tam_corr_mass": corr_mass_mean.detach(),
        "tam_n": n,
    }
    return loss, stats
