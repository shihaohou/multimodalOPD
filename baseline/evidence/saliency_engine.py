"""Differentiable logit-decomposition saliency engine (faithful to Saliency_R1).

This is a **differentiable** re-implementation of the saliency map used by
peterant330/Saliency_R1 (``trl/grpo_trainer.py`` lines ~1815-1847). Saliency_R1
computes the map under ``torch.no_grad()`` from the generation KV-cache + the
per-step ``output_attentions`` and uses it only as a (non-differentiable) GRPO
reward. We instead compute the *same quantity* inside a single grad-enabled
forward over ``prompt + completion`` so the map carries gradients into the
student's attention and value projections — that is what lets an evidence
alignment loss move *where the student looks*.

The Saliency_R1 definition, per generated **answer** token, is a two-hop
attention routing through the **reason** span:

    logits = 0
    for layer l:
        V_vis      = value_states[l] at the visual-token positions          # [H, P, d]
        think_attn = attn(answer_query -> reason_key)                       # [n_ans, H, T]
        token_attn = attn(reason_query -> visual_key)                       # [H, T, P]
        agg        = think_attn @ token_attn                                # [n_ans, H, P]
        sv         = (agg * V_vis) reshaped to [n_ans, P, H*d]
        logits    += o_proj_l(sv)                                           # [n_ans, P, hidden]
    logits = norm(logits) * ||logits||                                      # rescale, keep magnitude
    logits = logits / ||hidden_state(answer_query)||                        # answer-token normalization
    out    = lm_head(logits)                                                # [n_ans, P, vocab]
    sal    = out[:, :, generated_answer_token_id]                           # [n_ans, P]
    map    = relu(sal).reshape(H_grid, W_grid)                              # spatial map per answer token

Two adaptations vs Saliency_R1:

1.  **Differentiable value states.** Saliency_R1 reads ``past_key_values`` from
    ``generate`` (no grad). A plain forward does not expose the value cache, so
    we recompute it: ``v = v_proj(input_layernorm(hidden_states[l]))``. Value
    projections carry no rotary embedding, so this is byte-for-byte the cached
    value (and it now has a gradient). Qwen3's QK-norm touches only q/k, so this
    holds for Qwen3-VL too.
2.  **Per-token maps.** Saliency_R1 sums the per-answer-token logits into ONE map
    per sample (``out[...].sum(dim=0)``) to compare against one bbox. The
    evidence loss needs per-token teacher/student alignment, so we keep the
    ``[n_ans, ...]`` axis (no ``.sum(0)``).

Everything is **config-driven** (head counts, ``head_dim``, ``image_token_id``,
``spatial_merge_size`` are read from the model, not hardcoded) so the engine runs
on both Qwen2.5-VL and Qwen3-VL. Whether a *cross-size* teacher/student pair
shares a patch grid (required by the evidence loss) is an empirical check — see
``sanity_check.py``.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


def repeat_kv_heads(value: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand grouped-query value heads: ``[n_kv, S, d] -> [n_kv*n_rep, S, d]``.

    Matches ``transformers`` ``repeat_kv`` / Saliency_R1 ``repeat_v`` but on a
    head-first ``[heads, seq, dim]`` layout (no batch axis — we operate one
    sample at a time).
    """
    if n_rep == 1:
        return value
    n_kv, seq, dim = value.shape
    return (
        value[:, None, :, :]
        .expand(n_kv, n_rep, seq, dim)
        .reshape(n_kv * n_rep, seq, dim)
    )


@contextlib.contextmanager
def capture_qkv_attention(model: nn.Module, text_model: nn.Module, layer_ids):
    """Capture the post-RoPE / post-QK-norm ``q,k,v`` (+ mask + scale) for a SUBSET
    of decoder layers WITHOUT forcing eager — the forward stays on the model's
    configured fast kernel (SDPA/Flash). This is the recompute path's replacement
    for the legacy forced-eager ``output_attentions`` capture.

    Mechanism: ``transformers`` selects each attention module's kernel per-forward
    via ``ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]`` and hands it
    the already-RoPE'd / QK-normed / pre-``repeat_kv`` ``query,key,value`` plus the
    prepared ``attention_mask`` and ``scaling`` (exactly what ``eager_attention_forward``
    consumes). We **temporarily override that kernel's registry entry** with a
    wrapper that delegates to the real kernel (so the attention output — and thus
    the OPD logits — are byte-for-byte unchanged) and, for the target layers only,
    stashes those tensors. They stay in the autograd graph, so the saliency engine
    can redo just the rows it needs and gradients still flow into ``q/k/v`` -> the
    projections.

    Crucially we do **not** touch ``config._attn_implementation``: a custom impl
    string would make ``create_causal_mask`` (which checks the mask registry's
    global mapping) early-exit and return ``None``, silently dropping the
    causal+padding mask. Leaving the config on its real kernel keeps mask
    construction correct (left-padding, sliding window, …); the override only swaps
    the attention *function*, so the captured mask is exactly the one the model
    used. Returns ``{layer_idx: {"q","k","v","mask","scaling"}}``.

    ``model`` is accepted for call-site symmetry with ``force_eager_attention`` but
    only ``text_model`` (the ``.layers`` stack) is used.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    targets = {id(text_model.layers[l].self_attn): l for l in layer_ids}
    stash: dict[int, dict[str, Any]] = {}

    attn_cfg = getattr(text_model.layers[layer_ids[0]].self_attn, "config", None)
    orig_impl = getattr(attn_cfg, "_attn_implementation", None) if attn_cfg is not None else None
    if orig_impl is None or orig_impl == "eager":
        raise RuntimeError(
            "capture_qkv_attention needs a non-eager fast kernel (sdpa/flash) whose "
            f"attention function it can override; got _attn_implementation={orig_impl!r}. "
            "Run the student/teacher with attn_implementation=sdpa, or use "
            "evidence_attn_mode='eager'."
        )
    # The genuine kernel to delegate to (read from the global mapping so a stray
    # pre-existing local override can't make us delegate to ourselves).
    real_fn = ALL_ATTENTION_FUNCTIONS._global_mapping[orig_impl]

    def capture_fn(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
        out = real_fn(
            module, query, key, value, attention_mask,
            scaling=scaling, dropout=dropout, **kwargs,
        )
        lid = targets.get(id(module))
        if lid is not None:
            stash[lid] = {
                "q": query,
                "k": key,
                "v": value,
                "mask": attention_mask,
                "scaling": getattr(module, "scaling", None) if scaling is None else scaling,
            }
        return out

    # Override the kernel entry locally (getitem checks _local_mapping first), then
    # restore whatever was there before — config is untouched throughout.
    local = ALL_ATTENTION_FUNCTIONS._local_mapping
    had_prev = orig_impl in local
    prev = local.get(orig_impl)
    local[orig_impl] = capture_fn
    try:
        yield stash
    finally:
        if had_prev:
            local[orig_impl] = prev
        else:
            local.pop(orig_impl, None)


@dataclass
class SaliencyModelParts:
    """Resolved handles + dims needed to run the saliency engine on a model.

    Resolution is defensive because the module nesting differs across
    transformers versions (``model.language_model`` vs ``model.model.language_model``)
    and model families (Qwen2.5-VL vs Qwen3-VL).
    """

    text_model: nn.Module  # the decoder stack: has ``.layers`` and ``.norm``
    lm_head: nn.Module
    n_heads: int
    n_kv_heads: int
    head_dim: int
    hidden_size: int
    image_token_id: int
    spatial_merge_size: int

    @property
    def num_kv_groups(self) -> int:
        return self.n_heads // self.n_kv_heads


def _first_present(obj: object, names: tuple[str, ...]):
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return None


def _resolve_text_model(model: nn.Module) -> nn.Module:
    """Return the decoder module exposing ``.layers`` and ``.norm``."""
    candidates = (
        "language_model",
        "model.language_model",
        "model.model.language_model",
        "model",
        "model.model",
    )
    for path in candidates:
        node = model
        ok = True
        for part in path.split("."):
            if hasattr(node, part):
                node = getattr(node, part)
            else:
                ok = False
                break
        if ok and hasattr(node, "layers") and hasattr(node, "norm"):
            return node
    raise AttributeError(
        "Could not resolve the text decoder (a module with `.layers` and `.norm`) "
        f"on {type(model).__name__}; checked {candidates}."
    )


def _infer_head_dims(
    text_model: nn.Module, hidden_size: int
) -> tuple[int, int, int]:
    """(n_heads, n_kv_heads, head_dim) from the first layer's attention module.

    Inferred from projection shapes (robust across families) and cross-checked
    against config when available. ``v_proj.out_features = n_kv_heads*head_dim``
    and ``q_proj.out_features = n_heads*head_dim``.
    """
    attn = text_model.layers[0].self_attn
    q_out = attn.q_proj.out_features
    kv_out = attn.v_proj.out_features
    head_dim = _first_present(attn, ("head_dim",))
    if head_dim is None:
        # No explicit head_dim: derive from config's head count (Qwen heads are
        # square, head_dim = hidden / num_attention_heads).
        cfg = _first_present(attn, ("config",)) or None
        n_heads_cfg = _first_present(cfg, ("num_attention_heads",)) if cfg else None
        if not n_heads_cfg:
            raise ValueError(
                "Cannot infer head_dim: attention module exposes neither "
                "`head_dim` nor a config with `num_attention_heads`."
            )
        head_dim = hidden_size // int(n_heads_cfg)
    head_dim = int(head_dim)
    return q_out // head_dim, kv_out // head_dim, head_dim


_PARTS_CACHE: dict[int, SaliencyModelParts] = {}


def resolve_model_parts(
    model: nn.Module,
    *,
    image_token_id: int | None = None,
    spatial_merge_size: int | None = None,
) -> SaliencyModelParts:
    """Resolve (and cache) the modules + dims the engine needs from ``model``.

    Pass ``image_token_id`` / ``spatial_merge_size`` explicitly to override the
    config lookup (e.g. when the processor and config disagree).
    """
    key = id(model)
    cached = _PARTS_CACHE.get(key)
    if cached is not None:
        return cached

    text_model = _resolve_text_model(model)
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        raise AttributeError(f"No output embedding / lm_head on {type(model).__name__}.")

    hidden_size = text_model.layers[0].self_attn.o_proj.in_features
    n_heads, n_kv_heads, head_dim = _infer_head_dims(text_model, hidden_size)

    cfg = model.config
    if image_token_id is None:
        image_token_id = _first_present(cfg, ("image_token_id", "image_token_index"))
    if image_token_id is None:
        raise ValueError(
            "image_token_id not found on model.config; pass it explicitly to "
            "resolve_model_parts()."
        )
    if spatial_merge_size is None:
        vision_cfg = getattr(cfg, "vision_config", None)
        spatial_merge_size = _first_present(
            vision_cfg or cfg, ("spatial_merge_size",)
        )
    if spatial_merge_size is None:
        spatial_merge_size = 2  # Qwen2.5-VL / Qwen3-VL default.

    parts = SaliencyModelParts(
        text_model=text_model,
        lm_head=lm_head,
        n_heads=int(n_heads),
        n_kv_heads=int(n_kv_heads),
        head_dim=int(head_dim),
        hidden_size=int(hidden_size),
        image_token_id=int(image_token_id),
        spatial_merge_size=int(spatial_merge_size),
    )
    _PARTS_CACHE[key] = parts
    return parts


def _ov_contrib(
    o_proj: nn.Module,
    think_attn: torch.Tensor,
    token_attn: torch.Tensor,
    v_vis: torch.Tensor,
    n_ans: int,
    n_patch: int,
    n_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """One layer's OV contribution: compose the two attention hops, weight the
    visual value states, fold heads, project. Shared by the eager and the
    recompute engines so the OV circuit is byte-identical between them.

    ``think_attn`` ``[n_ans, H, T]`` (answer->reason), ``token_attn`` ``[H, T, P]``
    (reason->visual), ``v_vis`` ``[H, P, d]`` -> ``[n_ans, P, hidden]``.
    """
    agg_attn = torch.einsum("aht,htp->ahp", think_attn, token_attn)  # [n_ans, H, P]
    sv = agg_attn.unsqueeze(-1) * v_vis.unsqueeze(0)                 # [n_ans, H, P, d]
    sv = sv.permute(0, 2, 1, 3).reshape(n_ans, n_patch, n_heads * head_dim)
    return o_proj(sv)                                               # [n_ans, P, hidden]


def _finalize_saliency(
    parts: SaliencyModelParts,
    logits_accum: torch.Tensor,
    hidden_last_b: torch.Tensor,
    answer_query_positions: torch.Tensor,
    direction_ids: torch.Tensor,
    grid_hw: tuple[int, int],
    signed: bool,
) -> torch.Tensor:
    """Saliency_R1 normalization + direction-only unembed (grpo_trainer.py
    1835-1847). Shared tail of both engines.

    Applies the final RMSNorm *direction* but restores the pre-norm magnitude,
    divides by the answer token's last-hidden norm, then contracts the lm_head
    weight row for each generated direction id (never materializing the full
    ``[n_ans, P, vocab]`` logits). ``[n_ans, P, hidden] -> [n_ans, H_grid, W_grid]``.
    """
    tm = parts.text_model
    normed = tm.norm(logits_accum)                                 # [n_ans, P, hidden]
    logits_accum = normed * logits_accum.norm(dim=-1, keepdim=True)
    h_ans = hidden_last_b.index_select(0, answer_query_positions)  # [n_ans, hidden]
    n_ans = int(h_ans.shape[0])
    n_patch = int(logits_accum.shape[1])
    hidden_norm = h_ans.norm(dim=-1).reshape(n_ans, 1, 1).clamp_min(1e-6)
    logits_accum = logits_accum / hidden_norm

    device = logits_accum.device
    direction_ids = direction_ids.to(device=device, dtype=torch.long)
    head_weight = getattr(parts.lm_head, "weight", None)
    if head_weight is not None:
        w_sel = head_weight.index_select(0, direction_ids).to(logits_accum.dtype)  # [n_ans, hidden]
        sel = torch.einsum("aph,ah->ap", logits_accum, w_sel)      # [n_ans, P]
        bias = getattr(parts.lm_head, "bias", None)
        if bias is not None:
            sel = sel + bias.index_select(0, direction_ids).to(sel.dtype).unsqueeze(1)
    else:  # exotic head without .weight — fall back to the dense projection
        out = parts.lm_head(logits_accum)                          # [n_ans, P, vocab]
        sel = out[torch.arange(n_ans, device=device), :, direction_ids]
    if not signed:
        sel = torch.relu(sel)

    h_grid, w_grid = grid_hw
    if h_grid * w_grid != n_patch:
        raise ValueError(
            f"grid {h_grid}x{w_grid} = {h_grid * w_grid} != #visual tokens {n_patch}; "
            "the visual-token -> patch-grid mapping is inconsistent (check "
            "spatial_merge_size / image_grid_thw)."
        )
    return sel.reshape(n_ans, h_grid, w_grid)


def _recompute_attention_rows(
    q_b: torch.Tensor,
    k_full: torch.Tensor,
    scaling: float,
    attention_mask_b: torch.Tensor | None,
    query_positions: torch.Tensor,
) -> torch.Tensor:
    """Post-softmax attention weights for a SUBSET of query rows, matching
    ``transformers`` ``eager_attention_forward`` exactly but only for the rows we
    need (the whole point of the recompute path — never materialize ``[H,S,S]``).

    ``q_b`` ``[H, S, d]`` and ``k_full`` ``[H, S, d]`` are the model's own
    post-RoPE / post-QK-norm query & (group-expanded) key states, captured from the
    attention interface; ``scaling`` and ``attention_mask_b`` are the same values
    the model passed to that interface. Returns ``[H, n_q, S]``.
    """
    q_sel = q_b.index_select(1, query_positions)                   # [H, n_q, d]
    scores = torch.matmul(q_sel, k_full.transpose(-1, -2)) * scaling  # [H, n_q, S]
    key_len = k_full.shape[-2]
    if attention_mask_b is not None:
        # Additive (or boolean) mask the model built (causal + left-padding, and
        # any sliding window). Slice the selected query rows; eager uses
        # attention_mask[..., :key_len]. Broadcasts over heads if mask has 1 head.
        m = attention_mask_b
        if m.dtype == torch.bool:
            m = torch.zeros_like(m, dtype=scores.dtype).masked_fill(~m, float("-inf"))
        m_rows = m.index_select(-2, query_positions)[..., :key_len]  # [mh, n_q, S]
        scores = scores + m_rows
    else:
        # No mask passed (pure causal, no padding — e.g. the CPU equivalence test):
        # disallow keys strictly after each query's absolute position.
        key_pos = torch.arange(key_len, device=scores.device)
        causal = key_pos.unsqueeze(0) > query_positions.unsqueeze(-1)  # [n_q, S]
        scores = scores.masked_fill(causal.unsqueeze(0), float("-inf"))
    return torch.softmax(scores, dim=-1, dtype=torch.float32).to(q_b.dtype)


def compute_token_saliency_maps(
    model: nn.Module,
    attentions: tuple[torch.Tensor, ...],
    hidden_states: tuple[torch.Tensor, ...],
    *,
    batch_index: int,
    answer_query_positions: torch.Tensor,
    reason_key_positions: torch.Tensor,
    reason_query_positions: torch.Tensor,
    visual_positions: torch.Tensor,
    direction_ids: torch.Tensor,
    grid_hw: tuple[int, int],
    layers: tuple[int, ...] | None = None,
    signed: bool = True,
    parts: SaliencyModelParts | None = None,
) -> torch.Tensor:
    """Per-answer-token saliency maps for one sample.

    Args:
        model: the VLM (student, grad; or teacher, no_grad). Only its decoder
            value/output projections, final norm and lm_head are used.
        attentions: ``output_attentions`` tuple, length = #layers, each
            ``[B, H, S, S]`` (requires the **eager** attention implementation).
        hidden_states: ``output_hidden_states`` tuple, length #layers+1, each
            ``[B, S, hidden]`` (``hidden_states[l]`` = residual input to layer l;
            ``hidden_states[-1]`` = last layer output).
        batch_index: which sample (b) in the batch to score.
        answer_query_positions: ``[n_ans]`` absolute positions of the query rows
            that *predict* the answer tokens (predictor position = answer token
            position - 1).
        reason_key_positions: ``[T]`` absolute positions of the reason tokens
            used as attention **keys** in the answer->reason hop.
        reason_query_positions: ``[T]`` absolute positions of the query rows that
            predict the reason tokens (reason->visual hop). Same length T.
        visual_positions: ``[P]`` absolute positions of the image placeholder
            tokens (``input_ids == image_token_id``).
        direction_ids: ``[n_ans]`` vocab ids to read off lm_head — the actual
            generated answer token at each answer position (Stage 1A direction).
        grid_hw: ``(H_grid, W_grid)`` with ``H_grid * W_grid == P``; the merged
            visual patch grid (raw grid // spatial_merge_size).
        layers: subset of decoder layers to sum over (None = all). Fewer layers
            => cheaper OV/lm_head work (attention materialization is unaffected).
        signed: keep the sign of the contribution (True; preserves negative =
            "image argues against this token") or ReLU it (False; Saliency_R1's
            positive-only reward).
        parts: pre-resolved :class:`SaliencyModelParts` (else resolved here).

    Returns:
        ``[n_ans, H_grid, W_grid]`` saliency maps, one per answer token, in the
        compute dtype with a live gradient when ``model`` is in a grad context.
    """
    if parts is None:
        parts = resolve_model_parts(model)
    tm = parts.text_model
    n_heads, head_dim = parts.n_heads, parts.head_dim
    n_rep = parts.num_kv_groups

    layer_ids = tuple(range(len(attentions))) if layers is None else tuple(layers)

    b = batch_index
    # ``attentions`` may be a full tuple (output_attentions) or a {layer: weights}
    # dict (hook-captured subset); index by the first used layer either way.
    device = attentions[layer_ids[0]].device
    answer_query_positions = answer_query_positions.to(device)
    reason_key_positions = reason_key_positions.to(device)
    reason_query_positions = reason_query_positions.to(device)
    visual_positions = visual_positions.to(device)
    n_ans = int(answer_query_positions.shape[0])
    n_patch = int(visual_positions.shape[0])

    logits_accum: torch.Tensor | None = None
    for l in layer_ids:
        layer = tm.layers[l]
        attn = layer.self_attn

        # --- differentiable value states for the visual tokens -----------------
        # v = v_proj(input_layernorm(residual_in[l])); no RoPE on values, so this
        # equals Saliency_R1's cached past_key_values value states (now w/ grad).
        h_in = layer.input_layernorm(hidden_states[l][b])           # [S, hidden]
        v = attn.v_proj(h_in)                                       # [S, n_kv*head_dim]
        seq = v.shape[0]
        v = v.view(seq, parts.n_kv_heads, head_dim).permute(1, 0, 2)  # [n_kv, S, d]
        v = repeat_kv_heads(v, n_rep)                               # [H, S, d]
        v_vis = v.index_select(1, visual_positions)                # [H, P, d]

        # --- two-hop attention routing ----------------------------------------
        a = attentions[l][b]                                        # [H, S, S]
        # answer-query -> reason-key:  [H, n_ans, T] -> [n_ans, H, T]
        think_attn = a.index_select(1, answer_query_positions).index_select(
            2, reason_key_positions
        )
        think_attn = think_attn.permute(1, 0, 2).contiguous()      # [n_ans, H, T]
        # reason-query -> visual-key:  [H, T, P]
        token_attn = a.index_select(1, reason_query_positions).index_select(
            2, visual_positions
        )                                                          # [H, T, P]

        contrib = _ov_contrib(
            attn.o_proj, think_attn, token_attn, v_vis, n_ans, n_patch, n_heads, head_dim
        )
        logits_accum = contrib if logits_accum is None else logits_accum + contrib

    assert logits_accum is not None, "no layers selected for saliency"
    return _finalize_saliency(
        parts,
        logits_accum,
        hidden_states[-1][b],
        answer_query_positions,
        direction_ids,
        grid_hw,
        signed,
    )


def compute_token_saliency_maps_from_qkv(
    model: nn.Module,
    captured: dict[int, dict[str, torch.Tensor]],
    hidden_states: tuple[torch.Tensor, ...],
    *,
    batch_index: int,
    answer_query_positions: torch.Tensor,
    reason_key_positions: torch.Tensor,
    reason_query_positions: torch.Tensor,
    visual_positions: torch.Tensor,
    direction_ids: torch.Tensor,
    grid_hw: tuple[int, int],
    layers: tuple[int, ...] | None = None,
    signed: bool = True,
    parts: SaliencyModelParts | None = None,
) -> torch.Tensor:
    """Per-answer-token saliency maps — the **recompute** path (Stage-2).

    Identical output to :func:`compute_token_saliency_maps`, but instead of reading
    full ``[H, S, S]`` attention matrices captured from a forced-**eager** forward,
    it reconstructs only the handful of attention *rows* the two-hop routing needs
    (``answer_query`` ≤8 rows, ``reason_query`` T rows) from the model's own
    post-RoPE / post-QK-norm ``q``/``k``/``v`` states. Those were captured from the
    attention **interface** during a normal **SDPA/Flash** forward (see
    ``OPDEvidenceTrainer.capture_qkv_attention``), so:

    * the main forward stays SDPA (no ``S²`` eager tax on the other ~30 layers);
    * memory drops from ``L·H·S²`` retained to ``K·H·(n_ans+T)·S`` transient;
    * RoPE / QK-norm / GQA / the causal+padding mask are all the model's own — we
      only redo ``softmax((q·kᵀ)·scaling + mask)`` for the selected rows, which
      reproduces ``eager_attention_forward`` exactly (see ``test_recompute_equiv``).

    ``captured[l]`` holds ``{"q","k","v","mask","scaling"}``: ``q`` ``[B,H,S,d]``,
    ``k``/``v`` ``[B,H_kv,S,d]`` (pre-group-expansion, as the interface receives
    them), ``mask`` the additive/boolean attention mask the interface got (or
    ``None``), ``scaling`` the attention scale.
    """
    if parts is None:
        parts = resolve_model_parts(model)
    n_heads, head_dim = parts.n_heads, parts.head_dim
    n_rep = parts.num_kv_groups

    layer_ids = tuple(sorted(captured)) if layers is None else tuple(layers)

    b = batch_index
    device = captured[layer_ids[0]]["q"].device
    answer_query_positions = answer_query_positions.to(device)
    reason_key_positions = reason_key_positions.to(device)
    reason_query_positions = reason_query_positions.to(device)
    visual_positions = visual_positions.to(device)
    n_ans = int(answer_query_positions.shape[0])
    n_patch = int(visual_positions.shape[0])

    tm = parts.text_model
    logits_accum: torch.Tensor | None = None
    for l in layer_ids:
        layer = captured[l]
        q_b = layer["q"][b]                                         # [H, S, d]
        k_full = repeat_kv_heads(layer["k"][b], n_rep)             # [H, S, d]
        v_full = repeat_kv_heads(layer["v"][b], n_rep)            # [H, S, d]
        v_vis = v_full.index_select(1, visual_positions)           # [H, P, d]
        mask_b = layer["mask"][b] if layer["mask"] is not None else None
        scaling = float(layer["scaling"])

        # answer-query -> reason-key:  [H, n_ans, S] -> [n_ans, H, T]
        a_ans = _recompute_attention_rows(
            q_b, k_full, scaling, mask_b, answer_query_positions
        )
        think_attn = a_ans.index_select(2, reason_key_positions).permute(1, 0, 2).contiguous()
        # reason-query -> visual-key:  [H, T, S] -> [H, T, P]
        a_rea = _recompute_attention_rows(
            q_b, k_full, scaling, mask_b, reason_query_positions
        )
        token_attn = a_rea.index_select(2, visual_positions)       # [H, T, P]

        contrib = _ov_contrib(
            tm.layers[l].self_attn.o_proj,
            think_attn, token_attn, v_vis, n_ans, n_patch, n_heads, head_dim,
        )
        logits_accum = contrib if logits_accum is None else logits_accum + contrib

    assert logits_accum is not None, "no layers captured for saliency"
    return _finalize_saliency(
        parts,
        logits_accum,
        hidden_states[-1][b],
        answer_query_positions,
        direction_ids,
        grid_hw,
        signed,
    )
