"""CPU numerical-equivalence test for the evidence recompute path (Stage-2).

The recompute engine (``evidence_attn_mode='recompute'``) replaces the legacy
forced-**eager** ``output_attentions`` forward with a fast SDPA forward plus a
reconstruction of *only the attention rows the saliency engine needs*, from the
model's own captured post-RoPE / post-QK-norm ``q/k/v``. This test proves that
reconstruction is numerically identical to what eager attention produces, so the
speed fix does not change the loss or its gradient.

Two checks, both on CPU (no GPU, no network — random-init tiny model):

1. ``test_unit_recompute_rows`` — ``_recompute_attention_rows`` vs transformers'
   own ``eager_attention_forward`` on random ``q/k/v`` + a causal mask. This is
   the core math (softmax / scaling / causal+pad mask / GQA ``repeat_kv``).
2. ``test_engine_equivalence`` — the WHOLE saliency engine, eager
   (:func:`compute_token_saliency_maps`) vs recompute
   (:func:`compute_token_saliency_maps_from_qkv`), driven through a real tiny
   ``Qwen2_5_VLTextModel`` so the actual ``capture_qkv_attention`` dispatch is
   exercised. Also checks the student gradient flows into ``q/k/v`` projections.

Run::

    python -m baseline.evidence.test_recompute_equiv
"""

from __future__ import annotations

import os
import sys

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from baseline.evidence.saliency_engine import (
    SaliencyModelParts,
    _recompute_attention_rows,
    capture_qkv_attention,
    compute_token_saliency_maps,
    compute_token_saliency_maps_from_qkv,
    repeat_kv_heads,
)


def test_unit_recompute_rows() -> float:
    """``_recompute_attention_rows`` must equal ``eager_attention_forward`` rows."""
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import eager_attention_forward
    from types import SimpleNamespace

    torch.manual_seed(0)
    H, H_kv, S, d = 6, 2, 17, 8
    n_rep = H // H_kv
    scaling = d ** -0.5
    q = torch.randn(1, H, S, d, dtype=torch.float32)
    k = torch.randn(1, H_kv, S, d, dtype=torch.float32)
    v = torch.randn(1, H_kv, S, d, dtype=torch.float32)

    # Causal additive mask [1,1,S,S] (0 on/under the diagonal, -inf above), as the
    # text model builds for eager.
    neg = torch.finfo(torch.float32).min
    causal = torch.triu(torch.full((S, S), neg), diagonal=1).view(1, 1, S, S)

    module = SimpleNamespace(num_key_value_groups=n_rep, training=False)
    _, eager_w = eager_attention_forward(module, q, k, v, causal, scaling=scaling)
    #   eager_w: [1, H, S, S]

    # The rows the two-hop routing needs (arbitrary valid query positions).
    q_positions = torch.tensor([4, 9, 16])
    k_full = repeat_kv_heads(k[0], n_rep)  # [H, S, d]
    got = _recompute_attention_rows(q[0], k_full, scaling, causal[0], q_positions)  # [H, nq, S]
    want = eager_w[0].index_select(1, q_positions)  # [H, nq, S]

    diff = (got - want).abs().max().item()
    print(f"[unit] _recompute_attention_rows vs eager_attention_forward: max|Δ|={diff:.3e}")
    assert diff < 1e-5, f"unit row-recompute mismatch {diff}"

    # And against a None mask (pure causal rebuilt from positions) — same result.
    got_none = _recompute_attention_rows(q[0], k_full, scaling, None, q_positions)
    diff_none = (got_none - want).abs().max().item()
    print(f"[unit] None-mask causal rebuild vs eager:                    max|Δ|={diff_none:.3e}")
    assert diff_none < 1e-5, f"None-mask causal mismatch {diff_none}"
    return max(diff, diff_none)


def _build_tiny_text_model(seed: int = 0):
    """A tiny random-init ``Qwen2_5_VLTextModel`` + a matching lm_head + parts."""
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLTextConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLTextModel

    torch.manual_seed(seed)
    hidden, n_heads, n_kv, head_dim = 64, 4, 2, 16
    cfg = Qwen2_5_VLTextConfig(
        vocab_size=128,
        hidden_size=hidden,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv,
        head_dim=head_dim,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_scaling={"type": "mrope", "mrope_section": [2, 2, 4]},
        use_sliding_window=False,
    )
    model = Qwen2_5_VLTextModel(cfg).eval().to(torch.float32)
    lm_head = torch.nn.Linear(hidden, cfg.vocab_size, bias=False)
    torch.nn.init.normal_(lm_head.weight, std=0.02)
    parts = SaliencyModelParts(
        text_model=model,
        lm_head=lm_head,
        n_heads=n_heads,
        n_kv_heads=n_kv,
        head_dim=head_dim,
        hidden_size=hidden,
        image_token_id=0,
        spatial_merge_size=2,
    )
    return model, parts, cfg


def test_engine_equivalence() -> float:
    """Full saliency engine: eager output_attentions vs SDPA-capture recompute."""
    model, parts, cfg = _build_tiny_text_model(seed=1)
    S = 14
    inputs_embeds = torch.randn(1, S, cfg.hidden_size, dtype=torch.float32, requires_grad=True)
    position_ids = torch.arange(S).view(1, 1, S).expand(3, 1, S).contiguous()

    # Routing positions (all causal-valid): 4 visual patches -> 2x2 grid.
    visual_positions = torch.tensor([0, 1, 2, 3])
    reason_k = torch.tensor([5, 6, 7])
    reason_q = torch.tensor([4, 5, 6])
    answer_q = torch.tensor([10, 11])
    direction_ids = torch.tensor([12, 34])
    grid_hw = (2, 2)
    layers = tuple(range(cfg.num_hidden_layers))

    common = dict(
        batch_index=0,
        answer_query_positions=answer_q,
        reason_key_positions=reason_k,
        reason_query_positions=reason_q,
        visual_positions=visual_positions,
        direction_ids=direction_ids,
        grid_hw=grid_hw,
        layers=layers,
        signed=True,
        parts=parts,
    )

    # --- eager path: full [H,S,S] attentions ---------------------------------
    model.config._attn_implementation = "eager"
    eager_out = model(
        inputs_embeds=inputs_embeds,
        position_ids=position_ids,
        output_attentions=True,
        output_hidden_states=True,
        use_cache=False,
    )
    eager_map = compute_token_saliency_maps(
        model, eager_out.attentions, eager_out.hidden_states, **common
    )

    # --- recompute path: SDPA forward + capture q/k/v ------------------------
    model.config._attn_implementation = "sdpa"
    with capture_qkv_attention(model, model, layers) as captured:
        cap_out = model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            output_hidden_states=True,
            use_cache=False,
        )
    assert all(l in captured for l in layers), "capture_qkv_attention did not fire on all layers"
    recompute_map = compute_token_saliency_maps_from_qkv(
        model, captured, cap_out.hidden_states, **common
    )

    diff = (eager_map - recompute_map.detach()).abs().max().item()
    scale = eager_map.abs().max().item()
    print(
        f"[engine] eager vs recompute saliency map: max|Δ|={diff:.3e} "
        f"(map scale={scale:.3e}, rel={diff / max(scale, 1e-9):.3e})"
    )
    assert diff < 1e-3 * max(scale, 1.0), f"engine map mismatch {diff} (scale {scale})"

    # Gradient sanity: the recompute map must backprop into the q/k/v projections.
    recompute_map.float().sum().backward()
    g = model.layers[layers[-1]].self_attn.v_proj.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, "no grad into v_proj"
    gi = inputs_embeds.grad
    assert gi is not None and gi.abs().sum() > 0, "no grad into inputs_embeds"
    print(f"[engine] grad into v_proj.weight Σ|g|={g.abs().sum().item():.3e} (live gradient OK)")
    return diff


def test_engine_equivalence_padded() -> float:
    """Same, but with a LEFT-PADDED batch — the path the config-flip approach would
    have silently broken (no causal+padding mask). The captured mask must be the
    real padded one, so eager and recompute still agree on the unpadded row."""
    model, parts, cfg = _build_tiny_text_model(seed=2)
    B, S, pad = 2, 16, 5  # row 1 has `pad` left-pad tokens
    inputs_embeds = torch.randn(B, S, cfg.hidden_size, dtype=torch.float32)
    attention_mask = torch.ones(B, S, dtype=torch.long)
    attention_mask[1, :pad] = 0  # left padding on row 1
    # mRoPE positions: padded row starts counting at its first real token.
    pos_row = torch.arange(S)
    pos_row1 = torch.clamp(torch.arange(S) - pad, min=0)
    position_ids = torch.stack([pos_row, pos_row1]).view(1, B, S).expand(3, B, S).contiguous()

    b = 1  # score the padded row; routing positions live in its REAL (post-pad) region
    visual_positions = torch.tensor([pad, pad + 1, pad + 2, pad + 3])
    reason_k = torch.tensor([pad + 5, pad + 6])
    reason_q = torch.tensor([pad + 4, pad + 5])
    answer_q = torch.tensor([pad + 8, pad + 9])
    direction_ids = torch.tensor([7, 21])
    common = dict(
        batch_index=b,
        answer_query_positions=answer_q,
        reason_key_positions=reason_k,
        reason_query_positions=reason_q,
        visual_positions=visual_positions,
        direction_ids=direction_ids,
        grid_hw=(2, 2),
        layers=tuple(range(cfg.num_hidden_layers)),
        signed=True,
        parts=parts,
    )
    layers = common["layers"]

    model.config._attn_implementation = "eager"
    eager_out = model(
        inputs_embeds=inputs_embeds, attention_mask=attention_mask, position_ids=position_ids,
        output_attentions=True, output_hidden_states=True, use_cache=False,
    )
    eager_map = compute_token_saliency_maps(model, eager_out.attentions, eager_out.hidden_states, **common)

    model.config._attn_implementation = "sdpa"
    with capture_qkv_attention(model, model, layers) as captured:
        cap_out = model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, position_ids=position_ids,
            output_hidden_states=True, use_cache=False,
        )
    # The captured mask must be a real (4D) padded mask, not None — else padding is ignored.
    assert captured[layers[-1]]["mask"] is not None, "captured mask is None — padding would be dropped!"
    recompute_map = compute_token_saliency_maps_from_qkv(model, captured, cap_out.hidden_states, **common)

    diff = (eager_map - recompute_map.detach()).abs().max().item()
    scale = eager_map.abs().max().item()
    print(f"[padded] eager vs recompute (left-padded row): max|Δ|={diff:.3e} (scale={scale:.3e})")
    assert diff < 1e-3 * max(scale, 1.0), f"padded engine mismatch {diff} (scale {scale})"
    return diff


def main() -> None:
    print("=" * 72)
    print("EVIDENCE recompute-path CPU equivalence test")
    print("=" * 72)
    u = test_unit_recompute_rows()
    e = test_engine_equivalence()
    p = test_engine_equivalence_padded()
    print("-" * 72)
    print(f"PASS — unit max|Δ|={u:.2e}, engine max|Δ|={e:.2e}, padded max|Δ|={p:.2e}")
    print("recompute == eager to fp32 tolerance; the Stage-2 speed fix is loss-preserving.")


if __name__ == "__main__":
    main()
