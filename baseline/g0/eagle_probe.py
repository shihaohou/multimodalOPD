"""EAGLE faithful causal attribution for G0 — "where it looks" + "what it relies on".

EAGLE (arXiv 2509.22496) is a *black-box, causal* visual attribution: it perturbs
image sub-regions and reads how the target token's probability changes, then
greedily orders regions by an insertion (sufficiency) + deletion (necessity)
objective. Unlike the gradient/attention probes (LH, GLIMPSE) it never trusts a
raw attention map — it measures what the answer actually *depends on*. That is
exactly the G0 cross-check: does the region the model needs match the GT box
(looking), and does the answer need the image at all (using)?

This module bridges our loaded Qwen model to the vendored EAGLE explainer
(:mod:`baseline.g0.eagle_src`):

* :class:`G0EagleAdaptor` — EAGLE's "MLLM" callable. Given perturbed images it
  returns ``P(answer tokens | perturbed image)`` under **our** OPD prompt (so the
  explained distribution is the G0 rollout's), scoring only the ``\\boxed{...}``
  answer tokens (the last ``K`` rows ⇒ ``logits_to_keep=K`` keeps it cheap).
* :func:`eagle_probe` — run the submodular search for one sample/condition and
  reduce it to the G0 metrics:
    - ``iou_eagle`` / ``pointing_eagle`` / ``area_eagle`` — the important-region
      map vs the GT box (**looking**, but causal, not attention);
    - ``visual_reliance`` = ``org − baseline`` (image lift over a blank image),
      ``text_reliance`` = ``baseline`` (answer prob with no image = prior),
      ``visual_fraction`` = ``(org−baseline)/org`` (**using**: ~0 ⇒ prior-driven);
    - ``sufficiency`` = insertion AUC, ``necessity`` = deletion AUC (causal sanity).

Cost: EAGLE is perturbation-based (~hundreds of forwards/sample), so the driver
downsizes the image (``eagle_image_size``) and uses a modest region count — this
is a small-budget diagnostic, not a full-8k pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from PIL import Image

from baseline.g0 import metrics
from baseline.g0.answer_spans import CompletionSpan, resolve_answer_spans
from baseline.g0.eagle_src import EfficientMLLMSubModularExplanationVisionV2, add_value, sub_region_division
from baseline.g0.engine import BoxNorm, G0Model, build_messages


def _resize_max_side(img: Image.Image, max_side: int) -> Image.Image:
    """Downsize so the longer side is ``max_side`` (keep aspect); never upscale."""
    w, h = img.size
    scale = max_side / float(max(w, h))
    if scale >= 1.0:
        return img
    return img.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.BILINEAR)


def _auc(x, y) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or x.size != y.size:
        return float("nan")
    order = np.argsort(x)
    return float(np.trapz(y[order], x[order]))


def faithfulness_auc(jf: dict) -> tuple[float, float]:
    """(insertion AUC = sufficiency, deletion AUC = necessity), EAGLE convention.

    Insertion: add regions best-first; high AUC ⇒ a few regions already recover the
    answer (sufficient). Deletion: remove regions best-first; LOW AUC ⇒ removing
    them collapses the answer (necessary).
    """
    try:
        region_area = list(jf["region_area"])
        ins = list(jf["insertion_score"])
        dele = list(jf["deletion_score"])
        n = min(len(region_area), len(ins), len(dele))
        region_area, ins, dele = region_area[:n], ins[:n], dele[:n]
        insertion_area = np.array([0.0] + region_area)
        deletion_area = 1.0 - insertion_area
        insertion_score = np.array([dele[-1]] + ins)
        deletion_score = np.array([ins[-1]] + dele)
        return _auc(insertion_area, insertion_score), _auc(deletion_area, deletion_score)
    except Exception:
        return float("nan"), float("nan")


class G0EagleAdaptor:
    """EAGLE "MLLM" callable: perturbed image(s) → ``P(answer tokens)``.

    Plain object (not ``nn.Module``) — EAGLE only needs ``.device`` and
    ``__call__(images)``. ``images`` is EAGLE's BGR-float tensor ``[H,W,3]`` or a
    batch ``[B,H,W,3]``; returns ``[N]`` or ``[B,N]`` answer-token probabilities
    (``N`` = #answer tokens). The chat ``text`` is fixed (pixels don't change it),
    so each call only re-runs the processor over the perturbed pixels and splices
    in the precomputed ``prompt+answer`` token sequence (EAGLE's own trick).
    """

    def __init__(
        self,
        gm: G0Model,
        problem: str,
        *,
        hint_bbox: Optional[BoxNorm],
        eval_image: Image.Image,
        eval_ids: torch.Tensor,
        target_positions: list[int],
        target_ids: torch.Tensor,
        system_prompt: Optional[str] = None,
    ):
        self.gm = gm
        self.device = gm.device
        self.problem = problem
        self.eval_ids = eval_ids.detach().to("cpu")
        self.target_positions = list(target_positions)
        self.target_ids = target_ids.detach().to("cpu")
        self.K = int(self.target_ids.numel())
        self.image_token_id = int(gm.parts.image_token_id)
        self._n_img_tokens = int((self.eval_ids == self.image_token_id).sum())
        messages = build_messages(eval_image, problem, hint_bbox=hint_bbox, system_prompt=system_prompt)
        self.text = gm.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # ---- image coercion (EAGLE BGR-float tensor → RGB PIL) ----
    def _to_pil(self, image: torch.Tensor) -> Image.Image:
        arr = image[..., [2, 1, 0]].clamp(0, 255).round().to(torch.uint8).cpu().numpy()  # BGR→RGB
        return Image.fromarray(arr)

    def _normalize(self, images):
        if isinstance(images, torch.Tensor):
            if images.dim() == 3:
                return [self._to_pil(images)], True
            return [self._to_pil(im) for im in images], False
        if isinstance(images, (list, tuple)):
            return [self._to_pil(im) for im in images], False
        return [self._to_pil(images)], True

    @torch.no_grad()
    def __call__(self, images):
        pils, single = self._normalize(images)
        b = len(pils)
        proc = self.gm.processor(text=[self.text] * b, images=pils, return_tensors="pt")
        # Visual-token count must match the precomputed prompt's (same dims ⇒ it does).
        got = int((proc["input_ids"] == self.image_token_id).sum()) // max(1, b)
        if got != self._n_img_tokens:
            raise ValueError(f"EAGLE adaptor: perturbed image yields {got} visual tokens, "
                             f"prompt expects {self._n_img_tokens} (resize/grid mismatch).")
        dev = self.device
        ids = self.eval_ids.to(dev).unsqueeze(0).expand(b, -1)
        kwargs = dict(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            pixel_values=proc["pixel_values"].to(dev),
            image_grid_thw=proc["image_grid_thw"].to(dev),
            use_cache=False,
        )
        try:
            out = self.gm.model(**kwargs, logits_to_keep=self.K)
            sel = out.logits  # [B, K, vocab] = last K rows = the answer rows, in order
        except TypeError:  # older signature without logits_to_keep
            out = self.gm.model(**kwargs)
            rows = torch.tensor([t - 1 for t in self.target_positions], device=dev)
            sel = out.logits.index_select(1, rows)  # [B, N, vocab]
        probs = torch.softmax(sel.float(), dim=-1)  # [B, N, vocab]
        tgt = self.target_ids.to(dev).view(1, self.K, 1).expand(probs.shape[0], -1, -1)
        p = probs.gather(2, tgt).squeeze(-1)  # [B, N]
        return p[0] if single else p


@dataclass
class EagleResult:
    iou_eagle: float          # important-region map vs GT (mask IoU)
    bbox_iou: float           # envelope box vs GT box (normalized)
    pointing_eagle: float     # argmax pixel in GT box
    area_eagle: float         # fraction of pixels in the important region
    energy: float             # positive attribution mass in GT box
    visual_reliance: float    # org − baseline (image lift, raw prob)
    text_reliance: float      # baseline (answer prob with no image = prior)
    visual_fraction: float    # (org − baseline) / org   (∈ ~[0,1]; low ⇒ prior-driven)
    visual_log_lift: float    # mean Δlog p(answer tok): org_logp − baseline_logp (less diluted by easy tokens)
    org_score: float          # P(answer | full image)   (mean token prob)
    baseline_score: float     # P(answer | blank image)
    org_logp: float           # mean log P(answer | full image)
    baseline_logp: float      # mean log P(answer | blank image)
    sufficiency: float        # insertion AUC
    necessity: float          # deletion AUC (LOW ⇒ region is necessary)
    n_regions: int
    region_mode_used: str     # slico | slic | grid (which backend actually ran)
    boxed_span_mode: str
    pred_box_norm: Optional[BoxNorm]
    attribution_map: Optional[np.ndarray] = None  # [H,W] downsized, for viz
    pred_mask: Optional[np.ndarray] = None        # [H,W] bool, the important region


def eagle_probe(
    gm: G0Model,
    image: Image.Image,
    problem: str,
    bbox: BoxNorm,
    completion_ids: torch.Tensor,
    *,
    hint_bbox: Optional[BoxNorm] = None,
    boxed_span: Optional[CompletionSpan] = None,
    answer_k: int = 8,
    n_regions: int = 49,
    search_scope: int = 8,
    pending_samples: int = 4,
    update_step: int = 10,
    lambda1: float = 1.0,
    lambda2: float = 1.0,
    batch_size: int = 8,
    eagle_image_size: int = 448,
    region_mode: str = "auto",
    threshold: str = "mean",
    top_frac: float = 0.25,
    keep_map: bool = False,
) -> EagleResult:
    """Run EAGLE for one (sample, condition) and reduce to the G0 metrics.

    ``completion_ids`` is the rollout's generated tokens (the answer being
    explained). ``boxed_span`` (completion coords) selects the answer tokens; if
    None it is resolved (``\\boxed{}`` → last-``answer_k`` fallback).
    """
    device = gm.device
    img = _resize_max_side(image.convert("RGB"), eagle_image_size)

    spans = resolve_answer_spans(completion_ids, gm.tokenizer, answer_k)
    span = boxed_span if boxed_span is not None else spans.primary
    s, e = span
    s = max(0, min(s, int(completion_ids.numel())))
    e = max(s + 1, min(e, int(completion_ids.numel())))

    # Re-anchor the fixed answer tokens after the DOWNSIZED prompt (we explain the
    # rollout's answer text; its original positions don't matter).
    from baseline.g0.engine import build_inputs

    inputs = build_inputs(gm, img, problem, hint_bbox=hint_bbox)
    prompt_len = int(inputs["input_ids"].shape[1])
    comp = completion_ids.to(device)
    full_ids = torch.cat([inputs["input_ids"][0], comp])
    target_positions = [prompt_len + j for j in range(s, e)]
    max_pos = max(target_positions)  # last answer token's position
    eval_ids = full_ids[:max_pos]    # input ends right before the last answer token
    target_ids = comp[s:e]

    adaptor = G0EagleAdaptor(
        gm, problem, hint_bbox=hint_bbox, eval_image=img, eval_ids=eval_ids,
        target_positions=target_positions, target_ids=target_ids,
    )

    image_bgr = np.array(img)[:, :, ::-1].copy()  # RGB→BGR, HxWx3 uint8
    V_set, region_mode_used = sub_region_division(
        image_bgr.astype(np.uint8), n_regions, mode=region_mode, return_mode=True)
    explainer = EfficientMLLMSubModularExplanationVisionV2(
        adaptor, lambda1=lambda1, lambda2=lambda2, search_scope=search_scope,
        pending_samples=pending_samples, update_step=update_step, batch_size=batch_size,
    )
    S_set, jf = explainer(image_bgr.astype(np.float32), V_set)

    # ---- attribution map → looking metrics (resolution-agnostic geometry) ----
    amap = add_value(S_set, jf)[0][:, :, 0].astype(np.float64)  # [H,W] in [0,1]
    res = metrics.iou_map_vs_gt(amap, bbox, sigma=0.0, threshold=threshold, top_frac=top_frac)
    pred_mask = (metrics.binarize_top_frac(amap, top_frac) if threshold == "top_frac"
                 else metrics.binarize_mean_relu(amap))
    pred_box = metrics.bbox_from_mask(pred_mask)
    h_grid, w_grid = amap.shape
    pred_norm = metrics.grid_box_to_norm(pred_box, h_grid, w_grid) if pred_box else None

    # ---- reliance / faithfulness (using metrics) ----
    org_tok = np.asarray(jf.get("org_score", []), dtype=np.float64)
    base_tok = np.asarray(jf.get("baseline_score", []), dtype=np.float64)
    org = float(org_tok.mean()) if org_tok.size else float("nan")
    base = float(base_tok.mean()) if base_tok.size else float("nan")
    visual_reliance = org - base
    visual_fraction = (visual_reliance / org) if org > 1e-6 else float("nan")
    # Log-lift: mean Δlog p over answer tokens. Less diluted than the raw-prob mean
    # by high-prob "easy" tokens (punctuation/common words) in a multi-token answer.
    org_logp = float(np.log(np.clip(org_tok, 1e-12, None)).mean()) if org_tok.size else float("nan")
    base_logp = float(np.log(np.clip(base_tok, 1e-12, None)).mean()) if base_tok.size else float("nan")
    visual_log_lift = org_logp - base_logp
    ins_auc, del_auc = faithfulness_auc(jf)

    return EagleResult(
        iou_eagle=float(res["mask_iou"]),
        bbox_iou=float(res["bbox_iou"]),
        pointing_eagle=float(res["pointing"]),
        area_eagle=float(pred_mask.mean()),
        energy=float(res["energy"]) if np.isfinite(res["energy"]) else 0.0,
        visual_reliance=float(visual_reliance),
        text_reliance=float(base),
        visual_fraction=float(visual_fraction),
        visual_log_lift=float(visual_log_lift),
        org_score=float(org),
        baseline_score=float(base),
        org_logp=float(org_logp),
        baseline_logp=float(base_logp),
        sufficiency=float(ins_auc),
        necessity=float(del_auc),
        n_regions=len(V_set),
        region_mode_used=region_mode_used,
        boxed_span_mode=spans.mode,
        pred_box_norm=pred_norm,
        attribution_map=amap if keep_map else None,
        pred_mask=pred_mask if keep_map else None,
    )
