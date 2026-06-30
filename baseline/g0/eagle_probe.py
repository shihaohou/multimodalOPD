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

from dataclasses import dataclass, field
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
        hint_template: Optional[str] = None,
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
        messages = build_messages(
            eval_image, problem, hint_bbox=hint_bbox, system_prompt=system_prompt,
            hint_template=hint_template,
        )
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

    def _process_images(self, pils):
        image_processor = getattr(self.gm.processor, "image_processor", None)
        if image_processor is not None:
            try:
                proc = image_processor(images=pils, return_tensors="pt")
                if "pixel_values" in proc and "image_grid_thw" in proc:
                    grid = proc["image_grid_thw"][0]
                    merge = int(self.gm.parts.spatial_merge_size)
                    got = int(grid[0]) * (int(grid[1]) // merge) * (int(grid[2]) // merge)
                    return proc, got
            except Exception:
                pass

        proc = self.gm.processor(text=[self.text] * len(pils), images=pils, return_tensors="pt")
        got = int((proc["input_ids"] == self.image_token_id).sum()) // max(1, len(pils))
        return proc, got

    @torch.no_grad()
    def __call__(self, images):
        pils, single = self._normalize(images)
        b = len(pils)
        proc, got = self._process_images(pils)
        # Visual-token count must match the precomputed prompt's (same dims ⇒ it does).
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
    pointing_at1: float
    energy_in_box: float
    iou_top10: float
    iou_top20: float
    deletion_logp_drop: float
    insertion_logp_recovery: float
    deletion_logp_drop_top10: float
    deletion_logp_drop_top20: float
    insertion_logp_recovery_top10: float
    insertion_logp_recovery_top20: float
    insertion_recovery_frac_top20: float
    token_map_mode: str
    token_map_count: int
    token_indices: list[int]
    token_details: list[dict] = field(default_factory=list)
    attribution_map: Optional[np.ndarray] = None  # [H,W] downsized, for viz
    pred_mask: Optional[np.ndarray] = None        # [H,W] bool, the important region
    token_maps: Optional[np.ndarray] = None       # [T,H,W] per-token EAGLE maps, for artifacts
    eagle_s_set: Optional[np.ndarray] = None      # EAGLE greedy regions, for official visualization
    eagle_json_file: Optional[dict] = None        # EAGLE scores, for official visualization


def _mean_log_probs(values) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.log(np.clip(arr, 1e-12, None)).mean())


def _top_area_mask(map_2d: np.ndarray, frac: float) -> np.ndarray:
    arr = np.asarray(map_2d, dtype=np.float64)
    if arr.size == 0:
        return np.zeros_like(arr, dtype=bool)
    k = max(1, int(np.ceil(arr.size * float(frac))))
    order = np.argsort(arr, axis=None)[::-1]
    mask = np.zeros(arr.size, dtype=bool)
    mask[order[:k]] = True
    return mask.reshape(arr.shape)


def _map_metrics(amap: np.ndarray, bbox: BoxNorm, threshold: str, top_frac: float) -> tuple[dict, np.ndarray, Optional[BoxNorm]]:
    res = metrics.iou_map_vs_gt(amap, bbox, sigma=0.0, threshold=threshold, top_frac=top_frac)
    pred_mask = (metrics.binarize_top_frac(amap, top_frac) if threshold == "top_frac"
                 else metrics.binarize_mean_relu(amap))
    pred_box = metrics.bbox_from_mask(pred_mask)
    h_grid, w_grid = amap.shape
    pred_norm = metrics.grid_box_to_norm(pred_box, h_grid, w_grid) if pred_box else None
    gt_mask = metrics.gt_box_to_grid_mask(bbox, h_grid, w_grid)
    res["iou_top10"] = metrics.mask_iou(_top_area_mask(amap, 0.10), gt_mask)
    res["iou_top20"] = metrics.mask_iou(_top_area_mask(amap, 0.20), gt_mask)
    return res, pred_mask, pred_norm


def _select_token_indices(s: int, e: int, limit: int) -> list[int]:
    idxs = list(range(int(s), int(e)))
    if limit and limit > 0 and len(idxs) > limit:
        pos = np.linspace(0, len(idxs) - 1, int(limit))
        idxs = [idxs[int(round(p))] for p in pos]
        idxs = list(dict.fromkeys(idxs))
    return idxs


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
    token_map_mode: str = "span",
    token_limit: int = 0,
    hint_template: Optional[str] = None,
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

    inputs = build_inputs(gm, img, problem, hint_bbox=hint_bbox, hint_template=hint_template)
    prompt_len = int(inputs["input_ids"].shape[1])
    comp = completion_ids.to(device)
    full_ids = torch.cat([inputs["input_ids"][0], comp])

    image_bgr = np.array(img)[:, :, ::-1].copy()  # RGB→BGR, HxWx3 uint8
    V_set, region_mode_used = sub_region_division(
        image_bgr.astype(np.uint8), n_regions, mode=region_mode, return_mode=True)

    def make_adaptor(ss: int, ee: int) -> G0EagleAdaptor:
        target_positions = [prompt_len + j for j in range(ss, ee)]
        max_pos = max(target_positions)  # last target token's position
        eval_ids = full_ids[:max_pos]    # input ends right before the last target token
        return G0EagleAdaptor(
            gm, problem, hint_bbox=hint_bbox, eval_image=img, eval_ids=eval_ids,
            target_positions=target_positions, target_ids=comp[ss:ee],
            hint_template=hint_template,
        )

    def run_once(ss: int, ee: int):
        adaptor = make_adaptor(ss, ee)
        explainer = EfficientMLLMSubModularExplanationVisionV2(
            adaptor, lambda1=lambda1, lambda2=lambda2, search_scope=search_scope,
            pending_samples=pending_samples, update_step=update_step, batch_size=batch_size,
        )
        s_set, jf = explainer(image_bgr.astype(np.float32), V_set.copy())
        amap = add_value(s_set, jf)[0][:, :, 0].astype(np.float64)
        return s_set, jf, amap

    def token_detail(j: int, jf: dict, amap: np.ndarray) -> dict:
        org_p = float(np.asarray(jf.get("org_score", [float("nan")]), dtype=np.float64).mean())
        base_p = float(np.asarray(jf.get("baseline_score", [float("nan")]), dtype=np.float64).mean())
        tok_res, _, _ = _map_metrics(amap, bbox, threshold, top_frac)
        return {
            "token_index": int(j),
            "token_id": int(comp[j].detach().cpu().item()),
            "token_text": gm.tokenizer.decode([int(comp[j].detach().cpu().item())],
                                              skip_special_tokens=False,
                                              clean_up_tokenization_spaces=False),
            "org_prob": org_p,
            "baseline_prob": base_p,
            "org_logp": _mean_log_probs([org_p]),
            "baseline_logp": _mean_log_probs([base_p]),
            "visual_log_lift": _mean_log_probs([org_p]) - _mean_log_probs([base_p]),
            "visual_reliance": org_p - base_p,
            "visual_fraction": (org_p - base_p) / org_p if org_p > 1e-6 else float("nan"),
            "pointing_at1": float(tok_res["pointing"]),
            "energy_in_box": float(tok_res["energy"]) if np.isfinite(tok_res["energy"]) else 0.0,
            "iou_top10": float(tok_res["iou_top10"]),
            "iou_top20": float(tok_res["iou_top20"]),
        }

    token_map_mode = str(token_map_mode or "span")
    if token_map_mode not in {"span", "per_token_mean", "per_token_max"}:
        raise ValueError(f"unknown token_map_mode={token_map_mode!r}")

    token_details: list[dict] = []
    token_maps_arr = None
    if token_map_mode == "span":
        S_set, jf, amap = run_once(s, e)
        token_indices = list(range(s, e))
        org_scores = np.asarray(jf.get("org_score", []), dtype=np.float64)
        base_scores = np.asarray(jf.get("baseline_score", []), dtype=np.float64)
        token_details = [
            {
                "token_index": int(j),
                "token_id": int(comp[j].detach().cpu().item()),
                "token_text": gm.tokenizer.decode([int(comp[j].detach().cpu().item())],
                                                  skip_special_tokens=False,
                                                  clean_up_tokenization_spaces=False),
                "org_prob": float(org_scores[k]) if k < org_scores.size else float("nan"),
                "baseline_prob": float(base_scores[k]) if k < base_scores.size else float("nan"),
                "org_logp": _mean_log_probs([org_scores[k]]) if k < org_scores.size else float("nan"),
                "baseline_logp": _mean_log_probs([base_scores[k]]) if k < base_scores.size else float("nan"),
                "visual_log_lift": (
                    _mean_log_probs([org_scores[k]]) - _mean_log_probs([base_scores[k]])
                    if k < min(org_scores.size, base_scores.size) else float("nan")
                ),
                "visual_reliance": (
                    float(org_scores[k] - base_scores[k])
                    if k < min(org_scores.size, base_scores.size) else float("nan")
                ),
                "visual_fraction": (
                    float((org_scores[k] - base_scores[k]) / org_scores[k])
                    if k < min(org_scores.size, base_scores.size) and org_scores[k] > 1e-6 else float("nan")
                ),
            }
            for k, j in enumerate(token_indices)
        ]
        agg_jf = jf
        agg_s_set = S_set
    else:
        token_indices = _select_token_indices(s, e, token_limit)
        token_maps = []
        for j in token_indices:
            _, jf_one, amap_one = run_once(j, j + 1)
            token_maps.append(amap_one)
            token_details.append(token_detail(j, jf_one, amap_one))
        token_maps_arr = np.stack(token_maps, axis=0) if token_maps else np.zeros((0, image_bgr.shape[0], image_bgr.shape[1]))
        amap = token_maps_arr.mean(axis=0) if token_map_mode == "per_token_mean" else token_maps_arr.max(axis=0)
        amap = amap - float(np.nanmin(amap))
        mx = float(np.nanmax(amap))
        amap = amap / mx if mx > 1e-8 else amap
        # Keep a span-level EAGLE run only for scalar insertion/deletion curves when
        # requested through per-token mode. Metrics and visualization use amap above.
        S_set, jf, _ = run_once(s, e)
        agg_jf = jf
        agg_s_set = S_set

    # ---- attribution map → looking metrics (resolution-agnostic geometry) ----
    res, pred_mask, pred_norm = _map_metrics(amap, bbox, threshold, top_frac)

    # ---- reliance / faithfulness (using metrics) ----
    target_adaptor = make_adaptor(s, e)
    src = torch.from_numpy(image_bgr.astype(np.float32))
    mask10 = torch.from_numpy(_top_area_mask(amap, 0.10).astype(np.float32))[:, :, None]
    mask20 = torch.from_numpy(_top_area_mask(amap, 0.20).astype(np.float32))[:, :, None]
    scoring_images = torch.stack([
        src,
        torch.zeros_like(src),
        src * (1.0 - mask10),
        src * mask10,
        src * (1.0 - mask20),
        src * mask20,
    ]).to(device)
    with torch.no_grad():
        full_scores = target_adaptor(scoring_images).detach().cpu().numpy()
    org_tok = np.asarray(full_scores[0], dtype=np.float64)
    base_tok = np.asarray(full_scores[1], dtype=np.float64)
    del10_logp = _mean_log_probs(full_scores[2])
    ins10_logp = _mean_log_probs(full_scores[3])
    del20_logp = _mean_log_probs(full_scores[4])
    ins20_logp = _mean_log_probs(full_scores[5])
    org = float(org_tok.mean()) if org_tok.size else float("nan")
    base = float(base_tok.mean()) if base_tok.size else float("nan")
    org_logp = _mean_log_probs(org_tok)
    base_logp = _mean_log_probs(base_tok)
    visual_reliance = org - base
    visual_fraction = (visual_reliance / org) if org > 1e-6 else float("nan")
    visual_log_lift = org_logp - base_logp
    ins_auc, del_auc = faithfulness_auc(agg_jf)
    rec_denom = org_logp - base_logp
    insertion_recovery_frac_top20 = (
        (ins20_logp - base_logp) / rec_denom
        if np.isfinite(rec_denom) and abs(rec_denom) > 1e-8 else float("nan")
    )

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
        pointing_at1=float(res["pointing"]),
        energy_in_box=float(res["energy"]) if np.isfinite(res["energy"]) else 0.0,
        iou_top10=float(res["iou_top10"]),
        iou_top20=float(res["iou_top20"]),
        deletion_logp_drop=float(org_logp - del20_logp),
        insertion_logp_recovery=float(ins20_logp - base_logp),
        deletion_logp_drop_top10=float(org_logp - del10_logp),
        deletion_logp_drop_top20=float(org_logp - del20_logp),
        insertion_logp_recovery_top10=float(ins10_logp - base_logp),
        insertion_logp_recovery_top20=float(ins20_logp - base_logp),
        insertion_recovery_frac_top20=float(insertion_recovery_frac_top20),
        token_map_mode=token_map_mode,
        token_map_count=len(token_indices),
        token_indices=[int(j) for j in token_indices],
        token_details=token_details,
        attribution_map=amap if keep_map else None,
        pred_mask=pred_mask if keep_map else None,
        token_maps=token_maps_arr if keep_map else None,
        eagle_s_set=np.asarray(agg_s_set) if keep_map else None,
        eagle_json_file=agg_jf if keep_map else None,
    )
