"""Shared Qwen-VL plumbing for the G0 diagnostic: load, prompt, generate, forward.

This wraps the bits both probes (LH + GLIMPSE) need so the probe modules stay
about *the analysis*, not about transformers boilerplate. It reuses the existing
evidence-engine loaders / part-resolver where possible:

  * model load → ``baseline.evidence.sanity_check._load_model`` (eager attn,
    bf16, ``output_attentions``-capable) + ``resolve_model_parts``
    (``image_token_id`` / ``spatial_merge_size`` / head dims from config).
  * the condition prompts → OPD's plain image+question prompt plus optional bbox
    hint text. One ``OPD_SYSTEM_PROMPT`` for all conditions.
  * correctness → ``baseline.eval.grading.attempt_correct`` (rule grader).

Two forward helpers expose the eager attention the probes consume:
  * :func:`nograd_attention_forward` — cheap, for LH (and head calibration).
  * :func:`grad_attention_forward` — keeps the graph so GLIMPSE can take
    ``torch.autograd.grad`` of a response scalar w.r.t. the attention maps. The
    model is **not** frozen (params keep ``requires_grad``), so the eager
    attention tensors are differentiable; we use ``autograd.grad`` (not
    ``.backward()``) so no per-parameter ``.grad`` is ever materialized.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if __package__ is None or __package__ == "":  # allow `python baseline/g0/engine.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

# Repo-internal modules (vigos data collator, evidence engine, hint collator) pull
# in transformers; import them lazily inside the functions that need them so this
# module — and the pure-torch/numpy math in glimpse.py / localization_heads.py —
# imports with just torch+numpy (and is CPU-unit-testable).
if TYPE_CHECKING:  # pragma: no cover
    from baseline.evidence.saliency_engine import SaliencyModelParts

BoxNorm = tuple[float, float, float, float]

VISIBLE_HINT_TEMPLATE = (
    "Hint: the evidence needed to answer the question is inside the bounding box "
    "{bbox} (normalized to [0,1], top-left origin, [x1, y1, x2, y2])."
)

HIDDEN_HINT_TEMPLATE = (
    "Hint: the evidence needed to answer the question is inside the bounding box "
    "{bbox} (normalized to [0,1], top-left origin, [x1, y1, x2, y2]). Use this only "
    "to decide where to look in the image, then answer the question directly. Do NOT "
    "mention the bounding box, the coordinates, this hint, or a crop in your "
    "reasoning or your answer."
)


def format_bbox_hint(
    bbox: BoxNorm,
    *,
    template: str = HIDDEN_HINT_TEMPLATE,
    decimals: int = 2,
) -> str:
    coords = "[" + ", ".join(f"{v:.{decimals}f}" for v in bbox) + "]"
    return template.format(bbox=coords)


# --------------------------------------------------------------------- correctness
def is_correct(completion_text: str, solution: str) -> bool:
    """Rule-based answer correctness (reuses the OPD eval grader, then a fallback)."""
    try:
        from baseline.eval.grading import attempt_correct

        return bool(attempt_correct(completion_text, solution))
    except Exception:
        from vigos.answer_utils import (
            extract_boxed_content,
            grade_answer,
            normalize_reference_answer,
        )

        pred = extract_boxed_content(completion_text) or completion_text
        return bool(
            grade_answer(
                normalize_reference_answer(pred), normalize_reference_answer(solution)
            )
        )


# ------------------------------------------------------------------------- model
@dataclass
class G0Model:
    """A loaded VLM + everything the probes need to address its image tokens."""

    name: str
    model: Any
    processor: Any
    tokenizer: Any
    parts: "SaliencyModelParts"
    device: torch.device

    @property
    def num_layers(self) -> int:
        return len(self.parts.text_model.layers)

    @property
    def num_heads(self) -> int:
        return self.parts.n_heads


def load_g0_model(
    path: str,
    name: str,
    *,
    attn: str = "eager",
    dtype: str | torch.dtype = "bfloat16",
    device: str = "cuda",
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
) -> G0Model:
    """Load a Qwen-VL model + processor for the diagnostic.

    ``attn`` must be ``eager`` for ``output_attentions`` to return real (and, for
    GLIMPSE, differentiable) attention tensors — SDPA/Flash return ``None``.
    ``max_pixels`` caps the image resolution (→ #visual tokens → the dominant
    memory term in the grad forward); set it for the GLIMPSE pass on big images.
    """
    from transformers import AutoProcessor

    from baseline.evidence.sanity_check import _load_model
    from baseline.evidence.saliency_engine import resolve_model_parts

    torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True, use_fast=False)
    tokenizer = getattr(processor, "tokenizer", processor)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None:
        if min_pixels is not None:
            image_processor.min_pixels = int(min_pixels)
            if hasattr(image_processor, "size") and isinstance(image_processor.size, dict):
                image_processor.size["shortest_edge"] = int(min_pixels)
        if max_pixels is not None:
            image_processor.max_pixels = int(max_pixels)
            if hasattr(image_processor, "size") and isinstance(image_processor.size, dict):
                image_processor.size["longest_edge"] = int(max_pixels)

    # _load_model hardcodes .to("cuda").eval(); honor a non-cuda device for dry runs.
    if device == "cuda":
        model = _load_model(path, attn, torch_dtype)
    else:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            path, attn_implementation=attn, dtype=torch_dtype, trust_remote_code=True
        ).to(device).eval()

    parts = resolve_model_parts(model)
    return G0Model(
        name=name,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        parts=parts,
        device=torch.device(device if device != "cuda" else model.device),
    )


# ------------------------------------------------------------------------ prompts
def build_messages(
    image: Any,
    problem: str,
    *,
    hint_bbox: Optional[BoxNorm] = None,
    system_prompt: Optional[str] = None,
    hint_decimals: int = 2,
    hint_template: Optional[str] = None,
) -> list:
    """The chat messages for a condition (plain vs C2 silent-hint).

    Factored out of :func:`build_inputs` so the EAGLE adaptor can build the chat
    *text* once and re-run the processor over many perturbed images.
    """
    from baseline.opd_data_collator import OPD_SYSTEM_PROMPT, build_opd_messages, format_opd_student_prompt

    if system_prompt is None:
        system_prompt = OPD_SYSTEM_PROMPT
    if hint_bbox is None:
        return build_opd_messages(problem, image, system_prompt=system_prompt, suffix="")
    hint = format_bbox_hint(hint_bbox, template=hint_template or HIDDEN_HINT_TEMPLATE, decimals=hint_decimals)
    content: list[dict[str, object]] = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": f"{format_opd_student_prompt(problem, '')}\n\n{hint}"})
    messages: list[dict[str, object]] = []
    if system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    messages.append({"role": "user", "content": content})
    return messages


def build_inputs(
    gm: G0Model,
    image: Any,
    problem: str,
    *,
    hint_bbox: Optional[BoxNorm] = None,
    system_prompt: Optional[str] = None,
    hint_decimals: int = 2,
    hint_template: Optional[str] = None,
) -> dict[str, torch.Tensor]:
    """Tokenize one (image, question) into model inputs for a condition.

    ``hint_bbox=None`` builds the plain student/natural prompt (C1 teacher, C3
    student). A box builds the **C2** privileged teacher prompt: the same image +
    question with the silent bbox hint appended (``HIDDEN_HINT_TEMPLATE``:
    "use this to decide where to look ... do NOT mention the box"). Identical to the
    natural prompt apart from that appended sentence. ``system_prompt=None`` uses
    the shared ``OPD_SYSTEM_PROMPT`` (the one prompt teacher GRPO / student / eval
    all agree on).
    """
    messages = build_messages(
        image, problem, hint_bbox=hint_bbox, system_prompt=system_prompt,
        hint_decimals=hint_decimals, hint_template=hint_template,
    )
    text = gm.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = gm.processor(text=[text], images=[image], return_tensors="pt")
    return {k: v.to(gm.device) for k, v in inputs.items()}


def sanitize_completion_ids(gm: G0Model, completion_ids: torch.Tensor) -> torch.Tensor:
    """Replace any image/video placeholder ids in a sampled completion with pad.

    Re-running ``prompt + completion`` through the model would otherwise trip
    Qwen's placeholder-count check (the completion has no pixels behind those
    ids). Text completions almost never contain them; this is a cheap guard.
    """
    bad = {int(gm.parts.image_token_id)}
    vid = getattr(gm.model.config, "video_token_id", None)
    if vid is not None:
        bad.add(int(vid))
    pad_id = gm.tokenizer.pad_token_id or gm.tokenizer.eos_token_id or 0
    out = completion_ids.clone()
    for tid in bad:
        out[out == tid] = pad_id
    return out


@torch.no_grad()
def generate_completion(
    gm: G0Model,
    inputs: dict[str, torch.Tensor],
    *,
    max_new_tokens: int = 320,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    seed: Optional[int] = None,
) -> tuple[torch.Tensor, str]:
    """Greedy (default) or sampled rollout. Returns (completion_ids, text)."""
    if seed is not None and do_sample:
        torch.manual_seed(seed)
    prompt_len = int(inputs["input_ids"].shape[1])
    gen_kwargs: dict[str, Any] = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=gm.tokenizer.pad_token_id,
    )
    if do_sample:
        gen_kwargs.update(temperature=temperature, top_p=top_p)
    gen = gm.model.generate(**inputs, **gen_kwargs)
    completion_ids = gen[0, prompt_len:]
    completion_ids = sanitize_completion_ids(gm, completion_ids)
    text = gm.tokenizer.decode(completion_ids, skip_special_tokens=True)
    return completion_ids, text


# -------------------------------------------------------------- grid / positions
def visual_grid(gm: G0Model, full_ids: torch.Tensor, image_grid_thw: torch.Tensor):
    """(visual_positions ``[P]``, (H_grid, W_grid)) for one sample.

    Visual positions are the image-placeholder runs in ``full_ids``; the merged
    patch grid is ``(grid_h // merge, grid_w // merge)`` and must equal ``P``.
    """
    parts = gm.parts
    visual_positions = (full_ids == parts.image_token_id).nonzero(as_tuple=True)[0]
    grid = image_grid_thw[0]
    merge = parts.spatial_merge_size
    h_grid, w_grid = int(grid[1]) // merge, int(grid[2]) // merge
    if h_grid * w_grid != int(visual_positions.numel()):
        raise ValueError(
            f"[{gm.name}] grid {h_grid}x{w_grid}={h_grid * w_grid} != "
            f"#visual tokens {visual_positions.numel()} — check spatial_merge_size."
        )
    return visual_positions, (h_grid, w_grid)


def _forward_kwargs(inputs: dict[str, torch.Tensor], full_ids: torch.Tensor):
    full_ids_b = full_ids.unsqueeze(0)
    kwargs = dict(
        input_ids=full_ids_b,
        attention_mask=torch.ones_like(full_ids_b),
        output_attentions=True,
        use_cache=False,
    )
    for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
        if key in inputs:
            kwargs[key] = inputs[key]
    return kwargs


@torch.no_grad()
def nograd_attention_forward(gm: G0Model, inputs: dict, full_ids: torch.Tensor):
    """Eager forward returning ``out.attentions`` (no graph) — for LH / calibration."""
    return gm.model(**_forward_kwargs(inputs, full_ids))


def grad_attention_forward(gm: G0Model, inputs: dict, full_ids: torch.Tensor):
    """Eager forward that KEEPS the graph so GLIMPSE can grad through attentions.

    Returns the model output; ``out.attentions[l]`` require grad (the model is not
    frozen and attn is eager). The caller takes ``torch.autograd.grad`` of a
    response scalar w.r.t. these tensors, then drops the graph.
    """
    return gm.model(**_forward_kwargs(inputs, full_ids))
