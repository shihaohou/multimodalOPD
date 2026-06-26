"""Data collation for vanilla multimodal On-Policy Distillation (OPD).

Unlike :class:`~vigos.data_collator.ViGOSDataCollator`, this collator builds only
the *student* prompt and uses the dataset's own ``problem`` text directly (no
``<description>`` scaffolding, no assistant prefill). The student and the frozen
teacher are scored on this same non-privileged prompt, which keeps the baseline
dataset-agnostic so the prompt does not have to change when the dataset changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

from vigos.answer_utils import normalize_reference_answer
from vigos.data_collator import (
    ViGOSDataCollator,
    _as_rgb_image,
    _format_reasoning_reference,
    _message_with_optional_image,
)

# Qwen2.5-VL / Qwen3-VL share the SAME HF image processor, which *infers* the
# channel axis from the raw array shape (before any resize) and which aborts on
# extreme aspect ratios. Two degenerate geometries crash it regardless of model:
#   * a side of 1 or 3 px (e.g. a 1-px-tall sliver): the size-1/3 spatial axis is
#     mistaken for the channel axis, so a 3-element mean/std hits a "1-channel"
#     image -> "ValueError: mean must have 1 elements if it is an iterable, got 3";
#   * aspect ratio > 200: smart_resize raises "absolute aspect ratio must be < 200".
# Center-pad the offending image so both sides are >= one patch and the ratio stays
# in range. Only pathological images are touched; normal ones pass through as-is.
_MIN_IMAGE_SIDE = 28
_MAX_ASPECT_RATIO = 180  # safely under the processor's hard limit of 200


def _safe_rgb_image(value: Any) -> Image.Image:
    """RGB-convert (via vigos) and center-pad away geometries the processor rejects."""
    image = _as_rgb_image(value)
    width, height = image.size
    target_w = max(width, _MIN_IMAGE_SIDE)
    target_h = max(height, _MIN_IMAGE_SIDE)
    # Grow the short side so max/min ratio stays within smart_resize's allowed range.
    if max(target_w, target_h) > _MAX_ASPECT_RATIO * min(target_w, target_h):
        if target_w < target_h:
            target_w = -(-target_h // _MAX_ASPECT_RATIO)  # ceil division
        else:
            target_h = -(-target_w // _MAX_ASPECT_RATIO)
    if target_w == width and target_h == height:
        return image
    canvas = Image.new("RGB", (target_w, target_h))
    canvas.paste(image, ((target_w - width) // 2, (target_h - height) // 2))
    return canvas

# Dataset-agnostic instruction appended to the raw problem so the rollout still
# emits a parseable final answer for the answer-accuracy metric / downstream eval.
# Set to "" (e.g. via --opd_prompt_suffix "") to use the raw dataset prompt only.
OPD_DEFAULT_PROMPT_SUFFIX = (
    "\nPlease reason step by step, and put your final answer within \\boxed{}."
)


def format_opd_student_prompt(
    problem: Any,
    suffix: str = OPD_DEFAULT_PROMPT_SUFFIX,
) -> str:
    """Vanilla OPD prompt = the dataset's problem text plus an optional suffix."""
    problem_text = str(problem).strip()
    if suffix:
        return f"{problem_text}{suffix}"
    return problem_text


# Short system prompt for the student OPD rollout AND eval. In OPD the frozen teacher
# scores the student's sampled tokens under this SAME prompt (one prompt is built; see
# OPDTrainer.compute_loss), so it must match the format the teacher emits -- keep teacher
# GRPO + student + eval on this one prompt.
#
# Deliberately minimal: a single <think>...</think> reasoning channel + a \boxed{} final
# answer (OPD-main / ViCuR convention). We dropped the long Vero-style instructions and the
# <answer> tags because the tags are not load-bearing here: the OPD loss is full-completion
# reverse KL (it does NOT slice the <think>/<answer> spans, unlike ViGOS), and answer
# extraction keys on \boxed{} everywhere (extract_boxed_content scans the whole completion;
# <answer> is only an eval fallback).
#
# Swap-in alternatives for ablations / a different teacher:
#   OPD_SYSTEM_PROMPT_FREECOT     - OPD-main free-CoT + \boxed, no tags (pair with a
#                                   non-format teacher, e.g. stock Qwen3-VL-8B-Instruct)
#   OPD_SYSTEM_PROMPT_REASON_TAGS - the earlier <reason></reason> + \boxed variant
OPD_SYSTEM_PROMPT = (
    "You are a helpful assistant. First think step by step inside "
    "<think> </think> tags, then give the final answer in \\boxed{}."
)

# OPD-main / DeepSeek-R1 convention: free-text CoT + \boxed{}, no rigid tags. Pair with a
# non-format-trained teacher (the student then has no strong tag signal to learn from).
OPD_SYSTEM_PROMPT_FREECOT = (
    "A conversation between user and assistant. The user asks a question, and the "
    "assistant solves it. The assistant first thinks about the reasoning process in "
    "the mind and then provides the user with the answer. The final answer MUST BE put "
    "in \\boxed{}."
)

# Earlier variant: CoT wrapped in <reason></reason> + \boxed{}. Kept for the ablation.
OPD_SYSTEM_PROMPT_REASON_TAGS = (
    "A conversation between user and assistant. The user asks a question, and the "
    "assistant solves it. The assistant first thinks about the reasoning process in "
    "the mind and then provides the user with the answer. The reasoning process "
    "should be enclosed within <reason></reason> tags. The final answer MUST BE put "
    "in \\boxed{}."
)


def build_opd_messages(
    problem: Any,
    image: Any,
    *,
    system_prompt: str = OPD_SYSTEM_PROMPT,
    suffix: str = "",
) -> list[dict[str, Any]]:
    """[system, user(image + question)] — the paper's unified template."""
    content: list[dict[str, Any]] = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": format_opd_student_prompt(problem, suffix)})
    messages: list[dict[str, Any]] = []
    if system_prompt:
        # Qwen-VL apply_chat_template requires list-of-parts content per message.
        messages.append(
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
        )
    messages.append({"role": "user", "content": content})
    return messages


@dataclass
class OPDDataCollator(ViGOSDataCollator):
    """Builds only the (non-privileged) student prompt for OPD training."""

    system_prompt: str = OPD_SYSTEM_PROMPT
    opd_prompt_suffix: str = ""

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        student_messages: list[list[dict[str, Any]]] = []
        student_prompt_texts: list[str] = []
        student_images: list[Any] = []
        problems: list[str] = []
        references: list[str] = []
        answers: list[str] = []
        sample_ids: list[int] = []

        for local_idx, feature in enumerate(features):
            image = _safe_rgb_image(feature.get("images", feature.get("image")))
            problem = str(feature["problem"]).strip()
            reference = _format_reasoning_reference(feature, self.answer_field)
            answer = normalize_reference_answer(feature.get(self.answer_field))
            sample_id = int(feature.get("problem_id", local_idx))

            student_message = build_opd_messages(
                problem,
                image,
                system_prompt=self.system_prompt,
                suffix=self.opd_prompt_suffix,
            )
            student_messages.append(student_message)
            # No assistant prefill: the model freely generates its own response.
            student_prompt_texts.append(
                self.processor.apply_chat_template(
                    student_message,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            student_images.append(image)
            problems.append(problem)
            references.append(reference)
            answers.append(answer)
            sample_ids.append(sample_id)

        result: dict[str, Any] = {}
        # Reuse ViGOSDataCollator._encode for tokenization/padding/truncation.
        result.update(self._encode("student", student_messages))
        result["student_prompt_texts"] = student_prompt_texts
        result["student_images"] = student_images
        result["vigos_problems"] = problems
        result["vigos_references"] = references
        result["vigos_answers"] = answers
        result["sample_ids"] = torch.tensor(sample_ids, dtype=torch.long)
        return result
