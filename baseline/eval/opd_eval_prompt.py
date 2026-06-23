"""General eval prompt for OPD-trained models.

Mirrors the vanilla OPD *training* prompt: the dataset's own ``problem`` text plus
an optional, dataset-agnostic boxed-answer suffix, with **no** ViGOS
``<description>``/``<think>`` scaffolding and **no** assistant prefill. Keeping the
eval prompt identical to training keeps the harness general (not tied to ViGOS)
and consistent across datasets.
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from baseline.opd_data_collator import (
    OPD_SYSTEM_PROMPT,
    format_opd_student_prompt,
)

GENERAL_PROMPT_DESCRIPTION = (
    "Paper unified prompt: system (CoT + \\boxed{}) + user(image + question)"
)


def build_general_eval_messages(
    problem: str,
    images: list[Image.Image],
    *,
    system_prompt: str = OPD_SYSTEM_PROMPT,
    suffix: str = "",
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for image in images:
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


def build_general_eval_prompt(
    processor: Any,
    problem: str,
    images: list[Image.Image],
    *,
    suffix: str = "",
) -> str:
    messages = build_general_eval_messages(problem, images, suffix=suffix)
    # No assistant prefill: the model generates freely (matches OPD training).
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
