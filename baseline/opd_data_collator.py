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

from vigos.answer_utils import normalize_reference_answer
from vigos.data_collator import (
    ViGOSDataCollator,
    _as_rgb_image,
    _format_reasoning_reference,
    _message_with_optional_image,
)

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


# Unified system prompt for ALL stages (teacher GRPO, student OPD, eval), from the
# paper appendix. Keeps teacher and student structurally aligned, which is what OPD
# needs (the teacher scores the student's tokens under the same prompt).
OPD_SYSTEM_PROMPT = (
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
            image = _as_rgb_image(feature.get("images", feature.get("image")))
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
