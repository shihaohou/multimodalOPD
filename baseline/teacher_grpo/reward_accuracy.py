"""Self-contained accuracy + format reward for Vision-SR1 GRPO (ms-swift plugin).

Registers two ORMs (no math_verify / mathruler needed in the swift env):
  vqa_accuracy : 1.0 if the model's final boxed answer matches `solution`
  vqa_format   : 1.0 if the completion contains a \\boxed{...}

Use with:
  --external_plugins baseline/teacher_grpo/reward_accuracy.py
  --reward_funcs vqa_accuracy vqa_format

The dataset must carry a `solution` column (prepare_vision_sr1.py writes it);
ms-swift passes it to __call__ as the `solution` kwarg.
"""

from __future__ import annotations

import re
from typing import List

from swift.rewards import ORM, orms


def _extract_boxed(text: str) -> str:
    text = str(text or "")
    idx = text.rfind("\\boxed{")
    if idx < 0:
        match = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.I | re.S)
        return match[-1].strip() if match else ""
    cursor = idx + len("\\boxed{")
    depth = 1
    start = cursor
    while cursor < len(text):
        char = text[cursor]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:cursor].strip()
        cursor += 1
    return text[start:].strip()


def _normalize(value: str) -> str:
    value = str(value or "").strip()
    value = re.sub(r"^\s*answer\s*[:：]\s*", "", value, flags=re.I)
    value = value.strip().strip("`'\"().").strip()
    return " ".join(value.casefold().split())


def _is_match(prediction: str, ground_truth: str) -> bool:
    pred = _normalize(prediction)
    gt = _normalize(ground_truth)
    if not pred or not gt:
        return False
    if pred == gt:
        return True
    try:
        if abs(float(pred.replace(",", "")) - float(gt.replace(",", ""))) < 1e-4:
            return True
    except (ValueError, OverflowError):
        pass
    # Multiple-choice: gt is a single option letter like "B" / "(B)".
    option = re.fullmatch(r"\(?([a-z])\)?", gt)
    if option and re.search(rf"\b{option.group(1)}\b", pred):
        return True
    return False


class VQAAccuracy(ORM):
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        return [
            1.0 if _is_match(_extract_boxed(completion), gt) else 0.0
            for completion, gt in zip(completions, solution)
        ]


class VQAFormat(ORM):
    """Paper format: reasoning in <reason></reason> + final answer in \\boxed{}."""

    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        for completion in completions:
            text = str(completion or "")
            has_reason = bool(re.search(r"<reason>.*?</reason>", text, flags=re.S))
            has_boxed = bool(re.search(r"\\boxed\{.+?\}", text, flags=re.S))
            rewards.append(1.0 if (has_reason and has_boxed) else 0.0)
        return rewards


orms["vqa_accuracy"] = VQAAccuracy
orms["vqa_format"] = VQAFormat
