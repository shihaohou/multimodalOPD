"""Rule-based grading for OPD eval — no LLM / API calls.

Reuses the SAME grader as OPD training (`mathruler.grade_answer` via
`vigos.answer_utils`) plus option-letter and normalized-exact matching. Handles
the common cases (math/numeric equivalence, multiple-choice letters, formatting
differences) deterministically and for free; an LLM judge is only needed as a
fallback for genuinely free-form answers (`--grader llm`).
"""

from __future__ import annotations

import re

from vigos.answer_utils import grade_answer
from vigos.eval_utils import extract_model_answer


def _normalize(value: str) -> str:
    text = re.sub(r"^\s*answer\s*[:：]\s*", "", str(value or "").strip(), flags=re.I)
    text = text.strip().strip("`'\"().").strip()
    return " ".join(text.casefold().split())


def attempt_correct(completion: str, ground_truth: str) -> bool:
    """Is this single completion correct vs the ground truth (rule-based)?"""
    prediction = extract_model_answer(completion)
    gt = str(ground_truth or "").strip()
    if not prediction or not gt:
        return False
    # 1) math / symbolic / numeric equivalence (mathruler — same as training).
    try:
        if grade_answer(prediction, gt):
            return True
    except Exception:
        pass
    # 2) normalized exact match.
    pred_norm, gt_norm = _normalize(prediction), _normalize(gt)
    if pred_norm and pred_norm == gt_norm:
        return True
    # 3) multiple choice: gt is a single option letter (e.g. "B" / "(B)").
    option = re.fullmatch(r"\(?([a-z])\)?", gt_norm)
    if option and re.search(rf"\b{option.group(1)}\b", pred_norm):
        return True
    return False
