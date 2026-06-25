"""Deterministic, official metrics for the short-answer VLM benchmarks
(POPE / ChartQA / VQAv2). No LLM judge — each benchmark has a canonical,
reproducible scoring rule, ported faithfully from its reference implementation:

* POPE    — yes/no mapping from Li et al. 2023 (POPE) ``evaluate.py``; the harness
            reports accuracy / precision / recall / F1 / yes-ratio (F1 is the
            headline metric).
* ChartQA — relaxed accuracy (Methani et al. 2020 / PaLI): a numeric answer is
            correct within a 5 % relative tolerance, otherwise case-insensitive
            exact match.
* VQAv2   — official VQA soft accuracy (Antol et al. 2015 ``VQAEval``): answer
            normalization + ``min(1, agreement / 3)`` averaged leave-one-out over
            the 10 human answers.

These are pure functions (no I/O, no model) so they are unit-testable offline.
"""

from __future__ import annotations

import re

# --------------------------------------------------------------------------- POPE
_POPE_NEG_WORDS = {"no", "not", "none", "nothing", "without", "cannot", "isn't"}


def pope_label(text: str) -> str:
    """Map a free-form answer to ``"yes"`` / ``"no"`` (official POPE rule).

    POPE's reference grader keeps the first sentence, drops commas, then labels
    the answer ``no`` iff it contains a negation word (``no`` / ``not`` / ...),
    else ``yes``. Anything ambiguous therefore defaults to ``yes`` — exactly as
    the original evaluation does, so the numbers stay comparable.
    """
    cleaned = str(text or "").strip().lower()
    if "." in cleaned:
        cleaned = cleaned.split(".")[0]
    cleaned = cleaned.replace(",", " ")
    words = set(cleaned.split())
    if words & _POPE_NEG_WORDS or "n't" in cleaned:
        return "no"
    return "yes"


def pope_scores(labels: list[str], preds: list[str]) -> dict[str, float]:
    """accuracy / precision / recall / F1 / yes-ratio with ``yes`` as positive.

    ``labels`` / ``preds`` are equal-length lists of ``"yes"`` / ``"no"``.
    """
    tp = fp = tn = fn = 0
    for gt, pred in zip(labels, preds):
        if pred == "yes" and gt == "yes":
            tp += 1
        elif pred == "yes" and gt == "no":
            fp += 1
        elif pred == "no" and gt == "no":
            tn += 1
        else:  # pred == "no" and gt == "yes"
            fn += 1
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_ratio": (tp + fp) / total if total else 0.0,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "total": total,
    }


# ------------------------------------------------------------------------ ChartQA
def _chartqa_to_float(text: str) -> float | None:
    """Parse a numeric answer, tolerating ``%`` / ``$`` / thousands commas."""
    s = str(text or "").strip().replace(",", "").replace("$", "").strip()
    is_percent = s.endswith("%")
    if is_percent:
        s = s[:-1].strip()
    try:
        value = float(s)
    except ValueError:
        return None
    return value / 100.0 if is_percent else value


def relaxed_correctness(prediction: str, target: str, max_relative_change: float = 0.05) -> bool:
    """ChartQA relaxed accuracy (Methani et al. 2020).

    Numeric answers match within ``max_relative_change`` (default 5 %) relative
    error; everything else falls back to case-insensitive exact match.
    """
    pred_f = _chartqa_to_float(prediction)
    target_f = _chartqa_to_float(target)
    if pred_f is not None and target_f is not None:
        if target_f == 0.0:
            return pred_f == 0.0
        return abs(pred_f - target_f) / abs(target_f) <= max_relative_change
    return str(prediction or "").strip().lower() == str(target or "").strip().lower()


# -------------------------------------------------------------------------- VQAv2
# Official VQA answer-normalization tables (Antol et al. 2015, ``VQAEval``).
_VQA_MANUAL_MAP = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
_VQA_ARTICLES = {"a", "an", "the"}
_VQA_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_VQA_COMMA_STRIP = re.compile(r"(\d)(\,)(\d)")
_VQA_PUNCT = [
    ";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-",
    ">", "<", "@", "`", ",", "?", "!",
]
# Contraction normalization: map common contracted forms to a single spelling.
_VQA_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
    "hadnt": "hadn't", "hasnt": "hasn't", "havent": "haven't", "hes": "he's",
    "im": "i'm", "isnt": "isn't", "its": "it's", "lets": "let's", "maam": "ma'am",
    "mightve": "might've", "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't",
    "shant": "shan't", "shes": "she's", "shouldve": "should've", "shouldnt": "shouldn't",
    "somebodys": "somebody's", "someones": "someone's", "somethings": "something's",
    "thats": "that's", "theres": "there's", "theyd": "they'd", "theyre": "they're",
    "theyve": "they've", "wasnt": "wasn't", "weve": "we've", "werent": "weren't",
    "whatre": "what're", "whats": "what's", "wheres": "where's", "whos": "who's",
    "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "youd": "you'd",
    "youll": "you'll", "youre": "you're", "youve": "you've",
}


def _vqa_process_punctuation(text: str) -> str:
    out = text
    for punct in _VQA_PUNCT:
        spaced = (punct + " " in text) or (" " + punct in text)
        if spaced or (_VQA_COMMA_STRIP.search(text) is not None):
            out = out.replace(punct, "")
        else:
            out = out.replace(punct, " ")
    return _VQA_PERIOD_STRIP.sub("", out)


def _vqa_process_digit_article(text: str) -> str:
    words = []
    for word in text.lower().split():
        word = _VQA_MANUAL_MAP.get(word, word)
        if word not in _VQA_ARTICLES:
            words.append(word)
    for i, word in enumerate(words):
        if word in _VQA_CONTRACTIONS:
            words[i] = _VQA_CONTRACTIONS[word]
    return " ".join(words)


def vqa_normalize(answer: str) -> str:
    """Official VQA answer normalization (whitespace, punctuation, digits, articles)."""
    text = str(answer or "").replace("\n", " ").replace("\t", " ").strip()
    text = _vqa_process_punctuation(text)
    text = _vqa_process_digit_article(text)
    return text


def vqa_accuracy(prediction: str, ground_truths: list[str]) -> float:
    """VQA soft accuracy: ``min(1, agreement / 3)`` averaged leave-one-out over the
    (typically 10) human answers, after official normalization of both sides."""
    pred = vqa_normalize(prediction)
    gts = [vqa_normalize(gt) for gt in ground_truths]
    if not gts:
        return 0.0
    scores = []
    for i in range(len(gts)):
        others = gts[:i] + gts[i + 1 :]
        matches = sum(1 for gt in others if gt == pred)
        scores.append(min(1.0, matches / 3.0))
    return sum(scores) / len(scores)
