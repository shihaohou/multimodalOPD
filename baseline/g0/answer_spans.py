"""Answer-span selection for the G0 probes — the ``\\boxed{...}`` token rows.

The G0 verdict cares about *where the answer comes from*, so every probe (LH /
GLIMPSE / EAGLE) must be told **which generated tokens are "the answer"**. The
existing answer-span proxy is the last-K generated tokens; that is noisy when the
model keeps explaining after the boxed answer, or when the answer is long. This
module adds the precise span: the tokens **inside ``\\boxed{...}``**, with a
clean fallback to last-K when no boxed answer was emitted.

Everything here is tokenizer-only (no model), so it is CPU-unit-testable:
``python -m baseline.g0.answer_spans``.

Token-index convention (all in *completion* coordinates, i.e. relative to the
first generated token):

* ``boxed_token_span(ids, tok) → (start, end)`` half-open token range covering
  the characters inside the **last** ``\\boxed{...}`` of the decoded completion,
  or ``None`` if there is no boxed answer.
* :func:`resolve_answer_spans` bundles the boxed span, the last-K span, and which
  one is "primary" (boxed if present, else last-K) into :class:`AnswerSpans`.
* :func:`span_predictor_rows` / :func:`span_completion_mask` turn a completion
  span into the absolute attention **query rows** (LH) / the per-token
  confidence **mask** (GLIMPSE β) the probes consume.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import torch

CompletionSpan = tuple[int, int]  # half-open [start, end) in completion-token coords

# Matches the opening of a boxed answer: ``\boxed {`` / ``\boxed{`` (optional ws).
_BOXED_OPEN = re.compile(r"\\boxed\s*\{")


def _decode(tokenizer, ids: list[int]) -> str:
    """Decode raw (specials kept, no space-cleanup) so prefix offsets stay stable.

    ``decode(ids[:k])`` must be a prefix of ``decode(ids)`` for the char→token
    mapping to be valid; ``clean_up_tokenization_spaces=False`` and keeping
    special tokens preserves that for BPE tokenizers (Qwen included).
    """
    return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def _find_boxed_char_span(text: str) -> Optional[tuple[int, int]]:
    """Char range ``[i, j)`` of the content inside the LAST ``\\boxed{...}``.

    Brace-balanced (handles nested ``{}`` inside the answer). Returns ``None`` if
    there is no ``\\boxed{`` or the content is empty. An unbalanced/never-closed
    brace takes the content to the end of the string.
    """
    matches = list(_BOXED_OPEN.finditer(text))
    if not matches:
        return None
    i = matches[-1].end()  # first content char, just after '{'
    depth = 1
    j = i
    while j < len(text):
        ch = text[j]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return (i, j) if j > i else None
        j += 1
    return (i, len(text)) if len(text) > i else None


def _char_to_token(clen, char_pos: int, n: int) -> int:
    """Index of the completion token that contains ``char_pos``.

    ``clen(k) = len(decode(ids[:k]))`` is non-decreasing, so the token holding a
    char position is the smallest ``k`` with ``clen(k) > char_pos``, minus one.
    Binary search keeps this to ``O(log n)`` decode calls.
    """
    lo, hi = 1, n
    while lo < hi:
        mid = (lo + hi) // 2
        if clen(mid) > char_pos:
            hi = mid
        else:
            lo = mid + 1
    return max(0, lo - 1)


def _as_int_list(completion_ids) -> list[int]:
    if hasattr(completion_ids, "tolist"):
        return [int(x) for x in completion_ids.tolist()]
    return [int(x) for x in completion_ids]


def boxed_token_span(completion_ids, tokenizer) -> Optional[CompletionSpan]:
    """Half-open token range of the ``\\boxed{...}`` content, or ``None``.

    ``completion_ids`` is the GENERATED tokens only (1-D tensor or list). The
    returned ``(start, end)`` are in completion-token coordinates.
    """
    ids = _as_int_list(completion_ids)
    n = len(ids)
    if n == 0:
        return None
    text = _decode(tokenizer, ids)
    span = _find_boxed_char_span(text)
    if span is None:
        return None
    c0, c1 = span  # content chars [c0, c1)

    cache: dict[int, int] = {0: 0, n: len(text)}

    def clen(k: int) -> int:
        if k not in cache:
            cache[k] = len(_decode(tokenizer, ids[:k]))
        return cache[k]

    start_tok = _char_to_token(clen, c0, n)
    end_tok = _char_to_token(clen, max(c0, c1 - 1), n)  # token holding last content char
    s = max(0, min(start_tok, n - 1))
    e = max(s + 1, min(end_tok + 1, n))  # inclusive end → half-open
    return (s, e)


@dataclass
class AnswerSpans:
    """The answer spans for one completion (token coords; half-open)."""

    comp_len: int
    lastk: CompletionSpan
    boxed: Optional[CompletionSpan]
    primary: CompletionSpan  # boxed if present, else lastk
    mode: str  # "boxed" | "lastk"

    @property
    def has_boxed(self) -> bool:
        return self.boxed is not None


def resolve_answer_spans(completion_ids, tokenizer, answer_k: int = 16) -> AnswerSpans:
    """Boxed span (+ last-K fallback) for a completion.

    ``primary`` is the boxed span when the completion has a ``\\boxed{...}``,
    otherwise the last ``answer_k`` generated tokens — the span the probes treat
    as "the answer" for ``*_boxed`` metrics.
    """
    comp_len = len(_as_int_list(completion_ids))
    k = max(1, min(int(answer_k), max(1, comp_len)))
    lastk = (max(0, comp_len - k), comp_len)
    boxed = boxed_token_span(completion_ids, tokenizer) if comp_len else None
    if boxed is not None:
        return AnswerSpans(comp_len, lastk, boxed, boxed, "boxed")
    return AnswerSpans(comp_len, lastk, None, lastk, "lastk")


def span_predictor_rows(prompt_len: int, span: CompletionSpan, device=None) -> torch.Tensor:
    """Absolute attention query rows that PREDICT a completion span.

    Completion token ``j`` sits at absolute position ``prompt_len + j`` and is
    predicted by the previous row ``prompt_len + j - 1``. Rows are clamped to
    ``>= prompt_len - 1`` (the last prompt row), matching the LH answer-span
    convention in :mod:`run_g0`.
    """
    s, e = span
    lo = prompt_len + s - 1
    hi = prompt_len + e - 1
    rows = torch.arange(lo, hi, device=device)
    return rows.clamp_min(prompt_len - 1)


def span_completion_mask(comp_len: int, span: CompletionSpan, device=None) -> torch.Tensor:
    """Float ``[comp_len]`` mask, 1.0 on the span — for GLIMPSE β reweighting."""
    s, e = span
    m = torch.zeros(int(comp_len), device=device)
    s = max(0, min(s, comp_len))
    e = max(s, min(e, comp_len))
    if e > s:
        m[s:e] = 1.0
    return m


# --------------------------------------------------------------------- self-test
class _FakeTokenizer:
    """Decodes ids→pieces by lookup; ``decode(ids[:k])`` is an exact prefix."""

    def __init__(self, pieces: list[str]):
        self.pieces = pieces  # piece[id] = its string

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "".join(self.pieces[i] for i in ids)


def _selftest() -> None:
    # completion: "Reasoning here. \boxed{cat} done"
    pieces = ["Reasoning", " here.", " \\boxed{", "ca", "t", "}", " done"]
    ids = list(range(len(pieces)))
    tok = _FakeTokenizer(pieces)
    text = tok.decode(ids)
    assert "\\boxed{cat}" in text, text
    span = boxed_token_span(ids, tok)
    assert span is not None
    s, e = span
    # the content "cat" is split across pieces 3 ("ca") and 4 ("t")
    assert (s, e) == (3, 5), (s, e, [tok.decode([i]) for i in ids])
    # decoding exactly the span reconstructs the answer content.
    assert tok.decode(ids[s:e]) == "cat"

    # nested braces inside the answer.
    pieces2 = ["x ", "\\boxed{", "f(", "{a}", ")", "}", " y"]
    ids2 = list(range(len(pieces2)))
    tok2 = _FakeTokenizer(pieces2)
    s2, e2 = boxed_token_span(ids2, tok2)
    assert tok2.decode(ids2[s2:e2]) == "f({a})", tok2.decode(ids2[s2:e2])

    # no boxed → None, and resolve falls back to last-K.
    pieces3 = ["just", " some", " text"]
    ids3 = list(range(len(pieces3)))
    tok3 = _FakeTokenizer(pieces3)
    assert boxed_token_span(ids3, tok3) is None
    spans = resolve_answer_spans(ids3, tok3, answer_k=2)
    assert spans.mode == "lastk" and spans.primary == (1, 3), spans
    spans_b = resolve_answer_spans(ids, tok, answer_k=4)
    assert spans_b.mode == "boxed" and spans_b.primary == (3, 5), spans_b

    # predictor rows + completion mask.
    rows = span_predictor_rows(prompt_len=10, span=(3, 5))
    assert rows.tolist() == [12, 13], rows.tolist()  # predicts comp tokens 3,4
    m = span_completion_mask(6, (3, 5))
    assert m.tolist() == [0, 0, 0, 1, 1, 0], m.tolist()
    print("[g0.answer_spans] self-test passed.")


if __name__ == "__main__":
    _selftest()
