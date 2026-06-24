"""Parse an OPD completion into reason-span and answer-span token ranges.

The saliency engine's two-hop routing (answer -> reason -> visual) needs, per
rollout, the token positions of the **reason** span (``<reason>...</reason>``)
and the **answer** span (the ``\\boxed{}`` answer after ``</reason>``). This is
the OPD analogue of Saliency_R1's ``<think>``/answer span parsing
(``grpo_trainer.py`` ~1749-1775), adapted to the OPD unified prompt
(:data:`baseline.opd_data_collator.OPD_SYSTEM_PROMPT`).

Robustness choice: rather than re-encode the decoded text (whose token
boundaries can drift from the rollout's actual ids), we build a char->token
offset map by decoding the **actual** completion ids token-by-token. The span
markers (``<reason>``, ``</reason>``, ``<|im_end|>``) are ASCII and sit on token
boundaries, so marker offsets are exact even if multi-byte content tokens decode
imperfectly in isolation. Malformed rows (no well-formed reason+answer) are
flagged ``valid=False`` and simply skipped by the evidence loss (they still get
the OPD token loss).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

REASON_OPEN = "<reason>"
REASON_CLOSE = "</reason>"
TURN_END = "<|im_end|>"


@dataclass
class CompletionSpans:
    """Token-index spans **within the completion** (0-based, end-inclusive)."""

    reason: tuple[int, int] | None
    answer: tuple[int, int] | None
    valid: bool
    text: str


def _strip_ws(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def find_char_spans(
    text: str,
    *,
    reason_open: str = REASON_OPEN,
    reason_close: str = REASON_CLOSE,
    turn_end: str = TURN_END,
) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    """(reason_char_span, answer_char_span); either may be None."""
    ro = text.find(reason_open)
    rc = text.find(reason_close, ro + len(reason_open) if ro != -1 else 0)
    reason: tuple[int, int] | None = None
    answer: tuple[int, int] | None = None

    if ro != -1 and rc != -1 and rc > ro:
        rs, re_ = _strip_ws(text, ro + len(reason_open), rc)
        if re_ > rs:
            reason = (rs, re_)
        a_start = rc + len(reason_close)
        a_end = len(text)
        te = text.find(turn_end, a_start)
        if te != -1:
            a_end = te
        a_start, a_end = _strip_ws(text, a_start, a_end)
        if a_end > a_start:
            answer = (a_start, a_end)
    return reason, answer


def _offsets_from_ids(tokenizer: Any, ids: list[int]) -> tuple[str, list[tuple[int, int]]]:
    """Decode ids token-by-token, returning the joined text and per-token
    ``(char_start, char_end)`` offsets aligned to ``ids`` (no re-encoding)."""
    text_parts: list[str] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for tid in ids:
        piece = tokenizer.decode(
            [int(tid)], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        start = cursor
        cursor += len(piece)
        text_parts.append(piece)
        offsets.append((start, cursor))
    return "".join(text_parts), offsets


def _char_span_to_token_range(
    offsets: list[tuple[int, int]], char_start: int, char_end: int
) -> tuple[int, int] | None:
    """First/last token indices (end-inclusive) overlapping ``[char_start, char_end)``."""
    hits = [
        i
        for i, (s, e) in enumerate(offsets)
        if e > char_start and s < char_end and e > s
    ]
    if not hits:
        return None
    return hits[0], hits[-1]


def parse_completion_spans(tokenizer: Any, ids: list[int]) -> CompletionSpans:
    """Reason/answer token spans for one (unpadded) completion id sequence."""
    text, offsets = _offsets_from_ids(tokenizer, ids)
    reason_chars, answer_chars = find_char_spans(text)
    reason_tok = (
        _char_span_to_token_range(offsets, *reason_chars) if reason_chars else None
    )
    answer_tok = (
        _char_span_to_token_range(offsets, *answer_chars) if answer_chars else None
    )
    valid = (
        reason_tok is not None
        and answer_tok is not None
        and answer_tok[0] > reason_tok[1]  # answer strictly after the reason span
    )
    return CompletionSpans(reason=reason_tok, answer=answer_tok, valid=valid, text=text)


def parse_batch_spans(
    tokenizer: Any,
    completion_ids,  # [B, C] LongTensor
    completion_mask,  # [B, C] bool/int — True for real (non-pad) tokens
) -> list[CompletionSpans]:
    """Per-sample spans for a padded completion batch (pad tokens trimmed)."""
    out: list[CompletionSpans] = []
    for b in range(completion_ids.shape[0]):
        mask_row = completion_mask[b].to(dtype=bool)
        ids_row = completion_ids[b][mask_row].tolist()
        out.append(parse_completion_spans(tokenizer, ids_row))
    return out
