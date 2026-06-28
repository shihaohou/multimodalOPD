"""Locate-Once prompt constants — deliberately import-light (no training stack).

Kept separate from ``opd_locate_collator`` (which pulls in the full HF/accelerate
trainer chain) so the CPU-only sanity checks and any prompt inspection can import the
strings without a GPU environment.

The method's structure (paper thesis "Seeing Before Reasoning"): the CoT is three
steps — **locate** (state the region once as ``<box>``), **describe** (what is in it),
**reason** (from that description to the answer). Student and OPD teacher share this
structure so their thinking patterns stay in the same distribution (Rethinking-OPD).
The teacher is additionally *given* the GT box (it states it); the student finds it.
"""

from __future__ import annotations

# Student (and cold-start generator) system prompt: structured locate -> describe ->
# reason, one <box> at the head, [0,1] coords, \boxed{} answer.
LOCATE_SYSTEM_PROMPT = (
    "You are a helpful assistant. Reason inside <think> </think> in three steps: "
    "(1) state the single most relevant region for the question, once, as "
    "<box>[x1, y1, x2, y2]</box> (coordinates normalized to [0,1], top-left origin); "
    "(2) describe what is in that region; (3) reason from that description to the answer. "
    "Then give the final answer in \\boxed{}."
)

# OPD teacher hint (appended to the teacher's question; the teacher is privileged with
# the GT box). It STATES the box (no no-verbalize clause — post-cold-start the student
# emits boxes, so a box-stating teacher matches its pattern; the de1e4c5 collapse needed
# a box-FREE student). The box coordinate span is still masked from the OPD loss in the
# trainer (Option 3: OPD distills how-to-think; RL owns where-to-look). ``{bbox}`` is the
# per-sample normalized box.
LOCATE_TEACHER_HINT_TEMPLATE = (
    "Hint: the region that contains the answer is <box>{bbox}</box> (normalized to [0,1], "
    "top-left origin, [x1, y1, x2, y2]). Inside <think>, state this region once as "
    "<box>{bbox}</box>, then describe what is in it, then reason to the answer. Refer to "
    "it as \"that region\" afterwards; do not repeat the coordinates."
)
