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

# The cold-start trace GENERATION prompt (single source of truth, imported by
# coldstart_build's natural mode AND the OPD teacher's `gen` mode). In `natural` cold-start
# the teacher is shown the GT box and writes the whole structured trace under this prompt;
# the student is then SFT'd to reproduce those outputs from the box-FREE student prompt.
# So scoring the OPD teacher under this SAME prompt makes teacher(gen) ~= student(SFT'd) —
# the tightest distribution match (Rethinking-OPD) — while the teacher stays grounded (it
# sees the box). Safe for OPD because the box COORD span (+ its decision token) is masked
# from the loss and the template orders "do not repeat the coordinates", so no coordinate
# digits leak into the supervised describe/reason span (the old Option-3 salad was a
# weak-cold-start artifact, not this template). ``{bbox}`` = per-sample normalized box.
NATURAL_GEN_TEMPLATE = (
    "Hint: the region that contains the answer is <box>{bbox}</box> (coordinates normalized "
    "to [0,1], top-left origin, [x1, y1, x2, y2]). Inside <think>, follow three steps: "
    "(1) state this region once as <box>{bbox}</box>; (2) describe what is in that region "
    "(the visual details relevant to the question); (3) reason from that description to the "
    "answer. Refer to it as \"that region\" afterwards and do not repeat the coordinates. "
    "Then give the final answer in \\boxed{{}}."
)
