"""Locate-Once prompt constants — deliberately import-light (no training stack).

Kept separate from ``opd_locate_collator`` (which pulls in the full HF/accelerate
trainer chain) so the CPU-only sanity checks and any prompt inspection can import the
strings without a GPU environment.
"""

from __future__ import annotations

# Locate-once student prompt. Mirrors OPD_SYSTEM_PROMPT (<think></think> + \boxed{})
# but instructs the student to OPEN its reasoning with one normalized <box>...</box>.
# The box is a *commitment to where to look*, not a crop: the model still sees the
# full image at the same resolution. Coordinate convention is spelled out so a
# format-following VLM emits a parseable box that is directly IoU-comparable to GT.
LOCATE_SYSTEM_PROMPT = (
    "You are a helpful assistant. Begin your reasoning inside <think> </think> by "
    "locating the single image region most relevant to the question, stated once as "
    "<box>[x1, y1, x2, y2]</box> (coordinates normalized to [0,1], top-left origin, "
    "[x1, y1, x2, y2]). Then reason about that region and give the final answer in "
    "\\boxed{}."
)
