"""Locate-Once Grounding (LOG): hidden-hint OPD + a student box RL term.

The verified hidden-hint distillation spine (``baseline.hint``) plus an explicit,
*student-generated* evidence box trained by RL. The student is prompted to open its
``<think>`` with a single ``<box>[x1,y1,x2,y2]</box>`` (no crop, no zoom — the box
does not change the pixels it sees), then reason and answer. Two span-decoupled
gradient sources:

* **OPD** (``KL(student||teacher)``) on the answer/reasoning tokens, with the box
  span MASKED out — teaches *how to answer as if you knew where to look* by pulling
  the un-hinted student toward the box-privileged (hidden-hint) teacher.
* **RL** (GRPO, group-normalized IoU reward gated by answer correctness) on the box
  coordinate tokens — teaches *where to look*.

See ``baseline/locate/README.md``.
"""
