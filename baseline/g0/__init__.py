"""G0 — grounding diagnostic harness (looking-vs-using).

The single question this package answers: when a Qwen-VL student gets a
vision-grounded question wrong, is it a **looking failure** (its attention /
localization points at the wrong region) or a **using failure** (it localizes
the right region but the answer does not draw on it)?  The answer decides whether
any attention/map-level intervention can help (it cannot if failures are
"using") and explains why the TAM / Saliency-R1 map losses did not move accuracy.

Two faithful, complementary probes — kept deliberately separate:

* :mod:`baseline.g0.localization_heads` — "**looking**". A Qwen port of the
  LocalizationHeads method (paper: *Your LVLM Only Needs A Few Attention Heads
  For Visual Grounding*). Reads the model's first-generation-step attention over
  the image patch grid, calibrates per-head IoU against GT boxes to discover the
  localization heads (separately for the 8B teacher and the 2B student — never
  assume LLaVA's L14-H24), assembles the top-k heads into a predicted box and
  scores IoU vs the GT evidence box.
* :mod:`baseline.g0.glimpse` — "**using**". A tractable, one-backward
  re-implementation of GLIMPSE (arXiv 2506.18985): gradient x attention,
  response-level, faithful. Produces a visual saliency map (→ IoU vs GT,
  energy-in-bbox, pointing-game) and a ``vt_ratio`` = visual mass / (visual +
  textual prompt mass), i.e. how image-driven the answer actually is.

:mod:`baseline.g0.engine` is the shared model/data plumbing; :mod:`run_g0`
produces per-sample records over the three conditions (C1 teacher, C2 teacher+
hidden-hint, C3 student); :mod:`analyze_g0` turns those into the four analyses,
2x2 tables and figures.
"""
