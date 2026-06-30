"""Vendored minimal core of EAGLE (CVPR 2026) for the G0 faithful-attribution probe.

EAGLE — *Where MLLMs Attend and What They Rely On: Explaining Autoregressive
Token Generation* (Ruoyu Chen et al., arXiv 2509.22496;
https://github.com/RuoyuChen10/EAGLE, MIT License). Black-box visual attribution
by **submodular subset selection** over image sub-regions: greedily order the
regions by an insertion (sufficiency) + deletion (necessity) objective, reading
the target token's probability under each perturbed image.

We vendor only the three pieces the G0 probe needs so the diagnostic is
self-contained on the GPU box (no external EAGLE checkout / opencv-contrib
requirement):

* :mod:`submodular_vision` — the base ``MLLMSubModularExplanationVision`` (copied
  verbatim; algorithm unchanged).
* :mod:`efficient_attribution_v2` — the batched ``...V2`` explainer (copied
  verbatim; the one the modern EAGLE pipeline uses).
* :mod:`regions` — ``sub_region_division`` (SLICO → skimage SLIC → numpy-grid
  fallback) and ``add_value`` (region order → attribution heatmap), so region
  division works even without ``cv2.ximgproc``.

The model wrapper (the EAGLE "adaptor", which scores a perturbed image) lives in
:mod:`baseline.g0.eagle_probe`, where it reuses our own Qwen prompt builder so
the explained distribution matches the G0 rollout.
"""

from baseline.g0.eagle_src.efficient_attribution_v2 import (
    EfficientMLLMSubModularExplanationVisionV2,
)
from baseline.g0.eagle_src.regions import add_value, sub_region_division

__all__ = [
    "EfficientMLLMSubModularExplanationVisionV2",
    "sub_region_division",
    "add_value",
]
