"""Loader for ``peterant330/saliency-r1-8k`` for the evidence-reliance probe.

Each sample carries a GT *evidence* bounding box -- the image region that
contains the evidence needed to answer the question. Stage 0 masks / crops that
region to test whether a model's accuracy causally depends on it.

Schema (HF ``peterant330/saliency-r1-8k``, single ``train`` split):
  - ``dataset``     (str)   source subset: ``"CUB"`` (bird yes/no attribute QA)
                            or ``"DocVQA"`` (document text-extraction QA).
  - ``split``       (str)
  - ``question_id`` (int)
  - ``problem``     (str)   the question.
  - ``solution``    (str)   ground-truth answer ("Yes"/"No" or extracted text).
  - ``bbox``        (str)   ``"[x1, y1, x2, y2]"`` NORMALIZED to [0, 1]
                            (left, top, right, bottom). One box per sample.
  - ``image``       (Image)
"""

from __future__ import annotations

import ast
import io
import json
import os
from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional

from PIL import Image

BoxNorm = tuple[float, float, float, float]


@dataclass
class SaliencySample:
    sample_id: str
    subset: str  # "CUB" / "DocVQA" / raw `dataset` value
    problem: str
    solution: str
    image: Image.Image  # RGB
    bbox_norm: BoxNorm  # (x1, y1, x2, y2) in [0, 1]


def _to_pil(value: Any) -> Image.Image:
    """Coerce a HF image field (PIL / {bytes|path} dict / path str) to RGB PIL."""
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes"):
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
    if isinstance(value, str) and value:
        return Image.open(value).convert("RGB")
    raise TypeError(f"Unsupported image field type for saliency-r1-8k: {type(value)!r}")


def parse_bbox_norm(raw: Any) -> Optional[BoxNorm]:
    """Parse the ``bbox`` field into a normalized ``(x1,y1,x2,y2)`` tuple or None.

    The field ships as a *string* like ``"[0.094, 0.312, 0.44, 0.795]"``. Returns
    None for missing / unparseable / zero-area boxes (no region to probe).
    """
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        vals: Any = list(raw)
    else:
        text = str(raw).strip()
        if not text:
            return None
        try:
            vals = json.loads(text)
        except Exception:
            try:
                vals = ast.literal_eval(text)
            except Exception:
                return None
    if not isinstance(vals, (list, tuple)) or len(vals) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in vals)
    except (TypeError, ValueError):
        return None
    # Order-normalize (some boxes may be stored as (x2,x1)) and clamp to [0, 1].
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1, y1, x2, y2 = (min(max(v, 0.0), 1.0) for v in (x1, y1, x2, y2))
    if (x2 - x1) <= 1e-3 or (y2 - y1) <= 1e-3:
        return None  # degenerate / zero-area
    return (x1, y1, x2, y2)


def _load_hf_split(dataset: str, split: str):
    """Load a split from a HF id OR a local dir (save_to_disk dump / dataset dir /
    raw parquet|arrow|json). Local paths work under HF_HUB_OFFLINE=1."""
    from datasets import load_dataset

    if not os.path.isdir(dataset):
        return load_dataset(dataset, split=split)

    # save_to_disk dump (has dataset_info.json / state.json).
    if os.path.exists(os.path.join(dataset, "dataset_info.json")) or os.path.exists(
        os.path.join(dataset, "state.json")
    ):
        from datasets import load_from_disk

        disk = load_from_disk(dataset)
        if hasattr(disk, "keys"):  # DatasetDict
            return disk[split] if split in disk else disk[next(iter(disk.keys()))]
        return disk

    # Standard local dataset dir (datasets auto-detects parquet/arrow under data/).
    try:
        return load_dataset(dataset, split=split)
    except Exception:
        import glob

        for ext, builder in (("parquet", "parquet"), ("arrow", "arrow"), ("jsonl", "json"), ("json", "json")):
            files = sorted(glob.glob(os.path.join(dataset, "**", f"*.{ext}"), recursive=True))
            if files:
                return load_dataset(builder, data_files=files, split="train")
        raise


def load_saliency_samples(
    dataset: str = "peterant330/saliency-r1-8k",
    split: str = "train",
    *,
    limit: int | None = None,
    subsets: list[str] | None = None,
) -> list[SaliencySample]:
    """Load probe samples (those with a valid evidence bbox).

    ``limit`` is a **per-subset** cap (so both CUB and DocVQA are covered in one
    run; CUB has only ~80 samples). ``subsets`` optionally restricts to e.g.
    ``["CUB"]`` / ``["DocVQA"]``. Selection is deterministic in dataset order, so
    two model runs with the same args see identical samples (required for the
    paired analysis).
    """
    data = _load_hf_split(dataset, split)
    subset_filter = {s.strip().lower() for s in subsets} if subsets else None
    counts: Counter[str] = Counter()
    samples: list[SaliencySample] = []
    skipped_bbox = 0
    skipped_field = 0
    for index in range(len(data)):
        record = data[index]
        subset = str(record.get("dataset", "")).strip() or "unknown"
        if subset_filter is not None and subset.lower() not in subset_filter:
            continue
        if limit is not None and counts[subset] >= limit:
            continue
        bbox = parse_bbox_norm(record.get("bbox"))
        if bbox is None:
            skipped_bbox += 1
            continue
        problem = str(record.get("problem", "")).strip()
        solution = str(record.get("solution", "")).strip()
        if not problem or not solution:
            skipped_field += 1
            continue
        image = _to_pil(record.get("image"))
        sample_id = str(record.get("question_id", index))
        samples.append(SaliencySample(sample_id, subset, problem, solution, image, bbox))
        counts[subset] += 1
    summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "(none)"
    print(
        f"[saliency] loaded {len(samples)} samples ({summary}); "
        f"skipped {skipped_bbox} bad-bbox, {skipped_field} missing-field"
    )
    return samples
