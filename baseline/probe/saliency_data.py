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
    image_source: Optional[str] = None


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


def _image_source(value: Any) -> Optional[str]:
    """Best-effort source path for an image field, if the dataset exposes one."""
    if isinstance(value, dict) and value.get("path"):
        return str(value["path"])
    if isinstance(value, str) and value:
        return value
    return None


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


def _avoid_eager_image_decode(data):
    """Keep image rows as bytes/paths until a sample is actually selected.

    This matters for global-limit sharding: workers may scan metadata for rows that
    belong to another shard, and decoding every skipped image is pure overhead.
    """
    try:
        from datasets import Image as HFImage

        features = getattr(data, "features", {}) or {}
        if "image" in features and isinstance(features["image"], HFImage):
            return data.cast_column("image", HFImage(decode=False))
    except Exception:
        pass
    return data


def bbox_area(bbox: BoxNorm) -> float:
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)


# Subset-name aliases: saliency-r1-8k stores the *Visual-CoT source* in `dataset`
# under short names (e.g. ``v7w``, ``openimages``), but callers naturally type the
# long forms (``visual7w``, ``open_images``). We canonicalize BOTH the requested
# names and each row's `dataset` for filter-matching only — ``SaliencySample.subset``
# keeps the raw value, so records/analysis still group by the real dataset name.
_SUBSET_ALIASES = {
    "visual7w": "v7w",
    "visualgenome7w": "v7w",
    "v7w": "v7w",
    "openimages": "openimages",
    "flickr": "flickr30k",
    "flickr30k": "flickr30k",
}


def canon_subset(name: str) -> str:
    """Lowercase, drop separators, map known aliases → canonical (for matching)."""
    n = "".join(ch for ch in str(name).strip().lower() if ch.isalnum())
    return _SUBSET_ALIASES.get(n, n)


def load_saliency_samples(
    dataset: str = "peterant330/saliency-r1-8k",
    split: str = "train",
    *,
    limit: int | None = None,
    subsets: list[str] | None = None,
    max_bbox_area: float | None = None,
    min_bbox_area: float | None = None,
    num_shards: int | None = None,
    shard_index: int = 0,
    limit_before_shard: bool = False,
) -> list[SaliencySample]:
    """Load probe samples (those with a valid evidence bbox).

    ``limit`` is a **per-subset** cap (so every subset is covered in one run; some
    have only ~70-80 samples). ``subsets`` optionally restricts to a set of subset
    names (case-insensitive, e.g. ``["textvqa", "docvqa"]``). ``max_bbox_area`` /
    ``min_bbox_area`` filter by evidence-box area (fraction of image) -- set
    ``max_bbox_area`` (e.g. 0.5) to drop near-whole-image boxes where the
    equal-area random-mask control can't be placed disjointly (which would dilute
    Reliance toward 0). Selection is deterministic in dataset order, so two model
    runs with the same args see identical samples (required for the paired analysis).

    ``num_shards`` / ``shard_index`` stride the **raw dataset rows** (``index %
    num_shards == shard_index``) BEFORE the image is decoded, so N data-parallel
    workers each touch only 1/N of the images (no N× decode / RAM blow-up). With
    ``limit`` set, the per-subset cap normally applies *within each shard*.

    Set ``limit_before_shard=True`` to apply that cap globally first, then shard
    the capped sample list. This is useful when launching multiple workers on the
    same GPU: more shards increase parallelism without increasing total eval size.
    """
    data = _avoid_eager_image_decode(_load_hf_split(dataset, split))
    subset_filter = {canon_subset(s) for s in subsets} if subsets else None
    sharded = bool(num_shards and num_shards > 1)
    global_cap = bool(sharded and limit_before_shard and limit is not None)
    global_counts: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    samples: list[SaliencySample] = []
    skipped_bbox = 0
    skipped_field = 0
    skipped_area = 0
    for index in range(len(data)):
        if sharded and not global_cap and (index % num_shards) != shard_index:
            continue
        record = data[index]
        subset = str(record.get("dataset", "")).strip() or "unknown"
        if subset_filter is not None and canon_subset(subset) not in subset_filter:
            continue
        bbox = parse_bbox_norm(record.get("bbox"))
        if bbox is None:
            skipped_bbox += 1
            continue
        area = bbox_area(bbox)
        if (max_bbox_area is not None and area > max_bbox_area) or (
            min_bbox_area is not None and area < min_bbox_area
        ):
            skipped_area += 1
            continue
        problem = str(record.get("problem", "")).strip()
        solution = str(record.get("solution", "")).strip()
        if not problem or not solution:
            skipped_field += 1
            continue
        if global_cap:
            if global_counts[subset] >= limit:
                continue
            global_counts[subset] += 1
            if (index % num_shards) != shard_index:
                continue
        elif limit is not None and counts[subset] >= limit:
            continue
        image_field = record.get("image")
        image = _to_pil(image_field)
        sample_id = str(record.get("question_id", index))
        samples.append(SaliencySample(sample_id, subset, problem, solution, image, bbox, _image_source(image_field)))
        counts[subset] += 1
    summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "(none)"
    shard_tag = f" [shard {shard_index}/{num_shards}]" if sharded else ""
    print(
        f"[saliency]{shard_tag} loaded {len(samples)} samples ({summary}); skipped "
        f"{skipped_bbox} bad-bbox, {skipped_area} out-of-area, {skipped_field} missing-field"
    )
    if subset_filter is not None and not sharded:  # flag typos (only on the unsharded view)
        seen = {canon_subset(k) for k in counts}
        missing = sorted(subset_filter - seen)
        if missing:
            print(f"[saliency] WARNING: requested subset(s) matched 0 rows: {missing} "
                  f"(check spelling; aliases handled: visual7w→v7w, flickr→flickr30k)")
    return samples
