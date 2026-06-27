"""OPD dataset loading: ViRL39K-style local-parquet adapter + HF passthrough.

Vanilla OPD/ViGOS datasets load via :func:`vigos.dataset_utils.load_vigos_dataset`
(a HuggingFace id whose rows already expose ``problem`` / ``images`` / ``answer``).
**ViRL39K** (``TIGER-Lab/ViRL39K``) ships differently and trips that path:

* the QA lives in a single top-level parquet (``39Krelease.parquet``) with columns
  ``question`` / ``answer`` / ``qid`` / ... and an ``image`` column that is a
  ``Sequence(string)`` of **relative paths** (e.g. ``images/<qid>-0.jpg``) into a
  sibling ``images/`` dir — the images are **not** embedded;
* the ``question`` text is prefixed with a literal ``<image>`` placeholder token;
* pointing ``load_dataset`` at that directory triggers the **imagefolder** builder,
  which scans ``images/`` *and* the extracted ``images.zip`` (~2x the image files)
  and yields a single ``image`` column with **no text at all**.

:func:`load_opd_dataset` loads the parquet directly (no imagefolder) and adapts the
ViRL39K schema to the canonical OPD schema the collator expects:

* ``question`` -> ``problem`` with the ``<image>`` placeholder token stripped;
* ``image`` (list of relative paths) -> the **first** image, resolved to an
  absolute path and cast to :class:`datasets.Image` (so the tiny-image filter and
  the collator receive a decoded ``PIL`` image, not a bare path string that the
  size filter would drop);
* ``answer`` kept as-is (already ``\\boxed{}``-formatted).

Multi-image questions keep only their FIRST image (the OPD collator is
single-image); the count is logged so the drop is never silent. Anything that is
not a local ViRL39K-style parquet (a HuggingFace id, or a normal dataset dir whose
rows already have ``problem``) falls through to
:func:`vigos.dataset_utils.load_vigos_dataset` unchanged.

Lives in ``baseline/`` (not ``vigos/``) per the project's "ViGOS untouched" rule;
heavy imports (``datasets``) are lazy so the pure path/text helpers stay importable
without the training stack.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

# ViRL39K prefixes the question with a literal "<image>" placeholder (+ newline).
# The OPD collator adds the image as its own chat-template content item, so the
# literal token must be stripped or it lands in the prompt as garbage text.
_IMAGE_TOKEN_RE = re.compile(r"<image>\s*")


def _strip_image_tokens(text: Any) -> str:
    """Drop literal ``<image>`` placeholder tokens (and trailing space) from text."""
    return _IMAGE_TOKEN_RE.sub("", str(text)).strip()


def _local_parquet_files(dataset_name: str) -> tuple[list[str] | None, str | None]:
    """``(parquet_files, base_dir)`` for a ViRL39K-style local source, else ``(None, None)``.

    Matches a ``.parquet`` file (base = its parent) or a directory holding one or
    more **top-level** ``*.parquet`` (base = the dir). A HuggingFace id, or a dir
    whose parquet lives under ``data/`` (the standard HF layout that
    ``load_dataset`` already handles), returns ``(None, None)`` and falls through.
    """
    p = Path(dataset_name)
    if p.is_file() and p.suffix == ".parquet":
        return [str(p)], str(p.parent)
    if p.is_dir():
        pqs = sorted(str(x) for x in p.glob("*.parquet"))
        if pqs:
            return pqs, str(p)
    return None, None


def _resolve_image(value: Any, base_dir: str) -> str | None:
    """First image path -> absolute. Accepts a path string or a list of paths."""
    if isinstance(value, (list, tuple)):
        value = value[0] if len(value) else None
    if value is None:
        return None
    value = str(value)
    return value if os.path.isabs(value) else os.path.join(base_dir, value)


def _virl_columns(columns: list[str]) -> tuple[str | None, str | None]:
    """``(question_col, image_col)`` if the schema looks like ViRL39K, else ``(None, …)``.

    ViRL-like := has a ``question`` column, does **not** already have ``problem``
    (so we never clobber a canonical dataset), and has an ``image``/``images``
    column.
    """
    cols = set(columns)
    if "question" not in cols or "problem" in cols:
        return None, None
    image_col = "image" if "image" in cols else ("images" if "images" in cols else None)
    return "question", image_col


def load_opd_dataset(dataset_name: str, split: str = "train", *, verbose: bool = True):
    """Load an OPD training set, adapting ViRL39K-style local parquet on the way.

    A local ViRL39K parquet/dir is loaded via the parquet builder (avoiding the
    imagefolder trap) and remapped to ``problem`` / ``image`` / ``answer``. A local
    **Visual-CoT** dir (per-domain ``metadata/*.jsonl`` with ``bboxs``/``image``) is
    remapped to ``problem`` / ``image`` / ``answer`` / ``bbox`` (pixel box normalized
    to [0,1] — usable by the grounding-hint collator). Every other ``dataset_name``
    (HuggingFace id, canonical local dataset) defers to
    :func:`vigos.dataset_utils.load_vigos_dataset`.
    """
    if _looks_like_viscot(dataset_name):
        return load_viscot_dataset(dataset_name, split, verbose=verbose)

    parquet_files, base_dir = _local_parquet_files(dataset_name)
    if parquet_files is None:
        from vigos.dataset_utils import load_vigos_dataset

        return load_vigos_dataset(dataset_name, split)

    from datasets import load_dataset

    dataset = load_dataset("parquet", data_files=parquet_files, split=split)
    question_col, image_col = _virl_columns(dataset.column_names)
    if question_col is None or image_col is None:
        # A local parquet that already has the canonical schema (or no images) —
        # hand it back untouched.
        return dataset
    return _adapt_virl39k(
        dataset, base_dir, question_col=question_col, image_col=image_col, verbose=verbose
    )


def _adapt_virl39k(
    dataset,
    base_dir: str,
    *,
    question_col: str,
    image_col: str,
    answer_col: str = "answer",
    verbose: bool = True,
):
    """Remap a ViRL39K-schema dataset to canonical ``problem`` / ``image`` / ``answer``."""
    from datasets import Image

    has_answer = answer_col in dataset.column_names
    n_multi = sum(
        1 for v in dataset[image_col] if isinstance(v, (list, tuple)) and len(v) > 1
    )

    def _map(example: dict[str, Any]) -> dict[str, Any]:
        row = {
            "problem": _strip_image_tokens(example[question_col]),
            "image": _resolve_image(example[image_col], base_dir),
        }
        if has_answer:
            row["answer"] = example[answer_col]
        return row

    dataset = dataset.map(
        _map,
        remove_columns=dataset.column_names,
        desc="Adapt ViRL39K -> OPD schema",
    )
    dataset = dataset.cast_column("image", Image())
    if verbose:
        print(
            f"[opd-dataset] ViRL39K-style parquet adapted: {len(dataset)} rows -> "
            f"columns={dataset.column_names}; {n_multi} multi-image questions kept "
            f"FIRST image only; image paths resolved under {base_dir}.",
            flush=True,
        )
    return dataset


# --- Visual-CoT (deepcs233/Visual-CoT) --------------------------------------------
# Per-domain ``metadata/*.jsonl`` with COMMON columns question / answer / bboxs /
# dataset / image / width / height (other columns — reasoning/thought/full_answer/
# possible_answers/multiple_choices — vary across files, so we load each file and
# keep only the shared core). bboxs are NESTED PIXEL boxes ``[[x1,y1,x2,y2]]``;
# image is a bare basename; the actual files live under the extracted image root.
_VISCOT_CORE_COLS = ("question", "answer", "bboxs", "image", "width", "height")
_VISCOT_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")


def _viscot_jsonl_files(path: str) -> list[str]:
    """All Visual-CoT per-domain JSONLs (the canonical ``metadata/`` 363k set)."""
    import glob

    return sorted(glob.glob(os.path.join(path, "metadata", "*.jsonl")))


def _looks_like_viscot(dataset_name: str) -> bool:
    """A local Visual-CoT dir = has ``metadata/*.jsonl`` or the ``viscot_363k.json``."""
    if not isinstance(dataset_name, str) or not os.path.isdir(dataset_name):
        return False
    if os.path.exists(os.path.join(dataset_name, "viscot_363k.json")):
        return True
    return bool(_viscot_jsonl_files(dataset_name))


def _normalize_viscot_bbox(boxes: Any, width: Any, height: Any) -> str:
    """First pixel box ``[[x1,y1,x2,y2]]`` -> normalized ``"[x1, y1, x2, y2]"`` string
    in [0,1] (the saliency-r1-8k bbox form parse_bbox_norm expects). ``""`` if the
    box / dims are missing or degenerate (the collator then falls back to no hint)."""
    try:
        w, h = float(width), float(height)
    except (TypeError, ValueError):
        return ""
    if w <= 0 or h <= 0 or not boxes:
        return ""
    box = boxes[0] if isinstance(boxes[0], (list, tuple)) else boxes
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return ""
    try:
        x1, y1, x2, y2 = (float(v) for v in box)
    except (TypeError, ValueError):
        return ""
    x1, x2 = sorted((x1 / w, x2 / w))
    y1, y2 = sorted((y1 / h, y2 / h))
    clamp = lambda v: min(max(v, 0.0), 1.0)  # noqa: E731
    x1, y1, x2, y2 = (clamp(v) for v in (x1, y1, x2, y2))
    if (x2 - x1) <= 1e-3 or (y2 - y1) <= 1e-3:
        return ""
    return f"[{x1:.4f}, {y1:.4f}, {x2:.4f}, {y2:.4f}]"


def _build_viscot_image_index(root: str, *, verbose: bool = True) -> dict[str, str]:
    """``{basename: abspath}`` over the extracted image root (one os.walk, cached to
    ``<root>/.viscot_basename_index.json``). Layout-agnostic: Visual-CoT extracts the
    image tars into per-source subdirs, but the JSONL ``image`` is a bare basename, so
    a basename index resolves it regardless of the exact subdir structure."""
    import json

    cache_path = os.path.join(root, ".viscot_basename_index.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as handle:
                index = json.load(handle)
            if verbose:
                print(f"[viscot] loaded image index cache ({len(index)} basenames).")
            return index
        except Exception:  # noqa: BLE001 - rebuild on any cache read failure
            pass
    index: dict[str, str] = {}
    total = 0
    for dirpath, _dirs, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(_VISCOT_IMAGE_EXTS):
                total += 1
                index.setdefault(name, os.path.join(dirpath, name))
    if verbose:
        print(
            f"[viscot] indexed {total} image files ({len(index)} unique basenames) "
            f"under {root}.",
            flush=True,
        )
    # Atomic write (temp + os.replace): on shared storage several DDP ranks / both
    # machines may build the index at once; a reader must see the old or the complete
    # new file, never a partially written one.
    try:
        import tempfile

        fd, tmp = tempfile.mkstemp(dir=root, prefix=".viscot_idx_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(index, handle)
            os.replace(tmp, cache_path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except Exception:  # noqa: BLE001 - cache is best-effort
        pass
    return index


def _resolve_viscot_image(
    image_value: Any, dataset_name: Any, root: str, index: dict[str, str]
) -> str | None:
    """Resolve a Visual-CoT ``image`` to an absolute path: basename index first (no
    stat), then the ``<root>/<dataset>/...`` source-subdir guesses as a fallback."""
    if isinstance(image_value, (list, tuple)):
        image_value = image_value[0] if image_value else None
    if not isinstance(image_value, str) or not image_value:
        return None
    base = os.path.basename(image_value)
    hit = index.get(base) or index.get(image_value)
    if hit is not None:
        return hit
    candidates = [os.path.join(root, image_value)]
    if isinstance(dataset_name, str) and dataset_name:
        candidates += [
            os.path.join(root, dataset_name, image_value),
            os.path.join(root, dataset_name, base),
            os.path.join(root, dataset_name, "images", base),
        ]
    return next((c for c in candidates if os.path.exists(c)), None)


def load_viscot_dataset(
    path: str,
    split: str = "train",
    *,
    image_root: str | None = None,
    verbose: bool = True,
):
    """Load Visual-CoT into the canonical OPD schema ``problem`` / ``image`` (PIL) /
    ``answer`` / ``bbox`` (pixel box normalized to [0,1]).

    The per-domain ``metadata/*.jsonl`` (the canonical 363k) are read as raw JSON and
    folded into ONE ``Dataset.from_list`` — this sidesteps the schema-drift that makes
    ``load_dataset`` choke (the files' *extra* columns differ) and any cross-file
    feature-type mismatch. Per row: ``question``->``problem``; ``bboxs[0]`` (pixel) is
    normalized by ``width``/``height``; ``image`` (a bare basename) is resolved against
    ``image_root`` (env ``VISCOT_IMAGE_ROOT``, else the dataset dir). Rows whose image
    can't be found are dropped (loudly if *all* are — i.e. images not extracted yet).
    """
    import json

    from datasets import Dataset, Image

    files = _viscot_jsonl_files(path)
    if not files:
        raise ValueError(
            f"No metadata/*.jsonl under {path!r}; not a Visual-CoT dataset dir."
        )
    root = image_root or os.environ.get("VISCOT_IMAGE_ROOT") or path
    index = _build_viscot_image_index(root, verbose=verbose)

    records: list[dict[str, Any]] = []
    n_total = n_resolved = n_box = 0
    for jsonl in files:
        with open(jsonl, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "question" not in row or "answer" not in row or "image" not in row:
                    continue
                n_total += 1
                resolved = _resolve_viscot_image(
                    row.get("image"), row.get("dataset"), root, index
                )
                if resolved is None:
                    continue
                n_resolved += 1
                bbox = _normalize_viscot_bbox(
                    row.get("bboxs"), row.get("width"), row.get("height")
                )
                if bbox:
                    n_box += 1
                records.append(
                    {
                        "problem": str(row.get("question", "")).strip(),
                        "image": resolved,
                        "answer": row.get("answer"),
                        "bbox": bbox,
                    }
                )
    if not records:
        raise ValueError(
            f"Visual-CoT: 0/{n_total} rows had a resolvable image under {root!r}. "
            "Extract the image tars (cot_images_tar_split/) and point VISCOT_IMAGE_ROOT "
            "at the directory that contains the image files."
        )
    if verbose:
        print(
            f"[viscot] {n_resolved}/{n_total} rows with a resolved image "
            f"({n_box} with a usable evidence box) from {len(files)} metadata file(s); "
            f"image_root={root}.",
            flush=True,
        )
    return Dataset.from_list(records).cast_column("image", Image())
