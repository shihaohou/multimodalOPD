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
    imagefolder trap) and remapped to ``problem`` / ``image`` / ``answer``. Every
    other ``dataset_name`` (HuggingFace id, canonical local dataset) defers to
    :func:`vigos.dataset_utils.load_vigos_dataset`.
    """
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
