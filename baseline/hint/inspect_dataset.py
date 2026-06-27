"""Inspect a dataset's schema for Grounding-Hint Distillation (GHD) wiring.

Standalone (only needs ``datasets`` + ``PIL``; no training stack — run it by file
path so the package ``__init__`` isn't imported). For each dataset path/id it
prints the columns, row count, a truncated sample, and — crucially for GHD — what
the candidate **bbox** field looks like (nested? plural? normalized-[0,1] vs pixel
coords?) and how the **image** is stored (embedded PIL vs a filename string that
needs resolving). Ends with a suggested ``ANSWER_FIELD`` / ``BBOX_FIELD`` / bbox
normalization for the run command.

Usage (on the training box, in the OPD env):
    uv run python baseline/hint/inspect_dataset.py $D/saliency-r1-8k $D/vision-cot
    uv run python baseline/hint/inspect_dataset.py --split train peterant330/saliency-r1-8k
"""

from __future__ import annotations

import argparse
import os

_BBOX_CANDIDATES = ("bbox", "bboxs", "boxes", "bbox_norm", "bounding_box", "gt_bbox")
_QUESTION_CANDIDATES = ("problem", "question", "query", "prompt")
_ANSWER_CANDIDATES = ("answer", "solution", "label", "gt_answer", "full_answer")


def _load(path: str, split: str):
    """Load a HF id OR local dir (save_to_disk / parquet / jsonl / arrow / dir)."""
    from datasets import load_dataset

    if not os.path.isdir(path):
        return load_dataset(path, split=split)
    if os.path.exists(os.path.join(path, "dataset_info.json")) or os.path.exists(
        os.path.join(path, "state.json")
    ):
        from datasets import load_from_disk

        disk = load_from_disk(path)
        if hasattr(disk, "keys"):
            return disk[split] if split in disk else disk[next(iter(disk.keys()))]
        return disk
    try:
        return load_dataset(path, split=split)
    except Exception:
        import glob

        for ext, builder in (
            ("parquet", "parquet"),
            ("arrow", "arrow"),
            ("jsonl", "json"),
            ("json", "json"),
        ):
            files = sorted(glob.glob(os.path.join(path, "**", f"*.{ext}"), recursive=True))
            if files:
                return load_dataset(builder, data_files=files, split="train")
        raise


def _coerce(value):
    """A string like ``"[0.1, 0.2, 0.5, 0.6]"`` -> the parsed list (saliency-r1-8k
    stores its bbox as a string); anything else is returned unchanged."""
    if isinstance(value, str):
        import ast
        import json

        text = value.strip()
        if not text:
            return value
        try:
            return json.loads(text)
        except Exception:
            try:
                return ast.literal_eval(text)
            except Exception:
                return value
    return value


def _flatten_numbers(value):
    value = _coerce(value)
    out = []
    if isinstance(value, (list, tuple)):
        for v in value:
            out.extend(_flatten_numbers(v))
    elif isinstance(value, (int, float)):
        out.append(float(value))
    return out


def _summarize(value, key: str = "") -> str:
    cls = type(value).__name__
    if hasattr(value, "size") and hasattr(value, "mode"):  # PIL image
        return f"<PIL {value.mode} size={value.size}>"
    if isinstance(value, dict):
        if "bytes" in value or "path" in value:
            return f"<image dict path={value.get('path')!r} has_bytes={value.get('bytes') is not None}>"
        return f"<dict keys={list(value)[:8]}>"
    if isinstance(value, str):
        s = value.replace("\n", "\\n")
        return f"'{s[:200]}'{'…' if len(s) > 200 else ''} (str len={len(value)})"
    if isinstance(value, (list, tuple)):
        return f"{cls} len={len(value)} head={value[:4]}"
    return f"{value!r} ({cls})"


def _bbox_verdict(value) -> str:
    nums = _flatten_numbers(value)
    if not nums:
        return "no numeric values"
    value = _coerce(value)
    mx = max(nums)
    n = len(nums)
    nested = isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple))
    plural = nested and len(value) > 1
    norm = "NORMALIZED [0,1]" if mx <= 1.5 else f"PIXEL coords (max={mx:g})"
    shape = (
        f"nested {len(value)} box(es)" if nested else f"flat {n} numbers"
    )
    return f"{shape}; {norm}{'; MULTI-BOX (uses first)' if plural else ''}"


def _raw_jsonl_probe(path: str) -> None:
    """Fallback when ``load_dataset`` chokes on schema drift across files (Visual-CoT
    ships per-domain JSONLs with mismatched columns). Reads the first line of each
    JSONL raw, prints the per-file column sets + the columns common to ALL (the safe
    schema for an adapter), a sample row, the bbox verdict, and — crucially — which
    root the ``image`` path resolves under (so the adapter can locate the files)."""
    import glob
    import json

    files = sorted(glob.glob(os.path.join(path, "**", "*.jsonl"), recursive=True))
    if not files:
        files = sorted(glob.glob(os.path.join(path, "**", "*.json"), recursive=True))
    if not files:
        print("  (no .jsonl/.json under the dir to raw-probe)")
        return
    print(f"  load_dataset failed (schema drift); raw-probing {len(files)} file(s).")
    try:
        print(f"  top-level dir: {sorted(os.listdir(path))[:25]}")
    except OSError:
        pass

    colsets: dict[tuple, list] = {}
    sample_row = sample_file = None
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                row = json.loads(fh.readline())
        except Exception as exc:  # noqa: BLE001
            print(f"    {os.path.relpath(f, path)}: !! {type(exc).__name__}")
            continue
        colsets.setdefault(tuple(sorted(row)), []).append(os.path.relpath(f, path))
        if sample_row is None:
            sample_row, sample_file = row, f

    print("\n  -- column sets across files --")
    for keys, fs in colsets.items():
        print(f"    {list(keys)}  <- {len(fs)} file(s), e.g. {fs[0]}")
    if colsets:
        common = set.intersection(*[set(k) for k in colsets])
        print(f"  -- columns COMMON to ALL files (safe adapter schema): {sorted(common)} --")
    if sample_row is None:
        return

    print(f"\n  -- sample row ({os.path.relpath(sample_file, path)}) --")
    for k, v in sample_row.items():
        print(f"    {k:16s}: {_summarize(v, k)}")
    for bk in [k for k in sample_row if k in _BBOX_CANDIDATES or "box" in k.lower()]:
        print(f"  bbox field {bk!r}: {_bbox_verdict(sample_row[bk])}  raw={sample_row[bk]!r}")

    img_key = next(
        (k for k in ("image", "images", "image_path", "file_name") if k in sample_row),
        None,
    )
    if img_key is not None:
        val = sample_row[img_key]
        if isinstance(val, (list, tuple)):
            val = val[0] if val else None
        print(f"\n  -- image field {img_key!r} value: {val!r} --")
        if isinstance(val, str):
            meta_dir = os.path.dirname(sample_file)
            roots = [
                path,
                os.path.join(path, "images"),
                os.path.join(path, "image"),
                meta_dir,
                os.path.dirname(meta_dir),
            ]
            hit = next((r for r in roots if os.path.exists(os.path.join(r, val))), None)
            if hit:
                print(f"  -> RESOLVES: <root>/{val}  with root = {hit}")
            else:
                print("  -> NOT found; tried (need the real image root):")
                for r in roots:
                    print(f"       {os.path.join(r, val)}")


def inspect(path: str, split: str, n: int) -> None:
    print("\n" + "=" * 88)
    print(f"DATASET: {path}  (split={split})")
    print("=" * 88)
    try:
        ds = _load(path, split)
    except Exception as exc:  # noqa: BLE001
        print(f"  !! load_dataset failed: {type(exc).__name__}: {str(exc)[:200]}")
        if os.path.isdir(path):
            _raw_jsonl_probe(path)
        return
    cols = list(ds.column_names)
    print(f"rows={len(ds)}  columns={cols}")

    sample = ds[0]
    print("\n-- sample[0] --")
    for key in cols:
        print(f"  {key:14s}: {_summarize(sample[key], key)}")

    q = next((c for c in _QUESTION_CANDIDATES if c in cols), None)
    a = next((c for c in _ANSWER_CANDIDATES if c in cols), None)
    bbox_cols = [c for c in cols if c in _BBOX_CANDIDATES] or [
        c for c in cols if "box" in c.lower() or "bbox" in c.lower()
    ]
    img_col = next((c for c in ("images", "image", "image_path") if c in cols), None)

    print("\n-- bbox candidate field(s) --")
    if not bbox_cols:
        print("  (none found — this dataset has no obvious bbox column!)")
    for bc in bbox_cols:
        vals = [ds[i][bc] for i in range(min(n, len(ds)))]
        print(f"  {bc!r}: verdict={_bbox_verdict(vals[0])}")
        for v in vals[:3]:
            print(f"      raw: {v!r}")

    img_embedded = False
    if img_col is not None:
        v = sample[img_col]
        img_embedded = hasattr(v, "size") and hasattr(v, "mode")
        if isinstance(v, str):
            exists = os.path.exists(v)
            print(
                f"\n-- image field {img_col!r}: FILENAME string (exists={exists}); "
                "needs path resolution + an image root --"
            )
        elif isinstance(v, dict):
            print(f"\n-- image field {img_col!r}: dict (path/bytes) --")
        else:
            print(f"\n-- image field {img_col!r}: embedded {_summarize(v)} --")

    print("\n-- GHD wiring suggestion --")
    width_h = "width" in cols and "height" in cols
    bbox_pick = bbox_cols[0] if bbox_cols else "<none>"
    norm_guess = "normalized" if bbox_cols and max(_flatten_numbers(sample[bbox_cols[0]]) or [9]) <= 1.5 else "PIXEL (needs ÷ width,height)"
    print(f"  question->problem : {q!r}{'  (already problem)' if q == 'problem' else '  (needs rename)'}")
    print(f"  ANSWER_FIELD      : {a!r}")
    print(f"  BBOX_FIELD        : {bbox_pick!r}  ({norm_guess}; width/height cols present={width_h})")
    print(f"  image             : {'embedded PIL (OK as-is)' if img_embedded else 'filename/dict -> needs adapter+resolve'}")
    ready = (q == "problem") and img_embedded and bbox_cols and norm_guess == "normalized"
    print(f"  => {'READY for train_opd_hint.py as-is' if ready else 'NEEDS an adapter (rename/resolve/normalize)'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="Dataset dirs or HF ids to inspect.")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=5, help="Rows to scan for bbox values.")
    args = ap.parse_args()
    for path in args.paths:
        inspect(path, args.split, args.n)


if __name__ == "__main__":
    main()
