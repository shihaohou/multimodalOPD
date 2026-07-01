"""Rebuild sentence-level EAGLE geometry metrics from saved span artifacts."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from baseline.g0.analyze_g0 import load_records
from baseline.g0.eagle_probe import _map_metrics
from baseline.g0.eagle_src import add_value


def _json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _artifact_tag(record: dict) -> str:
    span_mode = str(record.get("eagle_target_span_mode", "sentence"))
    token_mode = str(record.get("eagle_token_mode", "per_token_mean"))
    return (
        f"{record.get('subset', '')}_{record.get('sample_id', '')}_"
        f"{record.get('condition', '')}_{span_mode}_{token_mode}"
    )


def _record_key(record: dict) -> dict:
    return {
        "model": record.get("model", ""),
        "condition": record.get("condition", ""),
        "subset": record.get("subset", ""),
        "sample_id": str(record.get("sample_id", "")),
    }


def rebuild_record(record: dict, run_dir: str, threshold: str, top_frac: float) -> dict:
    tag = _artifact_tag(record)
    artifact_dir = os.path.join(run_dir, "eagle_artifacts")
    json_path = os.path.join(artifact_dir, f"{tag}.json")
    npz_path = os.path.join(artifact_dir, f"{tag}.npz")
    if not os.path.exists(json_path) or not os.path.exists(npz_path):
        raise FileNotFoundError(f"{json_path} / {npz_path}")

    with open(json_path, encoding="utf-8") as handle:
        meta = json.load(handle)
    eagle_json = meta.get("eagle_json_file")
    if not isinstance(eagle_json, dict) or "smdl_score" not in eagle_json:
        raise ValueError(f"{json_path}: missing span-level eagle_json_file")

    with np.load(npz_path) as arrays:
        if "eagle_s_set" not in arrays:
            raise ValueError(f"{npz_path}: missing eagle_s_set")
        s_set = np.asarray(arrays["eagle_s_set"])
    amap = add_value(s_set, eagle_json)[0][:, :, 0].astype(np.float64)
    bbox = tuple(float(x) for x in record["gt_bbox"])
    result, pred_mask, pred_norm = _map_metrics(amap, bbox, threshold, top_frac)

    out = _record_key(record)
    out.update(
        {
            "spatial_metric_source": "sentence_span_artifact",
            "artifact_tag": tag,
            "iou_eagle": float(result["mask_iou"]),
            "eagle_bbox_iou": float(result["bbox_iou"]),
            "pointing_eagle": float(result["pointing"]),
            "pointing_at1": float(result["pointing"]),
            "area_eagle": float(pred_mask.mean()),
            "eagle_energy": float(result["energy"]) if np.isfinite(result["energy"]) else 0.0,
            "energy_in_box": float(result["energy"]) if np.isfinite(result["energy"]) else 0.0,
            "iou_top10": float(result["iou_top10"]),
            "iou_top20": float(result["iou_top20"]),
            "eagle_pred_box": list(pred_norm) if pred_norm is not None else None,
        }
    )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.run_dir)
    config_path = os.path.join(args.run_dir, "config.json")
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    threshold = str(config.get("eagle_threshold", "mean"))
    top_frac = float(config.get("eagle_top_frac", 0.25))
    output = args.output or os.path.join(args.run_dir, "sentence_span_metrics.jsonl")

    rebuilt = []
    missing = []
    for record in records:
        if str(record.get("eagle_target_span_mode", "")) != "sentence":
            continue
        try:
            rebuilt.append(rebuild_record(record, args.run_dir, threshold, top_frac))
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            missing.append(f"{record.get('subset')}/{record.get('sample_id')} {record.get('condition')}: {exc}")

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        for record in rebuilt:
            handle.write(json.dumps(record, default=_json_safe) + "\n")

    print(
        f"[eagle.span-metrics] wrote {len(rebuilt)}/{len(rebuilt) + len(missing)} "
        f"sentence records -> {output}"
    )
    if missing:
        print(f"[eagle.span-metrics] missing {len(missing)} artifact(s); first: {missing[0]}")
        if not args.allow_missing:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
