"""Live progress and ETA monitor for sharded EAGLE-G0 runs."""

from __future__ import annotations

import argparse
import glob
import json
import os
import time


def _record_count(run_dir: str) -> int:
    keys = set()
    for path in glob.glob(os.path.join(run_dir, "records*.jsonl")):
        try:
            handle = open(path, encoding="utf-8")
        except OSError:
            continue
        with handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                keys.add(
                    (
                        record.get("model", ""),
                        record.get("condition", ""),
                        record.get("subset", ""),
                        str(record.get("sample_id", "")),
                    )
                )
    return len(keys)


def _duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "?"
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _snapshot(run_dirs: list[str]) -> tuple[list[int], int]:
    counts = [_record_count(path) for path in run_dirs]
    return counts, sum(counts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--expected-records-per-model", type=int, default=0)
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--watch-pids", nargs="*", type=int, default=[])
    parser.add_argument("--label", default="")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected_each = max(0, args.expected_records_per_model)
    expected_total = expected_each * len(args.run_dirs)
    initial_counts, initial_total = _snapshot(args.run_dirs)
    started = time.monotonic()

    while True:
        counts, total = _snapshot(args.run_dirs)
        elapsed = time.monotonic() - started
        produced = max(0, total - initial_total)
        rate = produced / elapsed if elapsed > 0 else 0.0
        pct = 100.0 * total / expected_total if expected_total else None

        model_parts = []
        model_etas = []
        for path, count, initial_count in zip(args.run_dirs, counts, initial_counts):
            name = os.path.basename(os.path.normpath(path))
            target = f"/{expected_each}" if expected_each else ""
            model_rate = max(0, count - initial_count) / elapsed if elapsed > 0 else 0.0
            model_remaining = max(0, expected_each - count) if expected_each else 0
            model_eta = 0.0 if expected_each and model_remaining == 0 else (
                model_remaining / model_rate if expected_each and model_rate > 0 else None
            )
            if expected_each:
                model_etas.append(model_eta)
            model_parts.append(f"{name}={count}{target} ETA={_duration(model_eta)}")
        eta = max(model_etas) if model_etas and all(value is not None for value in model_etas) else None
        prefix = f"[eagle.progress:{args.label}]" if args.label else "[eagle.progress]"
        overall = f"{total}/{expected_total} ({pct:.1f}%)" if expected_total else f"{total}/?"
        print(
            f"{prefix} records={overall} elapsed={_duration(elapsed)} "
            f"rate={rate * 60:.2f} rec/min ETA={_duration(eta)} | "
            + " ".join(model_parts),
            flush=True,
        )

        if args.once:
            return
        if args.watch_pids and not any(_alive(pid) for pid in args.watch_pids):
            return
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    main()
