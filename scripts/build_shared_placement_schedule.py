"""Build the shared frozen placement schedule used by B1-B5 (guide §5.3).

All comparison arms must place their synthetic cycles on the *same* real
train background at the *same* start times, so arm differences come from the
waveforms alone. This script pre-builds that schedule once, deterministically,
and freezes it behind a SHA-256; `place_*` runs then consume the schedule
instead of re-rolling their own.

Durations come from ``--lengths-csv`` (a column named ``samples``, e.g. a
paired-cycle plan or a generation summary export) or default to one uniform
length. The schedule guarantees non-overlap inside the guard envelope.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.event_placement import idle_runs, schedule_lengths  # noqa: E402


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_train_background(aligned_dir: Path, sample_seconds: int
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate train shards exactly like the placement step sees them."""
    with open(aligned_dir / "aligned_partition_manifest.json",
              encoding="utf-8") as stream:
        manifest = json.load(stream)
    shard_items = manifest["partitions"]["train"]["shards"]
    timestamps_parts, appliance_parts = [], []
    for shard in shard_items:
        with np.load(aligned_dir / shard["path"]) as data:
            timestamps_parts.append(data["timestamp"].astype(np.int64))
            appliance_parts.append(data["appliance_w"].astype(np.float32))
    timestamps = np.concatenate(timestamps_parts)
    appliance = np.concatenate(appliance_parts)
    del sample_seconds
    return timestamps, appliance


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aligned-dir", required=True,
                    help="aligned_partitions_v2 dir (train background)")
    ap.add_argument("--count", type=int, required=True,
                    help="number of events to schedule")
    ap.add_argument("--lengths-csv", default=None,
                    help="CSV with a 'samples' column of event lengths")
    ap.add_argument("--samples-per-cycle", type=int, default=900,
                    help="uniform length when --lengths-csv is absent")
    ap.add_argument("--sample-seconds", type=int, default=6)
    ap.add_argument("--idle-threshold-w", type=float, default=20.0)
    ap.add_argument("--guard-seconds", type=int, default=300)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    aligned_dir = Path(args.aligned_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamps, appliance = load_train_background(
        aligned_dir, args.sample_seconds)
    if args.lengths_csv:
        lengths = pd.read_csv(args.lengths_csv)[
            "samples"].to_numpy(dtype=np.int64)
        if len(lengths) != args.count:
            raise SystemExit(
                f"lengths rows ({len(lengths)}) != --count ({args.count})")
    else:
        lengths = np.full(args.count, args.samples_per_cycle, dtype=np.int64)

    runs = idle_runs(timestamps, appliance,
                     sample_seconds=args.sample_seconds,
                     idle_threshold_w=args.idle_threshold_w)
    guard_samples = args.guard_seconds // args.sample_seconds
    try:
        starts = schedule_lengths(runs, lengths, total_rows=len(timestamps),
                                  guard_samples=guard_samples,
                                  seed=args.seed)
    except RuntimeError as error:
        raise SystemExit(
            f"{error}; reduce --count, --samples-per-cycle or "
            "--guard-seconds") from error
    if np.any(starts < 0):
        raise SystemExit(
            f"could only place {int((starts >= 0).sum())}/{args.count} "
            "events; reduce --count or --guard-seconds")

    rows = []
    for index, (start, length) in enumerate(zip(starts, lengths)):
        rows.append({
            "event_index": index,
            "start_global_index": int(start),
            "end_global_index_exclusive": int(start + length),
            "samples": int(length),
            "duration_seconds": int(length * args.sample_seconds),
            "start_unix": int(timestamps[start]),
        })
    schedule = pd.DataFrame(rows)
    schedule_path = output_dir / "placement_schedule.csv"
    schedule.to_csv(schedule_path, index=False)

    occupied = np.zeros(len(timestamps), dtype=bool)
    overlap = 0
    for start, length in zip(starts, lengths):
        span = slice(start, start + length + guard_samples)
        overlap += int(occupied[span].sum())
        occupied[span] = True

    summary = {
        "protocol": "shared_placement_schedule_v1",
        "count": args.count,
        "seed": args.seed,
        "sample_seconds": args.sample_seconds,
        "idle_threshold_w": args.idle_threshold_w,
        "guard_seconds": args.guard_seconds,
        "total_background_rows": int(len(timestamps)),
        "envelope_overlap_rows": overlap,
        "schedule_sha256": sha256_of_file(schedule_path),
    }
    (output_dir / "placement_schedule_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(f"[schedule] events={args.count} overlap_rows={overlap} "
          f"-> {schedule_path}")
    print(f"[schedule] schedule_sha256={summary['schedule_sha256']}")


if __name__ == "__main__":
    main()
