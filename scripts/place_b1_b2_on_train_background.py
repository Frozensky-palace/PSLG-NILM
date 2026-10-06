"""Place paired B1/B2 cycles on identical idle UK-DALE train backgrounds."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.event_placement import idle_runs, schedule_lengths  # noqa: E402


def _save_shard(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aligned-dir", required=True)
    ap.add_argument("--paired-cycle-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--sample-seconds", type=int, default=6)
    ap.add_argument("--idle-threshold-w", type=float, default=20.0)
    ap.add_argument("--guard-seconds", type=int, default=300)
    args = ap.parse_args()

    aligned_dir = Path(args.aligned_dir)
    paired_dir = Path(args.paired_cycle_dir)
    output_dir = Path(args.output_dir)
    with open(aligned_dir / "aligned_partition_manifest.json", encoding="utf-8") as stream:
        aligned_manifest = json.load(stream)
    shard_items = aligned_manifest["partitions"]["train"]["shards"]

    timestamps_parts, mains_parts, appliance_parts = [], [], []
    shard_slices = []
    cursor = 0
    for shard in shard_items:
        with np.load(aligned_dir / shard["path"]) as data:
            timestamps = data["timestamp"].copy()
            mains = data["mains_w"].copy()
            appliance = data["appliance_w"].copy()
        timestamps_parts.append(timestamps)
        mains_parts.append(mains)
        appliance_parts.append(appliance)
        shard_slices.append((cursor, cursor + len(timestamps), Path(shard["path"]).name))
        cursor += len(timestamps)
    timestamps = np.concatenate(timestamps_parts).astype(np.int64)
    mains = np.concatenate(mains_parts).astype(np.float32)
    appliance = np.concatenate(appliance_parts).astype(np.float32)

    plan = pd.read_csv(paired_dir / "paired_cycle_plan.csv")
    lengths = plan["samples"].to_numpy(dtype=np.int64)
    runs = idle_runs(
        timestamps, appliance, sample_seconds=args.sample_seconds,
        idle_threshold_w=args.idle_threshold_w)
    guard_samples = args.guard_seconds // args.sample_seconds
    starts = schedule_lengths(
        runs, lengths, total_rows=len(timestamps),
        guard_samples=guard_samples, seed=args.seed)

    overlay = {"B1": np.zeros(len(timestamps), dtype=np.float32),
               "B2": np.zeros(len(timestamps), dtype=np.float32)}
    schedule_records = []
    for row, start in zip(plan.itertuples(index=False), starts):
        length = int(row.samples)
        for arm, rel_path in (("B1", row.b1_path), ("B2", row.b2_path)):
            with np.load(paired_dir / rel_path) as data:
                waveform = data["appliance_w"].astype(np.float32)
            if len(waveform) != length:
                raise RuntimeError(f"{arm} length mismatch for {row.synthetic_index}")
            overlay[arm][start:start + length] = waveform
        schedule_records.append({
            "synthetic_index": int(row.synthetic_index),
            "template_cycle_id": row.template_cycle_id,
            "start_global_index": int(start),
            "end_global_index_exclusive": int(start + length),
            "start_unix": int(timestamps[start]),
            "end_unix": int(timestamps[start + length - 1]),
            "samples": length,
            "duration_seconds": length * args.sample_seconds,
            "base_max_appliance_w": float(appliance[start:start + length].max()),
        })

    schedule = pd.DataFrame(schedule_records).sort_values("synthetic_index")
    output_dir.mkdir(parents=True, exist_ok=True)
    schedule_path = output_dir / "common_event_schedule.csv"
    schedule.to_csv(schedule_path, index=False)
    arm_summaries = {}
    for arm in ("B1", "B2"):
        synthetic = overlay[arm]
        augmented_appliance = appliance + synthetic
        augmented_mains = mains + synthetic
        background_error = np.max(np.abs(
            (augmented_mains.astype(np.float64) - augmented_appliance.astype(np.float64))
            - (mains.astype(np.float64) - appliance.astype(np.float64))))
        shard_records = []
        for shard_index, (left, right, source_name) in enumerate(shard_slices):
            relative = Path(arm) / "shards" / f"shard_{shard_index:03d}.npz"
            _save_shard(output_dir / relative, {
                "timestamp": timestamps[left:right],
                "original_mains_w": mains[left:right],
                "original_appliance_w": appliance[left:right],
                "synthetic_appliance_w": synthetic[left:right],
                "augmented_mains_w": augmented_mains[left:right],
                "augmented_appliance_w": augmented_appliance[left:right],
            })
            shard_records.append({
                "path": relative.as_posix(),
                "source_aligned_shard": source_name,
                "rows": int(right - left),
            })
        arm_summaries[arm] = {
            "shards": shard_records,
            "synthetic_energy_wh": float(
                synthetic.astype(np.float64).sum() * args.sample_seconds / 3600.0),
            "nonzero_synthetic_rows": int((synthetic > 0).sum()),
            "max_background_invariance_error_w": float(background_error),
        }

    occupied = np.zeros(len(timestamps), dtype=np.int8)
    overlap = 0
    for start, length in zip(starts, lengths):
        overlap += int(occupied[start:start + length].sum())
        occupied[start:start + length] += 1
    summary = {
        "protocol": "common_background_placement_v1",
        "partition": "train",
        "seed": args.seed,
        "paired_events": int(len(schedule)),
        "same_event_schedule": True,
        "same_original_mains": True,
        "same_original_appliance": True,
        "idle_threshold_w": args.idle_threshold_w,
        "guard_seconds": args.guard_seconds,
        "event_overlap_rows": overlap,
        "max_base_appliance_during_placement_w": float(
            schedule["base_max_appliance_w"].max()),
        "arms": arm_summaries,
    }
    summary_path = output_dir / "placement_summary.json"
    with open(summary_path, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(f"[placement] events={len(schedule)}, overlaps={overlap}, "
          f"max base appliance={summary['max_base_appliance_during_placement_w']:.2f}W")
    print(f"[placement] summary -> {summary_path}")


if __name__ == "__main__":
    main()
