"""Place one or more synthetic-cycle datasets on the shared train background.

All routes share the same event slots (same seed -> same schedule, envelope
sized to the longest cycle across routes), so downstream comparisons differ
only by the synthetic waveforms themselves. Output mirrors the B1/B2 placed
layout: ``<LABEL>/shards/shard_XXX.npz`` with augmented mains/appliance
fields plus a placement_summary.json, so prepare_nilm_b0_b2_inputs.py can
consume each route via ``--extra-arm LABEL=placed_dir``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.event_placement import idle_runs, schedule_lengths  # noqa: E402


def _save_shard(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def load_background(aligned_dir: Path) -> tuple[np.ndarray, np.ndarray,
                                                np.ndarray, list[str]]:
    with open(aligned_dir / "aligned_partition_manifest.json",
              encoding="utf-8") as stream:
        aligned = json.load(stream)
    shard_items = aligned["partitions"]["train"]["shards"]
    timestamps_parts, mains_parts, appliance_parts = [], [], []
    names = []
    for shard in shard_items:
        with np.load(aligned_dir / shard["path"]) as data:
            timestamps_parts.append(data["timestamp"].astype(np.int64))
            mains_parts.append(data["mains_w"].astype(np.float32))
            appliance_parts.append(data["appliance_w"].astype(np.float32))
        names.append(Path(shard["path"]).name)
    return (np.concatenate(timestamps_parts),
            np.concatenate(mains_parts),
            np.concatenate(appliance_parts),
            names)


def load_cycles(synthetic_dir: Path) -> list[np.ndarray]:
    summary = json.loads((synthetic_dir / "generation_summary.json")
                         .read_text(encoding="utf-8"))
    waves = []
    for record in summary["records"]:
        with np.load(synthetic_dir / "cycles"
                     / f"{record['synthetic_cycle_id']}.npz") as data:
            waves.append(data["appliance_w"].astype(np.float32))
    return waves


def parse_routes(specs: list[str]) -> list[tuple[str, Path]]:
    routes = []
    for spec in specs:
        label, _, path = spec.partition("=")
        if not label or not path:
            raise SystemExit(f"bad --synthetic-dir spec: {spec!r} "
                             "(expected LABEL=dir)")
        if "test" in label.lower():
            raise SystemExit("test is not a valid route label")
        routes.append((label, Path(path)))
    return routes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aligned-dir", required=True)
    ap.add_argument("--synthetic-dir", action="append", required=True,
                    help="LABEL=path to a generation output; repeatable, "
                         "all routes share the same event slots")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--sample-seconds", type=int, default=6)
    ap.add_argument("--idle-threshold-w", type=float, default=20.0)
    ap.add_argument("--guard-seconds", type=int, default=300)
    args = ap.parse_args()

    aligned_dir = Path(args.aligned_dir)
    output_dir = Path(args.output_dir)
    routes = parse_routes(args.synthetic_dir)
    route_waves = {label: load_cycles(path) for label, path in routes}
    counts = {len(w) for w in route_waves.values()}
    if len(counts) != 1:
        raise SystemExit(f"route cycle counts disagree: "
                         f"{ {k: len(v) for k, v in route_waves.items()} }")
    count = counts.pop()
    if count == 0:
        raise SystemExit("synthetic datasets are empty")
    envelope = max(max(len(w) for w in waves)
                   for waves in route_waves.values())

    timestamps, mains, appliance, shard_names = load_background(aligned_dir)
    runs = idle_runs(timestamps, appliance,
                     sample_seconds=args.sample_seconds,
                     idle_threshold_w=args.idle_threshold_w)
    guard_samples = args.guard_seconds // args.sample_seconds
    lengths = np.full(count, envelope, dtype=np.int64)
    starts = schedule_lengths(runs, lengths, total_rows=len(timestamps),
                              guard_samples=guard_samples, seed=args.seed)
    if np.any(starts < 0):
        raise SystemExit(
            f"could only place {int((starts >= 0).sum())}/{count} slots; "
            "reduce --count or --guard-seconds")

    output_dir.mkdir(parents=True, exist_ok=True)
    route_summaries = {}
    for label, waves in route_waves.items():
        overlay = np.zeros(len(timestamps), dtype=np.float32)
        placements = []
        for index, (start, wave) in enumerate(zip(starts, waves)):
            if len(wave) > envelope:
                raise SystemExit(
                    f"{label} cycle {index} longer than envelope")
            overlay[start:start + len(wave)] = wave
            placements.append({
                "synthetic_index": index,
                "start_global_index": int(start),
                "samples": int(len(wave)),
                "start_unix": int(timestamps[start]),
            })
        augmented_appliance = appliance + overlay
        augmented_mains = mains + overlay
        background_error = float(np.max(np.abs(
            (augmented_mains.astype(np.float64)
             - augmented_appliance.astype(np.float64))
            - (mains.astype(np.float64) - appliance.astype(np.float64)))))
        shard_records = []
        # Shard slices must match the aligned manifest exactly.
        with open(aligned_dir / "aligned_partition_manifest.json",
                  encoding="utf-8") as stream:
            shard_items = json.load(stream)["partitions"]["train"]["shards"]
        cursor = 0
        for shard_index, shard in enumerate(shard_items):
            with np.load(aligned_dir / shard["path"]) as data:
                rows = len(data["timestamp"])
            left, right = cursor, cursor + rows
            relative = Path(label) / "shards" / f"shard_{shard_index:03d}.npz"
            _save_shard(output_dir / relative, {
                "timestamp": timestamps[left:right],
                "original_mains_w": mains[left:right],
                "original_appliance_w": appliance[left:right],
                "synthetic_appliance_w": overlay[left:right],
                "augmented_mains_w": augmented_mains[left:right],
                "augmented_appliance_w": augmented_appliance[left:right],
            })
            shard_records.append({
                "path": relative.as_posix(),
                "source_aligned_shard": shard["path"],
                "rows": int(rows),
            })
            cursor = right
        route_summaries[label] = {
            "shards": shard_records,
            "cycles_placed": count,
            "envelope_samples": int(envelope),
            "synthetic_energy_wh": float(
                overlay.astype(np.float64).sum()
                * args.sample_seconds / 3600.0),
            "nonzero_synthetic_rows": int((overlay > 0).sum()),
            "max_background_invariance_error_w": background_error,
            "placements": placements,
        }

    occupied = np.zeros(len(timestamps), dtype=bool)
    overlap = 0
    for start in starts:
        span = slice(start, start + envelope)
        overlap += int(occupied[span].sum())
        occupied[span] = True

    summary = {
        "protocol": "synthetic_background_placement_v1",
        "partition": "train",
        "seed": args.seed,
        "shared_event_slots": True,
        "envelope_samples": int(envelope),
        "events": count,
        "event_overlap_rows": overlap,
        "idle_threshold_w": args.idle_threshold_w,
        "guard_seconds": args.guard_seconds,
        "arms": route_summaries,
    }
    summary_path = output_dir / "placement_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False,
                                       indent=2), encoding="utf-8")
    print(f"[placement] routes={list(route_summaries)} events={count} "
          f"envelope={envelope} overlaps={overlap}")
    print(f"[placement] summary -> {summary_path}")


if __name__ == "__main__":
    main()
