"""Freeze B0/B1/B2 Seq2Point inputs, windows and normalization statistics."""
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

from src.data.research_split import file_sha256  # noqa: E402
from src.nilm.window_dataset import valid_center_ranges  # noqa: E402


def _portable_path(path: Path) -> str:
    """Store a project-relative POSIX path so the manifest works on servers."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _stats(paths: list[Path], mains_field: str, appliance_field: str) -> dict:
    count = 0
    sums = {"mains_w": 0.0, "appliance_w": 0.0}
    squares = {"mains_w": 0.0, "appliance_w": 0.0}
    maxima = {"mains_w": -np.inf, "appliance_w": -np.inf}
    for path in paths:
        with np.load(path) as data:
            for name, field in (("mains_w", mains_field),
                                ("appliance_w", appliance_field)):
                values = data[field].astype(np.float64)
                sums[name] += float(values.sum())
                squares[name] += float(np.square(values).sum())
                maxima[name] = max(maxima[name], float(values.max()))
            count += len(data[mains_field])
    result = {}
    for name in sums:
        mean = sums[name] / count
        variance = max(0.0, squares[name] / count - mean * mean)
        result[name] = {
            "mean": mean,
            "std": max(variance ** 0.5, 1e-6),
            "max": maxima[name],
            "samples": count,
            "source": "B0 real train only",
        }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aligned-dir", required=True)
    ap.add_argument("--placed-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--window-length", type=int, default=599)
    ap.add_argument("--train-stride", type=int, default=6)
    ap.add_argument("--eval-stride", type=int, default=1)
    ap.add_argument("--sample-seconds", type=int, default=6)
    ap.add_argument("--extra-arm", action="append", default=[],
                    help="LABEL=placed_dir for a D/E route placed by "
                         "place_synthetics_on_background.py; repeatable")
    args = ap.parse_args()

    aligned_dir = Path(args.aligned_dir)
    placed_dir = Path(args.placed_dir)
    output_dir = Path(args.output_dir)
    extra_arms: dict[str, Path] = {}
    for spec in args.extra_arm:
        label, _, placed = spec.partition("=")
        label = label.strip()
        if not label or not placed:
            raise SystemExit(f"bad --extra-arm spec: {spec!r}")
        if "test" in label.lower():
            raise SystemExit("test is not a valid arm label")
        if not all(c.isalnum() or c == "_" for c in label):
            raise SystemExit(
                f"arm label must be alphanumeric/underscore: {label!r}")
        extra_arms[label] = Path(placed)

    with open(aligned_dir / "aligned_partition_manifest.json", encoding="utf-8") as stream:
        aligned = json.load(stream)
    with open(placed_dir / "placement_summary.json", encoding="utf-8") as stream:
        placed = json.load(stream)

    sources = {arm: {p: [] for p in ("train", "validation", "test")}
               for arm in ("B0", "B1", "B2", *extra_arms)}
    for partition in ("train", "validation", "test"):
        for shard in aligned["partitions"][partition]["shards"]:
            entry = {
                "path": _portable_path(aligned_dir / shard["path"]),
                "mains_field": "mains_w",
                "appliance_field": "appliance_w",
                "rows": shard["valid_rows"],
            }
            for arm in sources:
                sources[arm][partition].append(dict(entry))
    for arm in ("B1", "B2"):
        sources[arm]["train"] = [{
            "path": _portable_path(placed_dir / shard["path"]),
            "mains_field": "augmented_mains_w",
            "appliance_field": "augmented_appliance_w",
            "rows": shard["rows"],
        } for shard in placed["arms"][arm]["shards"]]
    for label, extra_placed_dir in extra_arms.items():
        with open(extra_placed_dir / "placement_summary.json",
                  encoding="utf-8") as stream:
            extra_placed = json.load(stream)
        if label not in extra_placed.get("arms", {}):
            raise SystemExit(
                f"extra arm {label!r} missing from "
                f"{extra_placed_dir / 'placement_summary.json'}")
        sources[label]["train"] = [{
            "path": _portable_path(extra_placed_dir / shard["path"]),
            "mains_field": "augmented_mains_w",
            "appliance_field": "augmented_appliance_w",
            "rows": shard["rows"],
        } for shard in extra_placed["arms"][label]["shards"]]

    coverage = pd.read_csv(aligned_dir / "cycle_alignment_coverage.csv")
    excluded = coverage[
        (coverage["partition"] == "test")
        & (~coverage["analysis_eligible"].astype(bool))
    ]
    test_exclusions = [
        (int(row.start_unix), int(row.end_unix))
        for row in excluded.itertuples(index=False)
    ]
    range_records = []
    center_counts = {}
    active_counts = {}
    for partition in ("train", "validation", "test"):
        stride = args.train_stride if partition == "train" else args.eval_stride
        center_counts[partition] = 0
        active_counts[partition] = 0
        exclusions = test_exclusions if partition == "test" else []
        for shard_index, entry in enumerate(sources["B0"][partition]):
            with np.load(entry["path"]) as data:
                timestamps = data["timestamp"]
                appliance = data[entry["appliance_field"]]
                ranges = valid_center_ranges(
                    timestamps, window_length=args.window_length,
                    sample_seconds=args.sample_seconds, stride=stride,
                    excluded_intervals=exclusions)
                for first, stop, step, count in ranges:
                    centers = np.arange(first, stop, step, dtype=np.int64)
                    range_records.append({
                        "partition": partition,
                        "shard_index": shard_index,
                        "first_center": first,
                        "stop_center_exclusive": stop,
                        "stride": step,
                        "count": count,
                    })
                    center_counts[partition] += count
                    active_counts[partition] += int((appliance[centers] > 20).sum())

    output_dir.mkdir(parents=True, exist_ok=True)
    ranges_frame = pd.DataFrame(range_records)
    ranges_path = output_dir / "window_ranges.csv"
    ranges_frame.to_csv(ranges_path, index=False)
    train_pools = {}
    train_ranges = ranges_frame[ranges_frame["partition"] == "train"].reset_index(drop=True)
    for arm in sources:
        active_parts, inactive_parts = [], []
        ordinal = 0
        current_shard = None
        current_data = None
        try:
            for row in train_ranges.itertuples(index=False):
                shard_index = int(row.shard_index)
                if shard_index != current_shard:
                    if current_data is not None:
                        current_data.close()
                    entry = sources[arm]["train"][shard_index]
                    current_data = np.load(entry["path"])
                    current_shard = shard_index
                centers = np.arange(
                    int(row.first_center), int(row.stop_center_exclusive),
                    int(row.stride), dtype=np.int64)
                entry = sources[arm]["train"][shard_index]
                is_active = current_data[entry["appliance_field"]][centers] > 20
                ordinals = np.arange(ordinal, ordinal + len(centers), dtype=np.int64)
                active_parts.append(ordinals[is_active])
                inactive_parts.append(ordinals[~is_active])
                ordinal += len(centers)
        finally:
            if current_data is not None:
                current_data.close()
        active = np.concatenate(active_parts)
        inactive = np.concatenate(inactive_parts)
        active_path = output_dir / f"{arm.lower()}_train_active_indices.npy"
        inactive_path = output_dir / f"{arm.lower()}_train_inactive_indices.npy"
        np.save(active_path, active)
        np.save(inactive_path, inactive)
        train_pools[arm] = {
            "active_path": active_path.name,
            "inactive_path": inactive_path.name,
            "active_count": int(len(active)),
            "inactive_count": int(len(inactive)),
            "active_sha256": file_sha256(active_path),
            "inactive_sha256": file_sha256(inactive_path),
        }

    validation_count = center_counts["validation"]
    monitor_size = min(100_000, validation_count)
    monitor_rng = np.random.default_rng(20260920)
    monitor_indices = np.sort(monitor_rng.choice(
        validation_count, size=monitor_size, replace=False)).astype(np.int64)
    monitor_path = output_dir / "validation_monitor_indices.npy"
    np.save(monitor_path, monitor_indices)
    normalization = _stats(
        [Path(item["path"]) for item in sources["B0"]["train"]],
        "mains_w", "appliance_w")
    normalization_path = output_dir / "normalization.json"
    with open(normalization_path, "w", encoding="utf-8") as stream:
        json.dump(normalization, stream, ensure_ascii=False, indent=2)

    manifest = {
        "protocol": "nilm_b0_b2_seq2point_inputs_v1",
        "window_length": args.window_length,
        "window_duration_seconds": args.window_length * args.sample_seconds,
        "sample_seconds": args.sample_seconds,
        "train_stride": args.train_stride,
        "eval_stride": args.eval_stride,
        "target": "center appliance power",
        "normalization_source": "B0 real train only and shared by B0/B1/B2",
        "validation_and_test_real_only": True,
        "test_excluded_intervals": test_exclusions,
        "window_counts": center_counts,
        "real_active_center_counts": active_counts,
        "train_sampling_pools": train_pools,
        "validation_monitor": {
            "path": monitor_path.name,
            "count": int(monitor_size),
            "seed": 20260920,
            "sha256": file_sha256(monitor_path),
            "purpose": "early stopping only; final metrics use full real partitions",
        },
        "sources": sources,
        "source_hashes": {
            "aligned_manifest": file_sha256(
                aligned_dir / "aligned_partition_manifest.json"),
            "placement_summary": file_sha256(
                placed_dir / "placement_summary.json"),
            "window_ranges": file_sha256(ranges_path),
            "normalization": file_sha256(normalization_path),
        },
    }
    manifest_path = output_dir / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    print(f"[nilm-input] windows={center_counts}")
    print(f"[nilm-input] real active centers={active_counts}")
    print(f"[nilm-input] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
