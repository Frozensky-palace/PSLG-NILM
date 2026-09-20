"""Build a provenance-preserving real-cycle library from aligned shards."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ARRAY_COLUMNS = (
    "timestamp", "mains_w", "appliance_w", "background_signed_w",
    "background_clipped_w", "segment_id",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aligned-dir", required=True)
    ap.add_argument("--partition", choices=("train", "validation", "test"),
                    default="train")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    aligned_dir = Path(args.aligned_dir)
    output_dir = Path(args.output_dir)
    coverage_path = aligned_dir / "cycle_alignment_coverage.csv"
    aligned_manifest_path = aligned_dir / "aligned_partition_manifest.json"
    if not coverage_path.exists() or not aligned_manifest_path.exists():
        raise SystemExit("aligned coverage CSV and manifest are required")

    coverage = pd.read_csv(coverage_path)
    required = {"cycle_id", "partition", "start_unix", "end_unix",
                "expected_grid_samples", "analysis_eligible"}
    missing = required.difference(coverage.columns)
    if missing:
        raise SystemExit(f"coverage file missing columns: {sorted(missing)}")
    cycles = coverage[
        (coverage["partition"] == args.partition)
        & coverage["analysis_eligible"].astype(bool)
    ].sort_values(["start_unix", "cycle_id"]).reset_index(drop=True)
    if cycles.empty:
        raise SystemExit(f"no analysis-eligible cycles in {args.partition}")

    with open(aligned_manifest_path, encoding="utf-8") as stream:
        aligned_manifest = json.load(stream)
    shard_items = aligned_manifest["partitions"][args.partition]["shards"]
    buffers = {
        str(row.cycle_id): {column: [] for column in ARRAY_COLUMNS}
        for row in cycles.itertuples(index=False)
    }
    source_shards = {str(cid): [] for cid in cycles["cycle_id"]}

    for shard in shard_items:
        shard_path = aligned_dir / shard["path"]
        with np.load(shard_path) as data:
            timestamps = data["timestamp"]
            overlapping = cycles[
                (cycles["start_unix"] <= int(timestamps[-1]))
                & (cycles["end_unix"] >= int(timestamps[0]))
            ] if len(timestamps) else cycles.iloc[0:0]
            for row in overlapping.itertuples(index=False):
                left = int(np.searchsorted(timestamps, int(row.start_unix), "left"))
                right = int(np.searchsorted(timestamps, int(row.end_unix), "right"))
                if right <= left:
                    continue
                cycle_id = str(row.cycle_id)
                for column in ARRAY_COLUMNS:
                    buffers[cycle_id][column].append(data[column][left:right].copy())
                source_shards[cycle_id].append(shard["path"])

    cycle_dir = output_dir / "cycles"
    cycle_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, row in enumerate(cycles.itertuples(index=False)):
        cycle_id = str(row.cycle_id)
        arrays = {
            column: np.concatenate(buffers[cycle_id][column])
            if buffers[cycle_id][column] else np.array([])
            for column in ARRAY_COLUMNS
        }
        observed = len(arrays["timestamp"])
        expected = int(row.expected_grid_samples)
        if observed != expected:
            raise RuntimeError(
                f"{cycle_id}: expected {expected} aligned samples, got {observed}")
        arrays["relative_time_s"] = (
            arrays["timestamp"] - arrays["timestamp"][0]).astype(np.int64)
        relative_path = Path("cycles") / f"cycle_{index:04d}.npz"
        temporary = output_dir / relative_path.with_name(
            relative_path.name + ".tmp.npz")
        final_path = output_dir / relative_path
        np.savez_compressed(temporary, **arrays)
        temporary.replace(final_path)
        sample_seconds = int(aligned_manifest["sample_seconds"])
        records.append({
            "library_index": index,
            "cycle_id": cycle_id,
            "partition": args.partition,
            "path": relative_path.as_posix(),
            "sha256": _sha256(final_path),
            "start_unix": int(row.start_unix),
            "end_unix": int(row.end_unix),
            "samples": observed,
            "duration_grid_seconds": observed * sample_seconds,
            "appliance_energy_wh": float(
                arrays["appliance_w"].astype(np.float64).sum()
                * sample_seconds / 3600.0),
            "max_appliance_power_w": float(arrays["appliance_w"].max()),
            "source_shard_count": len(source_shards[cycle_id]),
            "source_shards": json.dumps(source_shards[cycle_id]),
        })

    inventory = pd.DataFrame(records)
    inventory_path = output_dir / "real_cycle_library.csv"
    inventory.to_csv(inventory_path, index=False)
    manifest = {
        "protocol": "real_cycle_library_v1",
        "partition": args.partition,
        "cycle_count": int(len(inventory)),
        "sample_seconds": int(aligned_manifest["sample_seconds"]),
        "source_aligned_manifest": str(aligned_manifest_path.resolve()),
        "source_aligned_manifest_sha256": _sha256(aligned_manifest_path),
        "source_coverage": str(coverage_path.resolve()),
        "source_coverage_sha256": _sha256(coverage_path),
        "forbid_validation_and_test": args.partition == "train",
        "total_samples": int(inventory["samples"].sum()),
        "total_appliance_energy_wh": float(
            inventory["appliance_energy_wh"].sum()),
    }
    manifest_path = output_dir / "real_cycle_library_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    print(f"[cycle-library] cycles={len(inventory):,}, samples={manifest['total_samples']:,}")
    print(f"[cycle-library] inventory -> {inventory_path}")
    print(f"[cycle-library] manifest  -> {manifest_path}")


if __name__ == "__main__":
    main()
