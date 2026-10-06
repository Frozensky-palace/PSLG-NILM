"""Export leakage-free UK-DALE mains/washer/background research partitions.

The output is sharded so multi-year data can be prepared on an ordinary local
machine.  Each NPZ contains aligned timestamp, mains, appliance, signed
background, clipped background and continuous-segment arrays.
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

from src.data.aligned_partitions import (  # noqa: E402
    PARTITIONS,
    aligned_power_frame,
    align_to_grid,
    cycle_grid_counts,
    partition_intervals,
    save_aligned_npz,
)


def _power_column(frame: pd.DataFrame):
    for column in frame.columns:
        labels = column if isinstance(column, tuple) else (column,)
        normalized = {str(item).strip().lower() for item in labels}
        if "power" in normalized and "active" in normalized:
            return column
    for column in frame.columns:
        if "power" in str(column).lower():
            return column
    raise ValueError(f"no active-power column in {list(frame.columns)!r}")


def _read_series(store: pd.HDFStore, key: str, start_unix: int,
                 end_unix: int) -> pd.Series:
    start = pd.Timestamp(start_unix, unit="s", tz="UTC").isoformat()
    end = pd.Timestamp(end_unix, unit="s", tz="UTC").isoformat()
    frame = store.select(
        key, where=f"index>=Timestamp('{start}') & index<Timestamp('{end}')")
    if frame.empty:
        return pd.Series(dtype=np.float64, index=pd.DatetimeIndex([], tz="UTC"))
    return frame[_power_column(frame)].astype(np.float64)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_cycle_samples(start: int, end: int, sample_seconds: int) -> int:
    first = align_to_grid(start, sample_seconds, direction="ceil")
    last = align_to_grid(end, sample_seconds, direction="floor")
    return max(0, (last - first) // sample_seconds + 1)


def _apply_quality_decision(coverage: pd.DataFrame, manifest: dict) -> None:
    """Add the model-independent complete-coverage eligibility decision."""
    coverage["analysis_eligible"] = coverage["complete_coverage"].astype(bool)
    coverage["analysis_exclusion_reason"] = np.where(
        coverage["analysis_eligible"], "", "incomplete_aligned_mains_coverage")
    manifest["alignment_quality_rule"] = {
        "require_complete_cycle_coverage": True,
        "rule_uses_model_results": False,
        "excluded_cycle_ids": coverage.loc[
            ~coverage["analysis_eligible"], "cycle_id"].tolist(),
        "usable_cycle_counts": {
            partition: int(part["analysis_eligible"].sum())
            for partition, part in coverage.groupby("partition")
        },
    }


def _finalize_existing_quality(output_dir: Path) -> None:
    """Update quality fields without reading or rewriting the large shards."""
    coverage_path = output_dir / "cycle_alignment_coverage.csv"
    manifest_path = output_dir / "aligned_partition_manifest.json"
    if not coverage_path.exists() or not manifest_path.exists():
        raise SystemExit("existing coverage CSV and manifest are required")
    coverage = pd.read_csv(coverage_path)
    with open(manifest_path, encoding="utf-8") as stream:
        manifest = json.load(stream)
    _apply_quality_decision(coverage, manifest)
    coverage.to_csv(coverage_path, index=False)
    with open(manifest_path, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    print(f"[aligned] finalized quality decision -> {output_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5", default="datasets/ukdale/ukdale.h5")
    ap.add_argument("--inventory", required=True,
                    help="cycle_inventory_with_split.csv")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--branch-key", default="/building1/elec/meter5")
    ap.add_argument("--mains-key", default="/building1/elec/meter54")
    ap.add_argument("--sample-seconds", type=int, default=6)
    ap.add_argument("--max-interpolate-seconds", type=int, default=150)
    ap.add_argument("--shard-days", type=int, default=31)
    ap.add_argument("--partitions", nargs="+", choices=PARTITIONS,
                    default=list(PARTITIONS))
    ap.add_argument("--uncompressed", action="store_true")
    ap.add_argument("--finalize-existing-only", action="store_true",
                    help="only add quality eligibility fields to existing outputs")
    args = ap.parse_args()

    h5_path = Path(args.h5)
    inventory_path = Path(args.inventory)
    output_dir = Path(args.output_dir)
    if args.finalize_existing_only:
        _finalize_existing_quality(output_dir)
        return
    if not h5_path.exists():
        raise SystemExit(f"HDF5 file not found: {h5_path}")
    if not inventory_path.exists():
        raise SystemExit(f"inventory not found: {inventory_path}")
    if args.shard_days <= 0:
        raise SystemExit("--shard-days must be positive")

    inventory = pd.read_csv(inventory_path)
    intervals = partition_intervals(inventory)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_seconds = args.shard_days * 86400
    padding = args.max_interpolate_seconds + args.sample_seconds
    manifest: dict = {
        "protocol": "aligned_research_partitions_v1",
        "source_h5": str(h5_path.resolve()),
        "source_inventory": str(inventory_path.resolve()),
        "source_inventory_sha256": _sha256(inventory_path),
        "branch_key": args.branch_key,
        "mains_key": args.mains_key,
        "sample_seconds": args.sample_seconds,
        "max_interpolate_seconds": args.max_interpolate_seconds,
        "background_definition": {
            "signed": "mains_w - appliance_w; retained for exact identity audit",
            "clipped": "max(0, mains_w - appliance_w); intended for composition",
        },
        "partitions": {},
    }
    coverage_rows: list[dict] = []

    with pd.HDFStore(h5_path, mode="r") as store:
        for partition in args.partitions:
            interval_start, interval_end = intervals[partition]
            grid_start = align_to_grid(
                interval_start, args.sample_seconds, direction="ceil")
            grid_end = align_to_grid(
                interval_end, args.sample_seconds, direction="ceil")
            cycles = inventory[inventory["partition"] == partition].copy()
            observed_counts = {str(cid): 0 for cid in cycles["cycle_id"]}
            shard_items = []
            totals = {
                "grid_rows": 0,
                "valid_rows": 0,
                "missing_rows": 0,
                "continuous_segments": 0,
                "negative_background_rows": 0,
                "negative_background_energy_wh": 0.0,
                "max_signed_identity_error_w": 0.0,
            }

            shard_start = grid_start
            shard_index = 0
            while shard_start < grid_end:
                shard_end = min(shard_start + shard_seconds, grid_end)
                read_start = shard_start - padding
                read_end = shard_end + padding
                branch = _read_series(
                    store, args.branch_key, read_start, read_end)
                mains = _read_series(
                    store, args.mains_key, read_start, read_end)
                frame, quality = aligned_power_frame(
                    branch, mains, start_unix=shard_start,
                    end_unix=shard_end, sample_seconds=args.sample_seconds,
                    max_interpolate_seconds=args.max_interpolate_seconds)
                filename = f"shard_{shard_index:03d}_{shard_start}_{shard_end}.npz"
                relative_path = Path(partition) / filename
                save_aligned_npz(
                    output_dir / relative_path, frame,
                    compressed=not args.uncompressed)

                timestamps = frame["timestamp"].to_numpy(dtype=np.int64)
                for cycle_id, count in cycle_grid_counts(
                        cycles, timestamps,
                        sample_seconds=args.sample_seconds).items():
                    observed_counts[cycle_id] += count
                for key in ("grid_rows", "valid_rows", "missing_rows",
                            "continuous_segments", "negative_background_rows"):
                    totals[key] += quality[key]
                totals["negative_background_energy_wh"] += quality[
                    "negative_background_energy_wh"]
                totals["max_signed_identity_error_w"] = max(
                    totals["max_signed_identity_error_w"],
                    quality["max_signed_identity_error_w"])
                shard_items.append({
                    "path": relative_path.as_posix(),
                    "sha256": _sha256(output_dir / relative_path),
                    **quality,
                })
                print(
                    f"[aligned] {partition} shard {shard_index:03d}: "
                    f"{quality['valid_rows']:,}/{quality['grid_rows']:,} valid",
                    flush=True)
                shard_start = shard_end
                shard_index += 1

            totals["valid_fraction"] = (
                totals["valid_rows"] / totals["grid_rows"]
                if totals["grid_rows"] else 0.0)
            totals["negative_background_fraction"] = (
                totals["negative_background_rows"] / totals["valid_rows"]
                if totals["valid_rows"] else 0.0)
            manifest["partitions"][partition] = {
                "interval_start_unix": interval_start,
                "interval_end_unix_exclusive": interval_end,
                "grid_start_unix": grid_start,
                "grid_end_unix_exclusive": grid_end,
                "cycle_count": int(len(cycles)),
                "totals": totals,
                "shards": shard_items,
            }

            for row in cycles.itertuples(index=False):
                expected = _expected_cycle_samples(
                    int(row.start_unix), int(row.end_unix),
                    args.sample_seconds)
                observed = observed_counts[str(row.cycle_id)]
                coverage_rows.append({
                    "cycle_id": row.cycle_id,
                    "partition": partition,
                    "start_unix": int(row.start_unix),
                    "end_unix": int(row.end_unix),
                    "expected_grid_samples": expected,
                    "observed_aligned_samples": observed,
                    "coverage_fraction": observed / expected if expected else 0.0,
                    "complete_coverage": bool(expected and observed == expected),
                })

    coverage = pd.DataFrame(coverage_rows)
    _apply_quality_decision(coverage, manifest)
    coverage_path = output_dir / "cycle_alignment_coverage.csv"
    coverage.to_csv(coverage_path, index=False)
    coverage_summary = {}
    for partition, part in coverage.groupby("partition"):
        coverage_summary[partition] = {
            "cycles": int(len(part)),
            "complete_cycles": int(part["complete_coverage"].sum()),
            "minimum_coverage_fraction": float(part["coverage_fraction"].min()),
            "mean_coverage_fraction": float(part["coverage_fraction"].mean()),
        }
    manifest["cycle_coverage"] = coverage_summary
    manifest["all_cycles_complete"] = bool(coverage["complete_coverage"].all())
    manifest_path = output_dir / "aligned_partition_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    print(f"[aligned] coverage -> {coverage_path}")
    print(f"[aligned] manifest -> {manifest_path}")
    print(f"[aligned] all cycles complete: {manifest['all_cycles_complete']}")


if __name__ == "__main__":
    main()
