"""Create a new aligned dataset by pair averaging; never overwrite sources.

Consumes haojun's aligned_partition_manifest.json and cycle coverage CSV.
Defaults to train/validation only. --include-test transforms the historical
test partition too, but does not fit anything or run test-set model selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.pairwise_sampling import coarsen_arrays, expected_count, validate_source_arrays
from src.generation.provenance import git_commit, utc_now_string


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def safe_source(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"source shard escapes input directory: {relative}")
    return path


def source_blocks(source, shards, dt, pair_across_shards):
    """Disjoint logical blocks, with at most one row borrowed from the next file.

    Original segment IDs are file-local under aligned_research_partitions_v1.
    Only a proven continuous file boundary may normalize the borrowed ID.
    Every original file is validated before borrowing or dropping its prefix.
    """
    previous_timestamp = None

    def load(index):
        nonlocal previous_timestamp
        shard = shards[index]
        path = safe_source(source, shard["path"])
        fingerprint = sha256(path)
        if shard.get("sha256") and fingerprint != shard["sha256"]:
            raise ValueError(f"source hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
        validate_source_arrays(arrays, source_seconds=dt)
        ts = arrays["timestamp"]
        if len(ts):
            if previous_timestamp is not None and ts[0] <= previous_timestamp:
                raise ValueError("overlapping or unordered source shards")
            previous_timestamp = ts[-1]
        return arrays, fingerprint

    current = load(0) if shards else None
    consumed_prefix = 0
    for index, shard in enumerate(shards):
        original, fingerprint = current
        following = load(index + 1) if index + 1 < len(shards) else None
        n = len(original["timestamp"])
        arrays = {name: values[consumed_prefix:] for name, values in original.items()}
        physical_rows = np.arange(consumed_prefix, n, dtype=np.int64)
        physical_shards = np.full(len(physical_rows), index, dtype=np.int64)
        boundary_pairs = []
        borrow = bool(pair_across_shards and len(physical_rows) and following is not None
                      and len(following[0]["timestamp"])
                      and arrays["timestamp"][-1] % (2 * dt) == 0
                      and following[0]["timestamp"][0] == arrays["timestamp"][-1] + dt)
        if borrow:
            next_arrays, next_hash = following
            boundary_pairs.append({
                "left_source_shard_index": index, "right_source_shard_index": index + 1,
                "left_source_path": shard["path"], "right_source_path": shards[index + 1]["path"],
                "left_source_sha256": fingerprint, "right_source_sha256": next_hash,
                "left_row": n - 1, "right_row": 0,
                "left_timestamp": int(arrays["timestamp"][-1]),
                "right_timestamp": int(next_arrays["timestamp"][0]),
                "left_original_segment_id": int(arrays["segment_id"][-1]),
                "right_original_segment_id": int(next_arrays["segment_id"][0]),
            })
            arrays = {name: np.concatenate((values, next_arrays[name][:1]))
                      for name, values in arrays.items()}
            arrays["segment_id"][-1] = arrays["segment_id"][-2]
            physical_rows = np.append(physical_rows, 0)
            physical_shards = np.append(physical_shards, index + 1)
        record = {"source_shard_index": index, "source_path": shard["path"],
                  "source_sha256": fingerprint, "source_file_rows": n,
                  "consumed_prefix_rows": consumed_prefix, "borrowed_next_rows": int(borrow),
                  "cross_shard_pair_count": int(borrow), "boundary_pairs": boundary_pairs}
        yield arrays, physical_rows, physical_shards, record
        consumed_prefix = int(borrow)
        current = following


def convert(aligned_dir, output_dir, *, factor=2, include_test=False, pair_across_shards=False):
    source, target = Path(aligned_dir).resolve(), Path(output_dir).resolve()
    if source == target or target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError("input/output must be separate non-nested directories")
    manifest_path = source / "aligned_partition_manifest.json"
    coverage_path = source / "cycle_alignment_coverage.csv"
    original = json.loads(manifest_path.read_text())
    dt = int(original["sample_seconds"])
    if dt != 6:
        raise ValueError(f"this frozen experiment expects 6-second source data, got {dt}")
    if factor not in (1, 2):
        raise ValueError("factor must be 1 or 2")
    if pair_across_shards and (factor != 2 or original.get("protocol") != "aligned_research_partitions_v1"):
        raise ValueError("cross-shard pairing requires factor=2 and aligned_research_partitions_v1")
    coverage = pd.read_csv(coverage_path)
    if coverage["cycle_id"].duplicated().any():
        raise ValueError("duplicate cycle IDs")
    parts = ["train", "validation"] + (["test"] if include_test else [])
    for part in parts:
        if part not in original["partitions"] or not (coverage.partition == part).any():
            raise ValueError(f"missing partition metadata: {part}")
    target.mkdir(parents=True, exist_ok=False)
    audit = {
        "status": "running", "started_utc": utc_now_string(), "git_commit": git_commit(ROOT),
        "source_dir": str(source), "source_manifest_sha256": sha256(manifest_path),
        "source_coverage_sha256": sha256(coverage_path), "factor": factor,
        "source_sample_seconds": dt, "output_sample_seconds": dt * factor,
        "partitions": parts, "test_transformed": include_test,
        "timestamp_convention": "epoch-aligned bin left edge; both source samples must exist",
        "background_clipped_definition": "max(mean(background_signed_w),0) for factor=2",
        "source_shard_boundaries_preserved": not pair_across_shards,
        "pair_across_continuous_storage_shards": pair_across_shards,
        "statistics_scope": "disjoint logical processing blocks; borrowed row is consumed once",
        "row_provenance": "source_left/right_shard_index index the original partition shard list; source_left/right_row are physical file rows",
        "shards": [],
    }
    write_json(target / "downsampling_audit.json", audit)
    try:
        output_manifest = {key: original[key] for key in
                           ("branch_key", "mains_key", "source_inventory_sha256") if key in original}
        output_manifest.update(protocol="aligned_pairwise_sampling_v2" if pair_across_shards else "aligned_pairwise_sampling_v1", sample_seconds=dt * factor,
                               source_aligned_manifest=str(manifest_path), partitions={})
        coverage_results = []
        for part in parts:
            cycles = coverage[coverage.partition == part].copy().sort_values("start_unix")
            intervals = list(zip(cycles.start_unix.astype(int), cycles.end_unix.astype(int)))
            observed = np.zeros(len(cycles), dtype=np.int64)
            shards = []
            blocks = source_blocks(source, original["partitions"][part]["shards"], dt, pair_across_shards)
            for arrays, physical_rows, physical_shards, record in blocks:
                result, stats = coarsen_arrays(arrays, source_seconds=dt, factor=factor,
                                               intervals=intervals)
                for side in ("left", "right"):
                    logical_rows = result[f"source_{side}_row"]
                    result[f"source_{side}_shard_index"] = physical_shards[logical_rows]
                    result[f"source_{side}_row"] = physical_rows[logical_rows]
                actual_cross_pairs = int(np.sum(result["source_left_shard_index"] != result["source_right_shard_index"]))
                if actual_cross_pairs != record["borrowed_next_rows"]:
                    raise ValueError("borrowed source row was not paired exactly once")
                if pair_across_shards:
                    stats["discard_policy"] = "incomplete epoch bin or real gap/segment/partition boundary; no interpolation"
                record.update(partition=part, **stats)
                audit["shards"].append(record)
                if not len(result["timestamp"]):
                    record["output_path"] = None
                    continue
                relative = Path(part) / f"shard_{len(shards):03d}.npz"
                destination = target / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(destination, **result)
                record.update(output_path=relative.as_posix(), output_sha256=sha256(destination))
                shards.append({"path": relative.as_posix(), "sha256": record["output_sha256"],
                               "valid_rows": len(result["timestamp"]),
                               "grid_start_unix": int(result["timestamp"][0]),
                               "grid_end_unix_exclusive": int(result["timestamp"][-1] + dt * factor)})
                for j, (start, end) in enumerate(intervals):
                    observed[j] += max(0, int(np.searchsorted(result["timestamp"], end - (factor - 1) * dt, side="right")
                                       - np.searchsorted(result["timestamp"], start, side="left")))
                print(f"[pair-average] {part}/{Path(record['source_path']).name}: {stats['input_rows']} -> "
                      f"{stats['output_rows']}; discarded={stats['discarded_input_rows']}", flush=True)
            if not shards:
                raise ValueError(f"no usable output rows in {part}")
            part_records = [r for r in audit["shards"] if r["partition"] == part]
            if sum(r["source_file_rows"] for r in part_records) != sum(r["input_rows"] for r in part_records):
                raise ValueError("logical blocks do not account for every original source row")
            cycles["source_analysis_eligible"] = cycles.analysis_eligible.map(
                lambda v: str(v).lower() == "true")
            cycles["expected_grid_samples"] = [expected_count(a, b, dt, factor) for a, b in intervals]
            cycles["source_start_unix"] = cycles.start_unix
            cycles["source_end_unix"] = cycles.end_unix
            # Real-cycle libraries include only bins whose two samples are in
            # the original cycle. Continuous mains retain all valid edge bins.
            cycles["start_unix"] = [-(-int(a) // (dt * factor)) * dt * factor for a, _ in intervals]
            cycles["end_unix"] = [((int(b) - (factor - 1) * dt) // (dt * factor)) * dt * factor for _, b in intervals]
            cycles["observed_aligned_samples"] = observed
            cycles["complete_coverage"] = ((observed == cycles.expected_grid_samples)
                                           & (cycles.expected_grid_samples > 0))
            cycles["analysis_eligible"] = cycles.source_analysis_eligible & cycles.complete_coverage
            cycles["analysis_exclusion_reason"] = np.where(
                cycles.analysis_eligible, "", "source_excluded_or_incomplete_pair_coverage")
            cycles["coverage_fraction"] = observed / np.maximum(cycles.expected_grid_samples, 1)
            coverage_results.append(cycles)
            old = original["partitions"][part]
            output_manifest["partitions"][part] = {
                **{key: old[key] for key in ("interval_start_unix", "interval_end_unix_exclusive") if key in old},
                "shards": shards, "cycle_count": len(cycles),
                "eligible_cycles": int(cycles.analysis_eligible.sum()),
                "totals": {"valid_rows": sum(s["valid_rows"] for s in shards)},
            }
        pd.concat(coverage_results).to_csv(target / "cycle_alignment_coverage.csv", index=False)
        output_manifest["pairwise_protocol"] = audit.copy()
        output_manifest["pairwise_protocol"]["status"] = "completed"
        write_json(target / "aligned_partition_manifest.json", output_manifest)
        audit.update(status="completed", ended_utc=utc_now_string(),
                     output_manifest_sha256=sha256(target / "aligned_partition_manifest.json"))
    except Exception as error:
        audit.update(status="failed", ended_utc=utc_now_string(), error=str(error))
        raise
    finally:
        write_json(target / "downsampling_audit.json", audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--factor", type=int, choices=(1, 2), default=2)
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--pair-across-shards", action="store_true",
                        help="pair across verified continuous storage boundaries within a partition")
    args = parser.parse_args()
    convert(args.aligned_dir, args.output_dir, factor=args.factor, include_test=args.include_test,
            pair_across_shards=args.pair_across_shards)


if __name__ == "__main__":
    main()
