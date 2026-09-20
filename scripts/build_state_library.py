"""Build a traceable train-only state and transition library."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(log_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else log_root / path


def build_state_library_from_manifest(*, run_manifest_path: Path,
                                      cluster_tag: str,
                                      cycle_library_dir: Path,
                                      segment_map_path: Path,
                                      output_dir: Path, run_id: str,
                                      sample_seconds: int = 6) -> dict:
    """Build one state library from a completed clustering run manifest."""
    log_root = run_manifest_path.parent
    cycle_library_dir = Path(cycle_library_dir)
    segment_map_path = Path(segment_map_path)
    output_dir = Path(output_dir)
    with open(run_manifest_path, encoding="utf-8") as stream:
        run_manifest = json.load(stream)
    result = run_manifest["steps"]["time_clustering"]["results"].get(cluster_tag)
    if result is None:
        raise SystemExit(f"cluster tag not found: {cluster_tag}")
    artifacts = result["artifacts"]
    blocks_path = _resolve(log_root, artifacts["blocks"])
    state_sequences_path = _resolve(log_root, artifacts["state_sequences"])
    with open(blocks_path, encoding="utf-8") as stream:
        blocks = json.load(stream)
    with open(state_sequences_path, encoding="utf-8") as stream:
        sequences = json.load(stream)

    mapping = pd.read_csv(segment_map_path).set_index("csv_idx")
    if set(mapping["partition"].astype(str)) != {"train"}:
        raise SystemExit("state library source must contain train only")
    cycle_cache = {}

    def load_cycle(csv_idx: int) -> dict[str, np.ndarray]:
        if csv_idx not in cycle_cache:
            row = mapping.loc[csv_idx]
            path = cycle_library_dir / row["source_npz"]
            with np.load(path) as data:
                cycle_cache[csv_idx] = {
                    name: data[name].copy()
                    for name in ("timestamp", "appliance_w")
                }
        return cycle_cache[csv_idx]

    by_cycle = {}
    for block in blocks:
        by_cycle.setdefault(int(block["csv_idx"]), []).append(block)
    for values in by_cycle.values():
        values.sort(key=lambda item: int(item["start"]))

    transition_counts: Counter[tuple[int, int]] = Counter()
    records = []
    flat_power = []
    flat_timestamp = []
    offsets = [0]
    coverage_ok = True
    for csv_idx in sorted(by_cycle):
        source = mapping.loc[csv_idx]
        cycle = load_cycle(csv_idx)
        cycle_blocks = by_cycle[csv_idx]
        coverage_ok &= (
            int(cycle_blocks[0]["start"]) == 0
            and int(cycle_blocks[-1]["end"]) == len(cycle["timestamp"])
            and sum(int(item["length_samples"]) for item in cycle_blocks)
            == len(cycle["timestamp"])
        )
        for position, block in enumerate(cycle_blocks):
            start, end = int(block["start"]), int(block["end"])
            power = cycle["appliance_w"][start:end].astype(np.float32)
            timestamps = cycle["timestamp"][start:end].astype(np.int64)
            if len(power) != int(block["length_samples"]):
                raise RuntimeError(f"block {block['block_id']} length mismatch")
            previous_label = (
                int(cycle_blocks[position - 1]["state_label"])
                if position > 0 else None)
            next_label = (
                int(cycle_blocks[position + 1]["state_label"])
                if position + 1 < len(cycle_blocks) else None)
            if next_label is not None:
                transition_counts[(int(block["state_label"]), next_label)] += 1
            diff = np.diff(power.astype(np.float64))
            state_label = int(block["state_label"])
            records.append({
                "state_block_id": int(block["block_id"]),
                "state_scope_id": (
                    f"{run_id}:{cluster_tag}:state_{state_label}"),
                "state_label": state_label,
                "dataset": "UK-DALE",
                "house_id": 1,
                "device_instance": 1,
                "appliance": "washing_machine",
                "source_partition": "train",
                "cycle_id": str(source["cycle_id"]),
                "csv_idx": csv_idx,
                "start_sample": start,
                "end_sample_exclusive": end,
                "start_unix": int(timestamps[0]),
                "end_unix": int(timestamps[-1]),
                "duration_seconds": len(power) * sample_seconds,
                "samples": len(power),
                "mean_power_w": float(power.mean()),
                "std_power_w": float(power.std()),
                "rms_power_w": float(np.sqrt(np.mean(power.astype(np.float64) ** 2))),
                "min_power_w": float(power.min()),
                "max_power_w": float(power.max()),
                "energy_wh": float(power.astype(np.float64).sum()
                                   * sample_seconds / 3600.0),
                "start_power_w": float(power[0]),
                "end_power_w": float(power[-1]),
                "mean_abs_slope_w_per_sample": (
                    float(np.mean(np.abs(diff))) if len(diff) else 0.0),
                "previous_state_label": previous_label,
                "next_state_label": next_label,
                "n_source_segments": int(block["n_segments"]),
                "source_feature_json": json.dumps(block["feature"]),
                "waveform_offset_start": offsets[-1],
                "waveform_offset_end": offsets[-1] + len(power),
            })
            flat_power.append(power)
            flat_timestamp.append(timestamps)
            offsets.append(offsets[-1] + len(power))

    output_dir.mkdir(parents=True, exist_ok=True)
    inventory = pd.DataFrame(records).sort_values("state_block_id")
    inventory_path = output_dir / "state_inventory.csv"
    inventory.to_csv(inventory_path, index=False)
    waveform_path = output_dir / "state_waveforms.npz"
    np.savez_compressed(
        waveform_path,
        power_w=np.concatenate(flat_power),
        timestamp=np.concatenate(flat_timestamp),
        offsets=np.asarray(offsets, dtype=np.int64),
        state_block_id=inventory["state_block_id"].to_numpy(dtype=np.int64),
    )

    state_summary = {}
    for label, group in inventory.groupby("state_label"):
        state_summary[str(label)] = {
            "blocks": int(len(group)),
            "cycles": int(group["cycle_id"].nunique()),
            "median_duration_seconds": float(group["duration_seconds"].median()),
            "median_power_w": float(group["mean_power_w"].median()),
            "median_energy_wh": float(group["energy_wh"].median()),
        }
    graph = {
        "nodes": state_summary,
        "edges": [
            {"from_state": left, "to_state": right, "count": count}
            for (left, right), count in sorted(transition_counts.items())
        ],
    }
    graph_path = output_dir / "transition_graph.json"
    with open(graph_path, "w", encoding="utf-8") as stream:
        json.dump(graph, stream, ensure_ascii=False, indent=2)

    summary = {
        "protocol": "state_library_v1",
        "status": "pilot_physical_stats_not_final",
        "run_id": run_id,
        "cluster_tag": cluster_tag,
        "source_partition": "train",
        "state_blocks": int(len(inventory)),
        "cycles": int(inventory["cycle_id"].nunique()),
        "states": state_summary,
        "transition_edges": int(len(transition_counts)),
        "all_records_train": bool((inventory["source_partition"] == "train").all()),
        "cycle_coverage_exact": bool(coverage_ok),
        "run_manifest_sha256": _sha256(run_manifest_path),
        "blocks_sha256": _sha256(blocks_path),
        "segment_map_sha256": _sha256(segment_map_path),
        "inventory_sha256": _sha256(inventory_path),
        "waveforms_sha256": _sha256(waveform_path),
    }
    summary_path = output_dir / "state_library_summary.json"
    with open(summary_path, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(f"[state-library] blocks={len(inventory):,}, states={len(state_summary)}, "
          f"cycles={inventory['cycle_id'].nunique():,}")
    print(f"[state-library] exact cycle coverage={coverage_ok}, train only=True")
    print(f"[state-library] inventory -> {inventory_path}")
    print(f"[state-library] graph     -> {graph_path}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--cluster-tag", required=True,
                    help="merged tag, for example kmeans_k3_merged")
    ap.add_argument("--cycle-library-dir", required=True)
    ap.add_argument("--segment-map", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--sample-seconds", type=int, default=6)
    args = ap.parse_args()
    build_state_library_from_manifest(
        run_manifest_path=Path("log") / args.run_id / "run_manifest.json",
        cluster_tag=args.cluster_tag,
        cycle_library_dir=Path(args.cycle_library_dir),
        segment_map_path=Path(args.segment_map),
        output_dir=Path(args.output_dir),
        run_id=args.run_id,
        sample_seconds=args.sample_seconds,
    )


if __name__ == "__main__":
    main()
