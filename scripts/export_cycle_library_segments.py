"""Export a real-cycle library as traceable CSV inputs for state discovery."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--library-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    library_dir = Path(args.library_dir)
    output_dir = Path(args.output_dir)
    inventory_path = library_dir / "real_cycle_library.csv"
    if not inventory_path.exists():
        raise SystemExit(f"library inventory not found: {inventory_path}")
    inventory = pd.read_csv(inventory_path)
    if set(inventory["partition"].astype(str)) != {"train"}:
        raise SystemExit("state-discovery segment export must contain train only")

    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for csv_idx, row in enumerate(inventory.itertuples(index=False)):
        source = library_dir / row.path
        with np.load(source) as data:
            timestamps = data["timestamp"].astype(np.int64)
            power = data["appliance_w"].astype(np.float32)
        filename = f"cycle_{csv_idx:04d}.csv"
        destination = output_dir / filename
        pd.DataFrame({
            "timestamp": timestamps,
            "power": power,
            "datetime": pd.to_datetime(timestamps, unit="s", utc=True),
            "cycle_id": str(row.cycle_id),
        }).to_csv(destination, index=False)
        records.append({
            "csv_idx": csv_idx,
            "filename": filename,
            "cycle_id": str(row.cycle_id),
            "partition": str(row.partition),
            "source_npz": str(row.path),
            "start_unix": int(row.start_unix),
            "end_unix": int(row.end_unix),
            "samples": int(row.samples),
        })

    mapping = pd.DataFrame(records)
    mapping_path = output_dir / "segment_source_map.csv"
    mapping.to_csv(mapping_path, index=False)
    manifest = {
        "protocol": "state_discovery_cycle_csv_v1",
        "source_library": str(library_dir.resolve()),
        "partition": "train",
        "cycle_count": int(len(mapping)),
        "total_samples": int(mapping["samples"].sum()),
        "filename_order_defines_csv_idx": True,
        "mapping": mapping_path.name,
    }
    manifest_path = output_dir / "segment_export_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    print(f"[segment-export] cycles={len(mapping):,}, samples={manifest['total_samples']:,}")
    print(f"[segment-export] directory -> {output_dir}")
    print(f"[segment-export] mapping   -> {mapping_path}")


if __name__ == "__main__":
    main()
