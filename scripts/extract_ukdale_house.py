"""Split one UK-DALE building into one HDF5 file per physical meter.

Each output file contains a pandas/PyTables table at ``/data`` and preserves
the source table, index, column metadata, and compression without loading a
whole channel into RAM.  A JSON manifest maps physical meters to appliances.
The requested appliance can additionally be streamed to a canonical
``timestamp,power`` CSV for the PSLG-NILM pipeline.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import tables

from prepare_dataset_series import _first_power_col, find_meter


def _meter_number(key: str) -> int:
    match = re.search(r"meter(\d+)$", key)
    if not match:
        raise ValueError(f"not a meter key: {key}")
    return int(match.group(1))


def _appliance_map(source: str, building: int) -> tuple[dict, dict]:
    from nilmtk import DataSet

    by_meter: dict[int, list[dict]] = defaultdict(list)
    meter_meta: dict[int, dict] = {}
    ds = DataSet(source)
    try:
        elec = ds.buildings[building].elec
        for meter in elec.meters:
            meter_meta[int(meter.instance())] = dict(meter.metadata or {})
        for appliance in elec.appliances:
            md = dict(appliance.metadata or {})
            compact = {
                "type": md.get("type"),
                "original_name": md.get("original_name"),
                "instance": md.get("instance"),
                "dates_active": md.get("dates_active"),
            }
            for meter_no in md.get("meters", []):
                by_meter[int(meter_no)].append(compact)
    finally:
        ds.close()
    return dict(by_meter), meter_meta


def _copy_channel(source: str, source_key: str, destination: Path) -> None:
    temp = destination.with_suffix(destination.suffix + ".tmp")
    if temp.exists():
        temp.unlink()
    with tables.open_file(source, mode="r") as src, \
            tables.open_file(temp, mode="w", title=source_key) as dst:
        src.copy_node(source_key, newparent=dst.root, newname="data", recursive=True)
        dst.flush()
    os.replace(temp, destination)


def _valid_existing_channel(path: Path, expected_rows: int) -> bool:
    if not path.exists():
        return False
    try:
        with pd.HDFStore(path, mode="r") as store:
            return "/data" in store.keys() and store.get_storer("/data").nrows == expected_rows
    except Exception:
        return False


def _export_appliance_csv(source: str, building: int, appliance: str,
                          allow_washer_dryer: bool, destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".tmp")
    if temp.exists():
        temp.unlink()

    with pd.HDFStore(source, mode="r") as store:
        key, metadata = find_meter(
            store, building, appliance,
            allow_washer_dryer=allow_washer_dryer,
        )
        if key is None:
            raise ValueError(f"no appliance matching {appliance!r} in building {building}")

        count = 0
        with open(temp, "w", encoding="utf-8", newline="") as stream:
            stream.write("timestamp,power\n")
            for frame in store.select(key, chunksize=1_000_000):
                column = _first_power_col(frame)
                timestamps = frame.index.view(np.int64) // 10**9
                chunk = pd.DataFrame({
                    "timestamp": timestamps,
                    "power": frame[column].to_numpy(dtype=np.float64),
                })
                chunk.to_csv(stream, index=False, header=False)
                count += len(chunk)
                print(f"[house-export] appliance CSV: {count:,} rows", flush=True)
    os.replace(temp, destination)
    return {"meter_key": key, "rows": count, "metadata": metadata}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="datasets/ukdale/ukdale.h5")
    ap.add_argument("--building", type=int, default=1)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--appliance", default="washing machine")
    ap.add_argument("--allow-washer-dryer", action="store_true")
    ap.add_argument("--appliance-csv", default="input/ukdale_house1_washing_machine.csv")
    args = ap.parse_args()

    source = os.path.abspath(args.source)
    if not os.path.exists(source):
        raise SystemExit(f"source not found: {source}")
    out_dir = Path(args.out_dir or f"datasets/ukdale/house{args.building}")
    out_dir.mkdir(parents=True, exist_ok=True)

    appliances, meter_meta = _appliance_map(source, args.building)
    manifest = {
        "source": source,
        "building": args.building,
        "format": "Each meterNN.h5 contains the original pandas table at /data.",
        "channels": [],
    }

    prefix = f"/building{args.building}/elec/meter"
    with pd.HDFStore(source, mode="r") as store:
        keys = sorted(
            (key for key in store.keys()
             if key.startswith(prefix) and "/cache/" not in key),
            key=_meter_number,
        )
        for position, key in enumerate(keys, start=1):
            number = _meter_number(key)
            rows = int(store.get_storer(key).nrows)
            first = store.select(key, start=0, stop=1)
            destination = out_dir / f"meter{number:02d}.h5"
            if _valid_existing_channel(destination, rows):
                status = "verified existing"
            else:
                _copy_channel(source, key, destination)
                status = "copied"
            channel = {
                "meter": number,
                "source_key": key,
                "file": destination.name,
                "table_key": "/data",
                "rows": rows,
                "columns": [list(c) if isinstance(c, tuple) else str(c)
                            for c in first.columns],
                "meter_metadata": meter_meta.get(number, {}),
                "appliances": appliances.get(number, []),
            }
            manifest["channels"].append(channel)
            print(f"[house-export] {position:02d}/{len(keys)} meter{number}: "
                  f"{rows:,} rows, {status} -> {destination}", flush=True)

    appliance_info = _export_appliance_csv(
        source, args.building, args.appliance, args.allow_washer_dryer,
        Path(args.appliance_csv),
    )
    manifest["appliance_csv"] = {
        "path": os.path.abspath(args.appliance_csv),
        **appliance_info,
    }
    manifest_path = out_dir / "channels.json"
    with open(manifest_path, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, default=str)
    channel_csv = out_dir / "channels.csv"
    pd.DataFrame([
        {
            "meter": item["meter"],
            "file": item["file"],
            "table_key": item["table_key"],
            "rows": item["rows"],
            "columns": json.dumps(item["columns"], ensure_ascii=False),
            "appliances": "; ".join(
                str(app.get("type") or app.get("original_name") or "")
                for app in item["appliances"]),
            "site_meter": bool(item["meter_metadata"].get("site_meter", False)),
        }
        for item in manifest["channels"]
    ]).to_csv(channel_csv, index=False, encoding="utf-8-sig")
    print(f"[house-export] manifest -> {manifest_path}")
    print(f"[house-export] channel index -> {channel_csv}")


if __name__ == "__main__":
    main()
