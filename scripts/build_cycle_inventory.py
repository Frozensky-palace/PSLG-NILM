"""Create a complete appliance cycle inventory from a canonical CSV."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cycle_inventory import (
    CycleDetector,
    parse_time,
    scan_cycle_csv,
    summarize_inventory,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="timestamp,power CSV")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dataset", default="UK-DALE")
    ap.add_argument("--building", type=int, default=1)
    ap.add_argument("--appliance", default="washing_machine")
    ap.add_argument("--threshold-w", type=float, default=20.0)
    ap.add_argument("--max-inactive-seconds", type=float, default=150.0)
    ap.add_argument("--min-duration-seconds", type=float, default=1800.0)
    ap.add_argument("--sample-seconds", type=float, default=6.0)
    ap.add_argument("--valid-start", default=None,
                    help="earliest cycle time with valid mains; ISO or Unix")
    ap.add_argument("--valid-end", default=None,
                    help="latest cycle time with valid mains; ISO or Unix")
    ap.add_argument("--device-instance-cutover", default=None,
                    help="optional ISO/Unix cutover between physical device instances")
    ap.add_argument("--before-instance", default="1")
    ap.add_argument("--after-instance", default="2")
    ap.add_argument("--chunk-rows", type=int, default=1_000_000)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detector = CycleDetector(
        threshold_w=args.threshold_w,
        max_inactive_seconds=args.max_inactive_seconds,
        min_duration_seconds=args.min_duration_seconds,
        sample_seconds=args.sample_seconds,
        dataset=args.dataset,
        building=args.building,
        appliance=args.appliance,
        valid_start=parse_time(args.valid_start),
        valid_end=parse_time(args.valid_end),
    )
    inventory, scan_metadata = scan_cycle_csv(
        args.input, detector, chunk_rows=args.chunk_rows)
    cutover = parse_time(args.device_instance_cutover)
    if cutover is not None and not inventory.empty:
        inventory["device_instance"] = np.where(
            inventory["start_unix"].astype(float) < cutover,
            args.before_instance,
            args.after_instance,
        )
        scan_metadata["device_instance_cutover_unix"] = cutover
        scan_metadata["device_instance_counts"] = {
            str(key): int(value)
            for key, value in inventory.loc[inventory["eligible"], "device_instance"]
            .value_counts().to_dict().items()
        }
    inventory_path = out_dir / "cycle_inventory.csv"
    summary_path = out_dir / "cycle_inventory_summary.json"
    inventory.to_csv(inventory_path, index=False, encoding="utf-8-sig")
    report = {
        "scan": scan_metadata,
        "inventory": summarize_inventory(inventory),
    }
    with open(summary_path, "w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(f"[cycle-inventory] inventory -> {inventory_path}")
    print(f"[cycle-inventory] summary   -> {summary_path}")
    print(f"[cycle-inventory] eligible={report['inventory']['eligible_cycles']} "
          f"excluded={report['inventory']['excluded_cycles']}")


if __name__ == "__main__":
    main()
