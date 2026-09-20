"""Create a chronological, cycle-level, leakage-checked research split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.research_split import (
    apply_protocol_filters,
    chronological_cycle_split,
    file_sha256,
    leakage_report,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--train-ratio", type=float, default=0.6)
    ap.add_argument("--validation-ratio", type=float, default=0.2)
    ap.add_argument("--test-ratio", type=float, default=0.2)
    ap.add_argument("--device-instance", default=None,
                    help="optional physical device instance to use in this protocol")
    args = ap.parse_args()

    inventory_path = Path(args.inventory)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory = pd.read_csv(inventory_path)
    inventory = apply_protocol_filters(
        inventory, device_instance=args.device_instance)
    assigned = chronological_cycle_split(
        inventory,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
    )
    report = leakage_report(assigned)
    manifest = {
        "protocol": "chronological_cycle_split_v1",
        "source_inventory": str(inventory_path.resolve()),
        "source_inventory_sha256": file_sha256(inventory_path),
        "ratios": {
            "train": args.train_ratio,
            "validation": args.validation_ratio,
            "test": args.test_ratio,
        },
        "filters": {
            "device_instance": args.device_instance,
        },
        "leakage_check_passed": bool(report["passed"]),
        "partition_counts": report["partition_counts"],
        "partition_ranges": report["partition_ranges"],
        "cycles": assigned[[
            "cycle_id", "start_unix", "end_unix", "eligible", "partition"
        ]].to_dict(orient="records"),
    }
    assigned_path = output_dir / "cycle_inventory_with_split.csv"
    manifest_path = output_dir / "partition_manifest.json"
    leakage_path = output_dir / "leakage_report.json"
    assigned.to_csv(assigned_path, index=False, encoding="utf-8-sig")
    with open(manifest_path, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    with open(leakage_path, "w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(f"[research-split] assigned -> {assigned_path}")
    print(f"[research-split] manifest -> {manifest_path}")
    print(f"[research-split] leakage  -> {leakage_path}")
    print(f"[research-split] passed={report['passed']} "
          f"counts={report['partition_counts']}")
    if not report["passed"]:
        raise SystemExit("leakage checks failed; inspect leakage_report.json")


if __name__ == "__main__":
    main()
