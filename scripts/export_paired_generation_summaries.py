"""Expose existing B1/B2 waves to the shared placement interface, without copying."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def export(directory, sample_seconds):
    root = Path(directory)
    if sample_seconds <= 0:
        raise ValueError("sample_seconds must be positive")
    with open(root / "paired_cycle_plan.csv", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("paired plan is empty")
    summaries = {}
    for arm in ("B1", "B2"):
        if (root / arm / "generation_summary.json").exists():
            raise FileExistsError(root / arm / "generation_summary.json")
        records = []
        seen = set()
        for row in rows:
            path = Path(row[f"{arm.lower()}_path"])
            if path.parent != Path(arm) / "cycles" or not (root / path).is_file():
                raise ValueError("unexpected paired waveform path")
            if path.stem in seen:
                raise ValueError("duplicate synthetic cycle ID")
            seen.add(path.stem)
            digest = hashlib.sha256()
            with open(root / path, "rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            fingerprint = digest.hexdigest()
            if fingerprint != row[f"{arm.lower()}_sha256"]:
                raise ValueError("paired waveform hash mismatch")
            with np.load(root / path, allow_pickle=False) as data:
                count = int(row["samples"])
                if (count <= 0 or data["appliance_w"].shape != (count,)
                        or not np.array_equal(data["relative_time_s"], np.arange(count) * sample_seconds)
                        or not np.isfinite(data["appliance_w"]).all()):
                    raise ValueError("paired waveform rate/length mismatch")
            records.append({"synthetic_cycle_id": path.stem,
                            "template_cycle_id": row["template_cycle_id"],
                            "samples": int(row["samples"]),
                            "source_file_sha256": row[f"{arm.lower()}_sha256"]})
        summaries[arm] = {"protocol": "paired_real_baseline_adapter_v1", "generator": arm,
                          "sample_seconds": sample_seconds, "count": len(rows), "records": records}
    # Check both arms before creating either summary.
    for arm, summary in summaries.items():
        with open(root / arm / "generation_summary.json", "x", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-dir", required=True)
    parser.add_argument("--sample-seconds", type=int, required=True)
    args = parser.parse_args()
    export(args.paired_dir, args.sample_seconds)
