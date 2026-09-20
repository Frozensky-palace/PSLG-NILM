"""Audit manifests and run records for any evidence of test access (guide §4.3).

Scans data manifests for test paths, run directories for recorded
``test_accessed`` flags, and prediction NPZ files for a ``partition``
attribute that claims test. Produces a JSON report and exits non-zero when
any test access is found, so the no-test guarantee stays auditable instead of
being an assertion in prose.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def audit_manifest(manifest_path: Path) -> dict:
    with open(manifest_path, encoding="utf-8") as stream:
        manifest = json.load(stream)
    test_paths = [record["path"] for record in manifest.get("files", [])
                  if "test" in Path(record["path"]).parts]
    return {"kind": "manifest", "path": str(manifest_path),
            "protocol": manifest.get("protocol"),
            "test_path_count": len(test_paths),
            "test_paths": test_paths[:10],
            "clean": not test_paths}


def audit_run(run_dir: Path) -> dict:
    findings = {"kind": "run", "path": str(run_dir), "problems": []}
    summary_paths = list(run_dir.rglob("training_summary.json"))
    for summary_path in summary_paths:
        with open(summary_path, encoding="utf-8") as stream:
            summary = json.load(stream)
        if summary.get("test_accessed") is True:
            findings["problems"].append(
                f"{summary_path}: test_accessed=true")
    for prediction in run_dir.rglob("*predictions*.npz"):
        try:
            import numpy as np

            with np.load(prediction) as data:
                partition = str(data["partition"]) if "partition" in data else ""
            if "test" in partition:
                findings["problems"].append(
                    f"{prediction}: partition={partition}")
        except Exception as error:  # noqa: BLE001 - report unreadable files
            findings["problems"].append(f"{prediction}: unreadable ({error})")
    findings["clean"] = not findings["problems"]
    return findings


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", action="append", default=[])
    ap.add_argument("--run-dir", action="append", default=[])
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    audits = [audit_manifest(Path(path)) for path in args.manifest]
    audits += [audit_run(Path(path)) for path in args.run_dir]
    all_clean = all(item["clean"] for item in audits)
    report = {"all_clean": all_clean, "audits": audits}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                      encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    sys.exit(0 if all_clean else 1)


if __name__ == "__main__":
    main()
