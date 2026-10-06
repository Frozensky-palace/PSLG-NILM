"""Verify a transferred bundle against its manifest (roadmap task 12).

Run on the server after upload: checks that every manifest file exists with
the exact byte count and SHA-256, that no manifest file is missing, and exits
non-zero on any mismatch so Slurm jobs never start on a corrupted transfer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True,
                    help="server_transfer_manifest.json produced locally")
    ap.add_argument("--repo-root", default=".",
                    help="repo root on this machine (default: cwd)")
    ap.add_argument("--report", default=None,
                    help="optional path for a JSON verification report")
    args = ap.parse_args()
    root = Path(args.repo_root).resolve()
    with open(args.manifest, encoding="utf-8") as stream:
        manifest = json.load(stream)

    failures = []
    for record in manifest["files"]:
        path = root / record["path"]
        if not path.exists():
            failures.append({"path": record["path"], "problem": "missing"})
            print(f"[verify] MISSING  {record['path']}", flush=True)
            continue
        actual_bytes = path.stat().st_size
        if actual_bytes != record["bytes"]:
            failures.append({
                "path": record["path"], "problem": "size",
                "expected": record["bytes"], "actual": actual_bytes})
            print(f"[verify] SIZE     {record['path']} "
                  f"({actual_bytes} != {record['bytes']})", flush=True)
            continue
        actual_hash = sha256_of_file(path)
        if actual_hash != record["sha256"]:
            failures.append({"path": record["path"], "problem": "sha256"})
            print(f"[verify] SHA256   {record['path']}", flush=True)
            continue
        print(f"[verify] ok       {record['path']}", flush=True)

    report = {
        "manifest": str(args.manifest),
        "repo_root": str(root),
        "exclude_test": manifest.get("exclude_test"),
        "expected_files": manifest["file_count"],
        "verified_files": manifest["file_count"] - len(failures),
        "failures": failures,
        "passed": not failures,
    }
    if args.report:
        Path(args.report).write_text(
            json.dumps(report, indent=2), encoding="utf-8")
    status = "PASSED" if report["passed"] else "FAILED"
    print(f"[verify] {manifest['file_count'] - len(failures)}/"
          f"{manifest['file_count']} files verified -> {status}", flush=True)
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
