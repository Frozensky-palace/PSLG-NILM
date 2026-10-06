"""Execute one explicitly supplied experiment command with immutable records.

Use inside the intended conda environment. Outputs must be NEW paths; failed
attempts are retained, never silently overwritten. No shell interpolation.
Example: python scripts/run_recorded_command.py --record-dir log/run1/record
  --input config.json --output log/run1/result -- python scripts/example.py ...
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.inspect_server_setup import inspect_setup


def fingerprints(paths):
    result = []
    for raw in paths:
        root = Path(raw).resolve()
        if not root.exists():
            result.append({"path": str(root), "status": "missing"})
            continue
        files = sorted(p for p in root.rglob("*") if p.is_file()) if root.is_dir() else [root]
        for path in files:
            digest = hashlib.sha256()
            with open(path, "rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            result.append({"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    return result


def run_recorded(command, record_dir, inputs=(), outputs=(), require_slurm=False):
    if not command:
        raise ValueError("an explicit command is required")
    if require_slurm and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("run this workload in a Slurm allocation, not on the login node")
    for item in inputs:
        if not Path(item).exists():
            raise FileNotFoundError(item)
    for item in outputs:
        if Path(item).exists():
            raise FileExistsError(f"refusing to reuse experiment output: {item}")
        out_path = Path(item).resolve()
        for source in inputs:
            in_path = Path(source).resolve()
            if out_path == in_path or out_path.is_relative_to(in_path) or in_path.is_relative_to(out_path):
                raise ValueError("input/output paths must not overlap")
    record = Path(record_dir).resolve()
    for item in (*inputs, *outputs):
        path = Path(item).resolve()
        if record == path or record.is_relative_to(path) or path.is_relative_to(record):
            raise ValueError("record directory must not overlap input/output paths")
    record.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    actual = list(command)
    if actual[0] in ("python", "python3"):
        actual[0] = sys.executable
    env = dict(os.environ)
    # Make the inherited hash-derived primitive RNG reproducible across processes.
    env["PYTHONHASHSEED"] = "0"
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    info = {"status": "running", "command": actual, "cwd": str(Path.cwd()),
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "environment": inspect_setup(root, []), "inputs": fingerprints(inputs),
            "forced_child_environment": {"PYTHONHASHSEED": "0"},
            "declared_outputs": list(map(str, outputs))}
    manifest = record / "run_manifest.json"
    def save():
        manifest.write_text(json.dumps(info, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    save()
    started = time.monotonic()
    returncode = 1
    print(f"[recorded-run] log={record / 'execution.log'}", flush=True)
    try:
        with open(record / "execution.log", "x", encoding="utf-8") as stream:
            result = subprocess.run(actual, env=env, stdout=stream, stderr=subprocess.STDOUT)
        returncode = result.returncode
        missing = [str(p) for p in outputs if not Path(p).exists()]
        info.update(returncode=returncode, missing_outputs=missing,
                    status="completed" if returncode == 0 and not missing else "failed")
        if missing and returncode == 0:
            returncode = 1
    except BaseException as error:
        info.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        info["wall_seconds"] = time.monotonic() - started
        info["ended_utc"] = datetime.now(timezone.utc).isoformat()
        try:
            info["outputs"] = fingerprints(outputs)
        except Exception as error:
            info.update(status="failed", output_inventory_error=str(error))
            returncode = 1
        save()
    print(f"[recorded-run] status={info['status']} manifest={manifest}", flush=True)
    return returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-dir", required=True)
    parser.add_argument("--input", action="append", default=[])
    parser.add_argument("--output", action="append", default=[])
    parser.add_argument("--require-slurm", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    sys.exit(run_recorded(command, args.record_dir, args.input, args.output, args.require_slurm))
