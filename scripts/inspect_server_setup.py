"""Lightweight server inventory; safe to run before GPU job configuration.

Uses only the standard library. Does not import ML frameworks, read waveforms,
install packages, query GPUs, submit jobs, or edit data/environment settings.
Package metadata is NOT evidence that imports or GPU execution work. Reports
are written only when --output-dir is explicitly supplied, with unique names.
"""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
from datetime import datetime, timezone


PACKAGES = (
    "numpy", "pandas", "scipy", "scikit-learn", "PyYAML", "torch",
    "tensorflow", "tensorflow-cpu", "tensorflow-gpu", "nilmtk", "pytest",
)
ENTRYPOINTS = (
    "scripts/train_nilm.py",
    "scripts/prepare_nilm_b0_b2_inputs.py",
    "scripts/train_state_discovery.py",
    "src/nilm/seq2point.py",
    "src/generation/primitive_composition.py",
)
DATA_CANDIDATES = (
    "house_1/labels.dat", "house_1/channel_1.dat", "house_1/channel_5.dat",
    "ukdale.h5",
)


def git_value(root, *args):
    try:
        result = subprocess.run(
            ["git", "--no-optional-locks", *args], cwd=str(root),
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return {"status": "ok", "value": result.stdout.strip()}
        return {"status": "error", "returncode": result.returncode}
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}
    except OSError as error:
        return {"status": "unavailable", "error_type": type(error).__name__}


def package_versions():
    result = {}
    for name in PACKAGES:
        try:
            result[name] = {"status": "installed", "version": metadata.version(name)}
        except metadata.PackageNotFoundError:
            result[name] = {"status": "not_installed"}
        except Exception as error:
            result[name] = {"status": "metadata_error", "error_type": type(error).__name__}
    return result


def path_info(path):
    path = Path(path)
    result = {"path": str(path)}
    try:
        info = path.stat()
        result["status"] = "exists"
        result["kind"] = (
            "file" if stat.S_ISREG(info.st_mode)
            else "directory" if stat.S_ISDIR(info.st_mode) else "other"
        )
        if result["kind"] == "file":
            result["bytes"] = info.st_size
    except FileNotFoundError:
        result["status"] = "missing"
    except OSError as error:
        result.update(status="inaccessible", error_type=type(error).__name__)
    return result


def inspect_setup(repo_root, data_roots):
    root = Path(repo_root).expanduser().resolve()
    data = []
    for value in data_roots:
        data_root = Path(value).expanduser().absolute()
        data.append({
            "root": path_info(data_root),
            "candidates": [path_info(data_root / name) for name in DATA_CANDIDATES],
        })
    return {
        "schema_version": 1,
        "status": "inventory_only_not_training_readiness",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {"executable": sys.executable, "version": platform.python_version()},
        "environment": {key: os.environ[key] for key in (
            "CONDA_DEFAULT_ENV", "CONDA_PREFIX", "SLURM_JOB_ID", "SLURM_JOB_PARTITION",
        ) if key in os.environ},
        "packages": package_versions(),
        "repo": {
            "root": str(root),
            "commit": git_value(root, "rev-parse", "HEAD"),
            "branch": git_value(root, "branch", "--show-current"),
            "shallow": git_value(root, "rev-parse", "--is-shallow-repository"),
            "tracked_changes": git_value(root, "status", "--short", "--untracked-files=no"),
            "entrypoints": {name: path_info(root / name) for name in ENTRYPOINTS},
        },
        "command_paths": {name: shutil.which(name) for name in ("sbatch", "sinfo", "nvidia-smi")},
        "data_candidates": data,
        "limitations": [
            "Package versions only; framework imports and GPU usability not checked.",
            "File metadata only; channel identities, contents, splits and leakage not checked.",
            "Missing candidate paths do not prove the dataset is absent elsewhere.",
            "Tracked changes only; untracked files are not listed. No training approval implied.",
        ],
    }


def save_report(report, output_dir):
    directory = Path(output_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=f"setup-{stamp}-", suffix=".json",
        dir=str(directory), delete=False,
    ) as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        return Path(stream.name).resolve()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-root", type=Path, action="append", default=[],
                        help="Candidate data directory; repeatable, no recursive scanning")
    parser.add_argument("--output-dir", type=Path,
                        help="Optional directory for a new uniquely named JSON report")
    args = parser.parse_args(argv)
    report = inspect_setup(args.repo_root, args.data_root)
    if args.output_dir:
        path = save_report(report, args.output_dir)
        print(f"[inventory] saved: {path}")
        print("[inventory] Metadata only; imports, data validity and GPUs remain unchecked.")
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
