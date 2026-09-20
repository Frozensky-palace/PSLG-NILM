"""Capture a reproducible, secret-free server environment record (guide §5.2).

Writes one JSON (and a sibling text log) per run: Python and key package
versions, a whitelist of safe environment variables, Git state, optional
nvidia-smi output and Slurm job identity. Values are filtered through a
variable whitelist so tokens or proxies can never leak into artifacts.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

SAFE_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES", "SLURM_JOB_ID", "SLURM_JOB_NAME",
    "SLURM_PARTITION", "SLURM_NODELIST", "SLURM_SUBMIT_DIR",
    "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE", "CONDA_DEFAULT_ENV",
    "CONDA_PREFIX",
)

KEY_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "tensorflow",
                "torch", "yaml", "matplotlib")


def _run(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        completed = subprocess.run(command, cwd=str(cwd) if cwd else None,
                                   capture_output=True, text=True, timeout=120)
        return completed.stdout.strip() if completed.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def capture(repo_root: Path) -> dict:
    import importlib.metadata as metadata

    packages = {}
    for name in KEY_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    record = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "safe_environment": {key: os.environ[key] for key in SAFE_ENV_KEYS
                             if key in os.environ},
        "git_commit": _run(["git", "rev-parse", "HEAD"], repo_root),
        "git_status_short": _run(["git", "status", "--short"], repo_root),
        "git_tag_points_at_head": _run(
            ["git", "tag", "--points-at", "HEAD"], repo_root),
        "nvidia_smi": _run(["nvidia-smi"]),
        "module_list": _run(["bash", "-lc", "module list 2>&1"]),
        "slurm_job_json": _run(["scontrol", "show", "job",
                                os.environ["SLURM_JOB_ID"]])
        if "SLURM_JOB_ID" in os.environ else None,
    }
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--output", required=True,
                    help="JSON output path, e.g. <artifact>/environment.json")
    args = ap.parse_args()
    record = capture(Path(args.repo_root).resolve())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, ensure_ascii=False),
                      encoding="utf-8")
    text_lines = [f"{key}: {value}" for key, value in record.items()
                  if not isinstance(value, dict)]
    output.with_suffix(".txt").write_text("\n".join(text_lines) + "\n",
                                          encoding="utf-8")
    print(f"[environment] wrote {output}")


if __name__ == "__main__":
    main()
