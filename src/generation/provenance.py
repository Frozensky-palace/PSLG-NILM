"""Provenance helpers: hashes, environment capture and run registry (Phase B2).

The registry (``reports/experiment_registry.csv`` per roadmap B3) accumulates
one row per formal run; writing is append-safe and idempotent per run id.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
from pathlib import Path

REGISTRY_FIELDS = (
    "run_id", "group", "state_library_version", "seed", "ratio",
    "git_commit", "config_hash", "train_status", "best_epoch",
    "val_mae_w", "test_accessed", "output_dir", "updated_utc",
)


def sha256_of_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_config_hash(config: dict) -> str:
    """Hash a config dict with sorted keys so equivalent configs match."""
    canonical = json.dumps(config, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":"))
    return sha256_of_bytes(canonical.encode("utf-8"))


def git_commit(repo_root: Path) -> str | None:
    """Best-effort HEAD lookup; returns None outside a git repository."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root),
            capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_record() -> dict:
    import torch

    return {
        "python": platform.python_version(),
        "numpy": __import__("numpy").__version__,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "platform": platform.platform(),
    }


def append_registry_row(registry_path: Path, row: dict) -> None:
    """Append or replace one registry row keyed by run_id."""
    registry_path = Path(registry_path)
    existing: list[dict] = []
    if registry_path.exists():
        with open(registry_path, encoding="utf-8", newline="") as stream:
            existing = [item for item in csv.DictReader(stream)
                        if item.get("run_id") != row.get("run_id")]
    fieldnames = list(REGISTRY_FIELDS)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames,
                                extrasaction="ignore")
        writer.writeheader()
        for item in existing:
            writer.writerow(item)
        writer.writerow({name: row.get(name, "") for name in fieldnames})


def utc_now_string() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
