"""Collect reproducibility files and write a hash manifest for one run."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(root: Path, *args: str) -> str:
    completed = subprocess.run(["git", *args], cwd=root, capture_output=True,
                               text=True, check=True)
    return completed.stdout.strip()


def copy_if_given(source: str | None, destination: Path) -> None:
    if not source:
        return
    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if source_path.resolve() != destination.resolve():
        shutil.copy2(source_path, destination)


def build_artifact_manifest(run_dir: Path) -> list[dict]:
    records = []
    for path in sorted(item for item in run_dir.rglob("*") if item.is_file()):
        if path.name == "artifact_manifest.json":
            continue
        records.append({
            "path": path.relative_to(run_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--phase", required=True)
    ap.add_argument("--data-manifest")
    ap.add_argument("--environment")
    ap.add_argument("--stdout-log")
    ap.add_argument("--stderr-log")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    repo_root = Path(args.repo_root).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    copy_if_given(args.data_manifest, run_dir / "data_manifest.json")
    copy_if_given(args.environment, run_dir / "environment.json")
    copy_if_given(args.stdout_log, run_dir / "stdout.log")
    copy_if_given(args.stderr_log, run_dir / "stderr.log")

    commit = git_value(repo_root, "rev-parse", "HEAD")
    status = git_value(repo_root, "status", "--short")
    (run_dir / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")
    summary = run_dir / "run_summary.md"
    if not summary.exists():
        summary.write_text(
            f"# Run summary\n\n"
            f"- Phase: `{args.phase}`\n"
            f"- Finalized UTC: `{datetime.now(timezone.utc).isoformat()}`\n"
            f"- Git commit: `{commit}`\n"
            f"- Git tree clean: `{not bool(status)}`\n",
            encoding="utf-8")
    records = build_artifact_manifest(run_dir)
    payload = {
        "protocol": "pslg_run_artifact_manifest_v1",
        "phase": args.phase,
        "git_commit": commit,
        "git_status_short": status,
        "file_count": len(records),
        "files": records,
    }
    (run_dir / "artifact_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(f"[finalize] {len(records)} artifacts -> {run_dir}")


if __name__ == "__main__":
    main()
