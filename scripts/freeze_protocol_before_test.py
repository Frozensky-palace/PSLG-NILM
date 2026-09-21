"""Freeze validation-selected runs before any final-test access.

This command never reads test data.  It hashes protocol configs, selected run
configs/checkpoints/metrics and verifies that every training summary records
``test_accessed: false``.  The explicit confirmation flag prevents an
accidental early freeze while validation work is still changing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_file(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def first_existing(run_dir: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        path = run_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"{run_dir}: none of {names} exists")


def inspect_run(run_dir: Path) -> dict:
    summary_path = first_existing(
        run_dir, ("training_summary.json", "generation_summary.json",
                  "state_discovery_summary.json"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("test_accessed") is not False:
        raise ValueError(
            f"{summary_path}: test_accessed must be explicitly false")
    config = first_existing(run_dir, ("config.json", "config.yaml",
                                      "config_effective.yaml"))
    metrics = first_existing(run_dir, ("validation_metrics.json",
                                       "metrics.json"))
    checkpoint = first_existing(
        run_dir, ("best_checkpoint.pt", "best_checkpoint.keras",
                  "checkpoint_sha256.txt"))
    return {
        "run_dir": str(run_dir.resolve()),
        "group": summary.get("arm") or summary.get("group")
        or summary.get("method") or run_dir.name,
        "summary": record_file(summary_path),
        "config": record_file(config),
        "validation_metrics": record_file(metrics),
        "checkpoint": record_file(checkpoint),
        "test_accessed": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", action="append", required=True,
                    help="selected validation run; repeat for every B0-B5 group")
    ap.add_argument("--protocol-config", action="append", required=True)
    ap.add_argument("--no-test-audit", required=True,
                    help="JSON produced by audit_no_test_access.py")
    ap.add_argument("--expected-group", action="append", default=[])
    ap.add_argument("--output", required=True,
                    help="output JSON; a sibling Markdown file is also written")
    ap.add_argument("--i-confirm-validation-complete", action="store_true")
    args = ap.parse_args()
    if not args.i_confirm_validation_complete:
        raise SystemExit(
            "refusing to freeze: add --i-confirm-validation-complete only "
            "after all validation choices are final")

    audit_path = Path(args.no_test_audit)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("all_clean") is not True:
        raise SystemExit("no-test audit is missing or not clean")
    runs = [inspect_run(Path(path)) for path in args.run_dir]
    selected_groups = {str(run["group"]) for run in runs}
    missing_groups = sorted(set(args.expected_group) - selected_groups)
    if missing_groups:
        raise SystemExit(f"selected runs are missing groups: {missing_groups}")

    repo_root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True,
        text=True, check=True).stdout.strip()
    payload = {
        "protocol": "protocol_freeze_before_test_v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "test_accessed": False,
        "no_test_audit": record_file(audit_path),
        "protocol_configs": [record_file(Path(path))
                             for path in args.protocol_config],
        "selected_runs": runs,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    lines = [
        "# Protocol freeze before final test", "",
        f"- Frozen UTC: `{payload['frozen_at_utc']}`",
        f"- Git commit: `{commit}`",
        "- Test accessed before freeze: `false`", "",
        "## Selected validation runs", "",
    ]
    lines += [f"- `{run['group']}`: `{run['run_dir']}`"
              for run in runs]
    lines += ["", "After this file is created, changes to code, configs, "
              "checkpoints, thresholds or selected runs require a new freeze "
              "version and a written reason.\n"]
    output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[freeze] {len(runs)} selected runs -> {output}")


if __name__ == "__main__":
    main()
