"""Check that a completed run directory contains its required artifacts.

A formal run must ship config, commit, logs, history, checkpoints (or their
hashes), predictions and metrics. Missing items are listed and the script
exits non-zero so incomplete runs cannot silently pass as finished.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REQUIRED_ARTIFACTS = (
    "config.json",
    "git_commit.txt",
    "data_manifest.json",
    "environment.json",
    "stdout.log",
    "stderr.log",
    "history.json",
    "metrics.json",
    "run_summary.md",
)

CHECKPOINT_ALTERNATIVES = ("best_checkpoint.pt", "checkpoint_sha256.txt")
PREDICTION_ALTERNATIVES = ("validation_predictions.npz",
                           "prediction_manifest.json")


def collect_missing(run_dir: Path) -> list[str]:
    missing = [name for name in REQUIRED_ARTIFACTS
               if not (run_dir / name).exists()]
    if not any((run_dir / name).exists() for name in CHECKPOINT_ALTERNATIVES):
        missing.append(f"one of {CHECKPOINT_ALTERNATIVES}")
    if not any((run_dir / name).exists() for name in PREDICTION_ALTERNATIVES):
        missing.append(f"one of {PREDICTION_ALTERNATIVES}")
    return missing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--output", default=None,
                    help="optional JSON report path")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"[artifacts] run dir does not exist: {run_dir}",
              file=sys.stderr)
        sys.exit(1)
    missing = collect_missing(run_dir)
    report = {"run_dir": str(run_dir),
              "complete": not missing,
              "missing": missing}
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                          encoding="utf-8")
    if missing:
        print(f"[artifacts] INCOMPLETE, missing {len(missing)}:")
        for name in missing:
            print(f"  - {name}")
        sys.exit(1)
    print(f"[artifacts] complete: {run_dir}")


if __name__ == "__main__":
    main()
