"""Audit a synthetic dataset for replication of real train cycles.

Every synthetic cycle is compared against the real train library by
shape distance; cycles closer than the real-vs-real baseline threshold
(or bit-exact duplicates) are counted. Exits non-zero when the exact
duplicate count is non-zero or the replication rate exceeds the cap.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.validation.memorization import audit_memorization  # noqa: E402
from src.validation.synthetic_quality import (  # noqa: E402
    load_real_reference,
    load_synthetic_cycles,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synthetic-dir", required=True)
    ap.add_argument("--real-library-dir", required=True)
    ap.add_argument("--threshold-percentile", type=float, default=1.0,
                    help="baseline quantile defining the replication cut")
    ap.add_argument("--max-replication-rate", type=float, default=0.01,
                    help="fail when more than this fraction replicate")
    ap.add_argument("--max-real-cycles", type=int, default=None)
    ap.add_argument("--output", required=True, help="JSON report path")
    args = ap.parse_args()

    synth, summary = load_synthetic_cycles(Path(args.synthetic_dir))
    real = load_real_reference(Path(args.real_library_dir),
                               args.max_real_cycles)
    report = audit_memorization(
        synth, real,
        threshold_percentile=args.threshold_percentile,
        seed=int(summary.get("seed", 0)))
    report["generator"] = summary.get("generator")
    report["config_hash"] = summary.get("config_hash")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                      encoding="utf-8")
    print(f"[memorization] replicated={report['replicated_count']}"
          f"/{report['n_synthetic']} "
          f"exact={report['exact_duplicate_count']} "
          f"threshold={report['threshold_distance']:.4f}")
    print(f"[memorization] report -> {output}")
    if report["exact_duplicate_count"] > 0:
        raise SystemExit("FAILED: exact duplicates of real cycles found")
    if report["replication_rate"] > args.max_replication_rate:
        raise SystemExit("FAILED: replication rate above cap")


if __name__ == "__main__":
    main()
