"""Run the synthetic-cycle quality gate for one generator output directory.

Reads ``generation_summary.json`` + ``cycles/*.npz`` produced by any
BaseGenerator route, compares them against the real train cycle library and
writes a JSON report. Exits non-zero when a hard check fails so that batch
pipelines stop before placing bad cycles on a background.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.validation.synthetic_quality import evaluate_synthetic_dataset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synthetic-dir", required=True,
                    help="generator output dir (generation_summary.json)")
    ap.add_argument("--real-library-dir", required=True,
                    help="real train cycle library dir")
    ap.add_argument("--sample-seconds", type=int, default=6)
    ap.add_argument("--max-real-cycles", type=int, default=None,
                    help="optional cap on real reference size for speed")
    ap.add_argument("--output", required=True, help="JSON report path")
    args = ap.parse_args()

    report = evaluate_synthetic_dataset(
        Path(args.synthetic_dir), Path(args.real_library_dir),
        sample_seconds=args.sample_seconds,
        max_real_cycles=args.max_real_cycles)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                      encoding="utf-8")
    flags = " ".join(f"{k}={v}" for k, v in report["flags"].items())
    print(f"[quality] n={report['n_synthetic']} flags: {flags}")
    print(f"[quality] report -> {output}")
    if not report["passed"]:
        raise SystemExit("quality gate FAILED; see flags in the report")


if __name__ == "__main__":
    main()
