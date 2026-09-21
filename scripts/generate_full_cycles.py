"""Unified entry for full-cycle generation routes (roadmap D1-D4).

Currently implemented routes:

- ``transform``  B3-T real-cycle transform (no neural training)

Routes are added here only after their module exists and passed local
smoke; the CLI fails fast on anything else instead of pretending.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.base_generator import BaseGenerator  # noqa: E402
from src.generation.full_cycle_transform import (  # noqa: E402
    RealCycleTransformGenerator,
)

ROUTE_CHOICES = ("transform",)


def build_generator(args: argparse.Namespace) -> BaseGenerator:
    if args.route == "transform":
        return RealCycleTransformGenerator(
            args.real_library_dir,
            sample_seconds=args.sample_seconds,
            time_scale_range=(args.time_scale_min, args.time_scale_max),
            power_scale_range=(args.power_scale_min, args.power_scale_max))
    raise SystemExit(
        f"route {args.route!r} is not implemented yet; choose from "
        f"{ROUTE_CHOICES}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--route", choices=ROUTE_CHOICES, required=True)
    ap.add_argument("--real-library-dir", required=True,
                    help="train-only real cycle library (donor source)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--sample-seconds", type=int, default=6)
    ap.add_argument("--time-scale-min", type=float, default=0.85)
    ap.add_argument("--time-scale-max", type=float, default=1.2)
    ap.add_argument("--power-scale-min", type=float, default=0.9)
    ap.add_argument("--power-scale-max", type=float, default=1.1)
    args = ap.parse_args()

    generator = build_generator(args)
    records = generator.generate_dataset(
        Path(args.output_dir), count=args.count, seed=args.seed,
        sample_seconds=args.sample_seconds)
    print(f"[generate] route={args.route} count={len(records)} "
          f"seed={args.seed} -> {args.output_dir}")


if __name__ == "__main__":
    main()
