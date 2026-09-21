"""B5 constrained composition from a primitive pool + frozen HSMM model.

The primitive pool is any generation output whose records carry labelled
segments (e.g. generate_primitive_cycles.py output); segments are sliced
into a per-state pool, the HSMM model (fit_hsmm_composer.py) dictates the
state order and target durations, and the boundary mode is an ablation
flag. Output is a standard synthetic dataset for the batch-0 gates.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.composition.duration_model import StateDurationModel  # noqa: E402
from src.composition.hsmm_sequence import HSMMPathSampler  # noqa: E402
from src.composition.transition_model import MarkovChain  # noqa: E402
from scripts.compose_generated_cycles import load_primitive_pool  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--primitives-dir", required=True,
                    help="pool source: generation output with labelled "
                         "segments")
    ap.add_argument("--hsmm-json", required=True,
                    help="frozen model from fit_hsmm_composer.py")
    ap.add_argument("--boundary-mode", default="none",
                    choices=("none", "endpoint_match"))
    ap.add_argument("--ignore-hsmm-durations", action="store_true",
                    help="ablation: keep donor lengths (B4+Markov rung)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--sample-seconds", type=int, default=6)
    args = ap.parse_args()

    if args.boundary_mode == "none" and not args.ignore_hsmm_durations:
        pass  # frozen B5 default: HSMM durations + no boundary surgery

    from src.composition.constrained_composer import ConstrainedComposer

    pool_waves, pool_segments, _ = load_primitive_pool(
        Path(args.primitives_dir))
    pool_waves_by_segment: list[np.ndarray] = []
    pool_labels: list[int] = []
    for cycle_wave, segments in zip(pool_waves, pool_segments):
        cursor = 0
        for segment in segments:
            length = int(segment["actual_samples"])
            pool_waves_by_segment.append(
                cycle_wave[cursor:cursor + length])
            pool_labels.append(int(segment["state_label"]))
            cursor += length

    model = json.loads(Path(args.hsmm_json).read_text(encoding="utf-8"))
    markov = MarkovChain.from_dict(model["markov"])
    durations = StateDurationModel.from_dict(model["durations"])
    hsmm = HSMMPathSampler(markov, durations, model["path_lengths"])

    composer = ConstrainedComposer(
        hsmm, pool_waves_by_segment, pool_labels,
        sample_seconds=args.sample_seconds,
        boundary_mode=args.boundary_mode,
        use_hsmm_durations=not args.ignore_hsmm_durations)
    records = composer.generate_dataset(
        Path(args.output_dir), count=args.count, seed=args.seed,
        sample_seconds=args.sample_seconds)
    print(f"[b5-compose] composed {len(records)} cycles "
          f"boundary={args.boundary_mode} -> {args.output_dir}")


if __name__ == "__main__":
    main()
