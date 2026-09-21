"""B4 basic composition: sample primitives and compose full cycles.

Loads a primitive-CVAE checkpoint (train_full_cycle_generator.py --route
primitive) plus the same frozen state library, fits the empirical path
model from train transitions and composes cycles whose segments come from
the decoded primitives. Output is a standard synthetic dataset that the
batch-0 quality and memorization gates can consume directly.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.full_cycle_cvae import (  # noqa: E402
    ConditionalWaveformCVAE,
    LengthBucketizer,
)
from src.generation.primitive_cvae import (  # noqa: E402
    PrimitiveComposer,
    fit_path_model,
    load_state_segments,
)


def load_checkpoint(checkpoint_dir: Path, device: str
                    ) -> tuple[ConditionalWaveformCVAE, LengthBucketizer,
                               dict, int, float]:
    payload = torch.load(Path(checkpoint_dir) / "model.pt",
                         map_location=device, weights_only=False)
    model = ConditionalWaveformCVAE(
        payload["wave_length"], payload["condition_dim"],
        latent_dim=payload["latent_dim"], width=payload["width"])
    model.load_state_dict(payload["model"])
    bucketizer = LengthBucketizer.from_dict(payload["bucketizer"])
    return (model, bucketizer, payload["normalizer"], payload["n_states"],
            float(payload.get("power_scale", 1.0)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--state-library-dir", required=True,
                    help="the same frozen state library used in training")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--sample-seconds", type=int, default=6)
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = ap.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    config = json.loads((checkpoint_dir / "config.json")
                        .read_text(encoding="utf-8"))
    if config.get("route") != "primitive":
        raise SystemExit("checkpoint is not a primitive-CVAE checkpoint")

    model, bucketizer, normalizer, n_states, power_scale = load_checkpoint(
        checkpoint_dir, args.device)
    donor_waves, donor_labels, _ = load_state_segments(
        Path(args.state_library_dir))
    rows = list(csv.DictReader(open(
        Path(args.state_library_dir) / "state_inventory.csv",
        encoding="utf-8")))
    path_model = fit_path_model(rows)

    composer = PrimitiveComposer(
        model, bucketizer, donor_waves, donor_labels, path_model,
        n_states, normalizer, sample_seconds=args.sample_seconds,
        device=args.device, power_scale=power_scale)
    records = composer.generate_dataset(
        Path(args.output_dir), count=args.count, seed=args.seed,
        sample_seconds=args.sample_seconds)
    print(f"[b4-compose] composed {len(records)} cycles -> {args.output_dir}")


if __name__ == "__main__":
    main()
