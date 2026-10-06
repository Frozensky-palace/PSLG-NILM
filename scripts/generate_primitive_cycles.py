"""B4 basic composition: sample primitives and compose full cycles.

Loads a primitive checkpoint (train_full_cycle_generator.py --route
primitive | primitive-wgan | primitive-diffusion) plus the same frozen
state library, fits the empirical path model from train transitions and
composes cycles whose segments come from the decoded primitives. Output is
a standard synthetic dataset that the batch-0 quality and memorization
gates can consume directly.

--primitive-source wgan / diffusion (B4WD arms) swap only the primitive
decoder backend; the path model, donor picks and numpy rng sequence are
untouched, so same-seed runs stay equivalent to the B4 archive in state
paths and segment lengths. The diffusion arm additionally applies the
preregistered per-state power scales (from the checkpoint) and per-state
peak caps (donor peak max per state, computed from the state library).

--primitive-source real (B4-real ablation) skips the checkpoint entirely:
the composer keeps its path sampling and donor selection unchanged and uses
the selected real donor waveforms as the primitives, isolating the effect
of swapping generated primitives for real ones.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.full_cycle_cvae import (  # noqa: E402
    ConditionalWaveformCVAE,
    LengthBucketizer,
)
from src.generation.full_cycle_diffusion import (  # noqa: E402
    DiffusionDenoiser,
    GaussianDiffusion,
)
from src.generation.full_cycle_wgan import WGANGenerator  # noqa: E402
from src.generation.primitive_cvae import (  # noqa: E402
    PrimitiveComposer,
    fit_path_model,
    load_state_segments,
)

CHECKPOINT_ROUTES = {"cvae": "primitive", "wgan": "primitive-wgan",
                     "diffusion": "primitive-diffusion"}


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


def load_wgan_checkpoint(checkpoint_dir: Path, device: str
                         ) -> tuple[WGANGenerator, LengthBucketizer,
                                    dict, int, float]:
    payload = torch.load(Path(checkpoint_dir) / "model.pt",
                         map_location=device, weights_only=False)
    generator = WGANGenerator(
        payload["wave_length"], payload["condition_dim"],
        latent_dim=payload["latent_dim"], width=payload["width"])
    generator.load_state_dict(payload["generator"])
    bucketizer = LengthBucketizer.from_dict(payload["bucketizer"])
    return (generator, bucketizer, payload["normalizer"],
            payload["n_states"], float(payload["power_scale"]))


def load_diffusion_checkpoint(checkpoint_dir: Path, device: str
                              ) -> tuple[DiffusionDenoiser,
                                         GaussianDiffusion, LengthBucketizer,
                                         dict, int, dict[int, float]]:
    payload = torch.load(Path(checkpoint_dir) / "model.pt",
                         map_location=device, weights_only=False)
    denoiser = DiffusionDenoiser(payload["wave_length"],
                                 payload["condition_dim"],
                                 width=payload["width"])
    denoiser.load_state_dict(payload["denoiser"])
    diffusion = GaussianDiffusion(n_steps=payload["diffusion_steps"])
    bucketizer = LengthBucketizer.from_dict(payload["bucketizer"])
    power_scales = {int(state): float(scale)
                    for state, scale in payload["power_scales"].items()}
    return (denoiser, diffusion, bucketizer, payload["normalizer"],
            payload["n_states"], power_scales)


def _state_peak_caps(donor_waves: list, donor_labels: list[int]
                     ) -> dict[int, float]:
    """Per-state donor peak max (preregistered B4DIFF post-processing)."""
    caps: dict[int, float] = {}
    for wave, label in zip(donor_waves, donor_labels):
        state = int(label)
        peak = float(wave.max()) if len(wave) else 0.0
        caps[state] = max(caps.get(state, 0.0), peak)
    return caps


def _write_peak_cap_report(output_dir: Path, records, peak_caps) -> None:
    """Prereg §4 evidence for capped arms (B4DIFF): 贴上限占比 + 截断触发率.

    After np.minimum no sample can exceed the cap, so ``power >= cap``
    identifies exactly the samples the per-state peak clip pinned.
    """
    per_state: dict[int, dict[str, int]] = {}
    pinned_total = samples_total = triggered_total = segments_total = 0
    for record in records:
        with np.load(output_dir / "cycles" / f"{record.synthetic_cycle_id}.npz"
                     ) as data:
            power = data["appliance_w"].astype(np.float64)
        cursor = 0
        for segment in record.segments:
            piece = power[cursor:cursor + segment.actual_samples]
            cursor += segment.actual_samples
            cap = peak_caps[segment.state_label]
            pinned = int((piece >= cap).sum())
            stats = per_state.setdefault(segment.state_label,
                                         {"pinned": 0, "samples": 0,
                                          "segments": 0, "triggered": 0})
            stats["pinned"] += pinned
            stats["samples"] += len(piece)
            stats["segments"] += 1
            stats["triggered"] += 1 if pinned else 0
            pinned_total += pinned
            samples_total += len(piece)
            segments_total += 1
            triggered_total += 1 if pinned else 0
    report = {
        "protocol": "peak_cap_report_v1",
        "definition": ("post-clip no sample exceeds the cap; power >= cap "
                       "counts exactly the samples pinned by the per-state "
                       "peak clip"),
        "peak_caps_w": {int(state): float(cap)
                        for state, cap in peak_caps.items()},
        "n_cycles": len(records),
        "cap_pinned_share": pinned_total / samples_total,
        "cap_trigger_rate": triggered_total / segments_total,
        "per_state": {
            str(state): {
                "cap_pinned_share": s["pinned"] / s["samples"],
                "cap_trigger_rate": s["triggered"] / s["segments"],
                "n_segments": s["segments"],
            } for state, s in sorted(per_state.items())},
    }
    (output_dir / "peak_cap_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[b4-compose] peak-cap evidence (prereg §4): pinned_share="
          f"{report['cap_pinned_share']:.4f} trigger_rate="
          f"{report['cap_trigger_rate']:.4f} -> "
          f"{output_dir / 'peak_cap_report.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-dir",
                    help="primitive checkpoint (required for --primitive-"
                         "source cvae/wgan/diffusion)")
    ap.add_argument("--primitive-source", default="cvae",
                    choices=("cvae", "real", "wgan", "diffusion"),
                    help="cvae: decode primitives with the checkpoint CVAE; "
                         "real: use the selected real donor waveforms "
                         "directly (B4-real ablation, zero training); "
                         "wgan/diffusion: B4WD arms — swap only the "
                         "primitive decoder backend")
    ap.add_argument("--state-library-dir", required=True,
                    help="the same frozen state library used in training")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--sample-seconds", type=int, default=6)
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = ap.parse_args()

    if args.primitive_source != "real" and not args.checkpoint_dir:
        raise SystemExit("--checkpoint-dir is required for --primitive-"
                         "source cvae/wgan/diffusion")

    rows = list(csv.DictReader(open(
        Path(args.state_library_dir) / "state_inventory.csv",
        encoding="utf-8")))
    path_model = fit_path_model(rows)
    donor_waves, donor_labels, donor_meta = load_state_segments(
        Path(args.state_library_dir))

    if args.primitive_source == "real":
        n_states = max(donor_labels) + 1
        composer = PrimitiveComposer(
            None, None, donor_waves, donor_labels, path_model,
            n_states, None, sample_seconds=args.sample_seconds,
            primitive_source="donor",
            donor_block_ids=donor_meta["state_block_ids"],
            donor_cycle_ids=donor_meta["cycle_ids"])
    elif args.primitive_source == "wgan":
        checkpoint_dir = Path(args.checkpoint_dir)
        config = json.loads((checkpoint_dir / "config.json")
                            .read_text(encoding="utf-8"))
        expected = CHECKPOINT_ROUTES["wgan"]
        if config.get("route") != expected:
            raise SystemExit(
                f"checkpoint route {config.get('route')!r} != {expected!r}")
        generator, bucketizer, normalizer, n_states, power_scale = (
            load_wgan_checkpoint(checkpoint_dir, args.device))
        composer = PrimitiveComposer(
            generator, bucketizer, donor_waves, donor_labels, path_model,
            n_states, normalizer, sample_seconds=args.sample_seconds,
            device=args.device, power_scale=power_scale,
            model_backend="wgan")
    elif args.primitive_source == "diffusion":
        checkpoint_dir = Path(args.checkpoint_dir)
        config = json.loads((checkpoint_dir / "config.json")
                            .read_text(encoding="utf-8"))
        expected = CHECKPOINT_ROUTES["diffusion"]
        if config.get("route") != expected:
            raise SystemExit(
                f"checkpoint route {config.get('route')!r} != {expected!r}")
        (denoiser, diffusion, bucketizer, normalizer, n_states,
         power_scales) = load_diffusion_checkpoint(checkpoint_dir,
                                                   args.device)
        composer = PrimitiveComposer(
            denoiser, bucketizer, donor_waves, donor_labels, path_model,
            n_states, normalizer, sample_seconds=args.sample_seconds,
            device=args.device, model_backend="diffusion",
            power_scales=power_scales,
            peak_caps=_state_peak_caps(donor_waves, donor_labels),
            diffusion=diffusion)
    else:
        checkpoint_dir = Path(args.checkpoint_dir)
        config = json.loads((checkpoint_dir / "config.json")
                            .read_text(encoding="utf-8"))
        expected = CHECKPOINT_ROUTES["cvae"]
        if config.get("route") != expected:
            raise SystemExit(
                f"checkpoint route {config.get('route')!r} != {expected!r}")
        model, bucketizer, normalizer, n_states, power_scale = load_checkpoint(
            checkpoint_dir, args.device)
        composer = PrimitiveComposer(
            model, bucketizer, donor_waves, donor_labels, path_model,
            n_states, normalizer, sample_seconds=args.sample_seconds,
            device=args.device, power_scale=power_scale)
    records = composer.generate_dataset(
        Path(args.output_dir), count=args.count, seed=args.seed,
        sample_seconds=args.sample_seconds)
    if getattr(composer, "peak_caps", None) is not None:
        _write_peak_cap_report(Path(args.output_dir), records,
                               composer.peak_caps)
    print(f"[b4-compose] composed {len(records)} cycles -> {args.output_dir}")


if __name__ == "__main__":
    main()
