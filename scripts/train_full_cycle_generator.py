"""Unified training entry for neural generators (B3-V full-cycle CVAE,
B4 shared primitive CVAE, and the B4WD segment-level WGAN/diffusion arms).
Saves a self-describing checkpoint directory:
model.pt + history.json + config.json. CPU smoke runs are first-class; the
same command trains formally on the server with --device cuda.
"""
from __future__ import annotations

import argparse
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
    condition_vector,
    train_cvae,
)
from src.generation.full_cycle_diffusion import (  # noqa: E402
    DiffusionDenoiser,
    GaussianDiffusion,
    train_diffusion,
)
from src.generation.full_cycle_wgan import (  # noqa: E402
    WGANCritic,
    WGANGenerator,
    train_wgan,
)
from src.generation.primitive_cvae import (  # noqa: E402
    build_segment_conditions,
    load_state_segments,
)
from src.generation.full_cycle_wgan import (  # noqa: E402
    WGANCritic,
    WGANGenerator,
    train_primitive_wgan,
)
from src.generation.full_cycle_diffusion import (  # noqa: E402
    DiffusionDenoiser,
    GaussianDiffusion,
    train_primitive_diffusion,
)
from src.validation.synthetic_quality import load_real_reference  # noqa: E402


def _mean_power_scale(waves: list[np.ndarray]) -> float:
    return max(float(np.quantile([w.mean() for w in waves], 0.95)), 1e-9)


def _save_checkpoint(output_dir: Path, payload: dict,
                     history: list[dict], config: dict) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_dir / "model.pt")
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


def _power_scale(waves: list[np.ndarray]) -> float:
    """q99 of peaks; raw watts make the reconstruction loss explode."""
    return float(max(np.quantile([w.max() for w in waves], 0.99), 1.0))


def _std_power_scale(waves: list[np.ndarray]) -> float:
    """Std-based scale for diffusion: the signal must have unit variance
    against the unit-variance noise the forward process adds, otherwise the
    denoiser converges to predicting ~zero and samples keep noise spikes
    (observed as impossible_peak FAIL, job 4136)."""
    stds = [w.std() for w in waves if len(w)]
    return float(max(np.quantile(stds, 0.95), 1.0))


def _state_std_power_scales(waves: list[np.ndarray],
                            labels: list[int]) -> dict[int, float]:
    """Per-state std scale for segment diffusion (B4WD prereg §2).

    q95 of each state's segment stds, floored at 1.0 W. Segment-levelized
    because state median powers span 153→1866 W: a single global std would
    starve low-power states of signal-to-noise and replay the job-4136
    failure. Scaling/post-processing in nature, so it lives in the scripts
    layer (d83f200 precedent, preregistered layering rule).
    """
    by_state: dict[int, list[float]] = {}
    for wave, label in zip(waves, labels):
        if len(wave):
            by_state.setdefault(int(label), []).append(float(wave.std()))
    return {state: float(max(np.quantile(stds, 0.95), 1.0))
            for state, stds in sorted(by_state.items())}


def train_full_cycle_cvae(args: argparse.Namespace) -> None:
    waves = load_real_reference(Path(args.real_library_dir),
                                max_cycles=args.max_cycles)
    power_scale = _power_scale(waves)
    lengths = np.array([len(w) for w in waves])
    bucketizer = LengthBucketizer.fit(lengths, n_buckets=args.n_buckets)
    length_scale = bucketizer.bucket_length(bucketizer.n_buckets - 1)
    mean_scale = _mean_power_scale(waves)
    conditions = np.stack([
        condition_vector(len(w), w, args.sample_seconds, mean_scale,
                         length_scale) for w in waves])
    bucket_length = bucketizer.bucket_length(bucketizer.n_buckets - 1)
    model = ConditionalWaveformCVAE(
        bucket_length, condition_dim=len(conditions[0]),
        latent_dim=args.latent_dim, width=args.width)
    history = train_cvae(model, [w / power_scale for w in waves],
                         conditions, bucketizer,
                         epochs=args.epochs, batch_size=args.batch_size,
                         learning_rate=args.learning_rate, seed=args.seed,
                         device=args.device)
    _save_checkpoint(
        Path(args.output_dir),
        {"model": model.state_dict(), "wave_length": bucket_length,
         "condition_dim": int(len(conditions[0])),
         "latent_dim": args.latent_dim, "width": args.width,
         "bucketizer": bucketizer.to_dict(),
         "length_scale": length_scale, "mean_power_scale": mean_scale,
         "power_scale": power_scale},
        history,
        {"route": "cvae", "epochs": args.epochs, "seed": args.seed,
         "n_waves": len(waves), "device": args.device,
         "power_scale": power_scale,
         "real_library_dir": str(args.real_library_dir)})
    print(f"[train-cvae] final loss={history[-1]['loss']:.4f} "
          f"first={history[0]['loss']:.4f} power_scale={power_scale:.1f} "
          f"-> {args.output_dir}")


def train_primitive_cvae(args: argparse.Namespace) -> None:
    waves, labels, _ = load_state_segments(Path(args.state_library_dir))
    n_states = max(labels) + 1
    power_scale = _power_scale(waves)
    conditions, bucketizer, normalizer = build_segment_conditions(
        waves, labels, n_states, sample_seconds=args.sample_seconds)
    bucket_length = bucketizer.bucket_length(bucketizer.n_buckets - 1)
    model = ConditionalWaveformCVAE(
        bucket_length, condition_dim=int(conditions.shape[1]),
        latent_dim=args.latent_dim, width=args.width)
    history = train_cvae(model, [w / power_scale for w in waves],
                         conditions, bucketizer,
                         epochs=args.epochs, batch_size=args.batch_size,
                         learning_rate=args.learning_rate, seed=args.seed,
                         device=args.device)
    _save_checkpoint(
        Path(args.output_dir),
        {"model": model.state_dict(), "wave_length": bucket_length,
         "condition_dim": int(conditions.shape[1]),
         "latent_dim": args.latent_dim, "width": args.width,
         "bucketizer": bucketizer.to_dict(), "normalizer": normalizer,
         "n_states": n_states, "power_scale": power_scale},
        history,
        {"route": "primitive", "epochs": args.epochs, "seed": args.seed,
         "n_segments": len(waves), "n_states": n_states,
         "device": args.device, "power_scale": power_scale,
         "state_library_dir": str(args.state_library_dir)})
    print(f"[train-primitive] final loss={history[-1]['loss']:.4f} "
          f"first={history[0]['loss']:.4f} power_scale={power_scale:.1f} "
          f"-> {args.output_dir}")


def train_primitive_wgan_arm(args: argparse.Namespace) -> None:
    """B4WGAN arm: segment-level masked WGAN-GP on the state library."""
    waves, labels, _ = load_state_segments(Path(args.state_library_dir))
    n_states = max(labels) + 1
    power_scale = _power_scale(waves)
    conditions, bucketizer, normalizer = build_segment_conditions(
        waves, labels, n_states, sample_seconds=args.sample_seconds)
    bucket_length = bucketizer.bucket_length(bucketizer.n_buckets - 1)
    generator = WGANGenerator(bucket_length, int(conditions.shape[1]),
                              latent_dim=args.latent_dim, width=args.width)
    critic = WGANCritic(bucket_length, int(conditions.shape[1]),
                        width=args.width)
    history = train_primitive_wgan(
        generator, critic, [w / power_scale for w in waves], conditions,
        bucketizer, labels, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, n_critic=args.n_critic,
        seed=args.seed, device=args.device)
    _save_checkpoint(
        Path(args.output_dir),
        {"generator": generator.state_dict(), "critic": critic.state_dict(),
         "wave_length": bucket_length, "condition_dim": int(conditions.shape[1]),
         "latent_dim": args.latent_dim, "width": args.width,
         "bucketizer": bucketizer.to_dict(), "normalizer": normalizer,
         "n_states": n_states, "power_scale": power_scale},
        history,
        {"route": "primitive-wgan", "epochs": args.epochs, "seed": args.seed,
         "n_segments": len(waves), "n_states": n_states,
         "device": args.device, "power_scale": power_scale,
         "learning_rate": args.learning_rate,
         "n_critic": args.n_critic, "gp_weight": 10.0,
         "state_library_dir": str(args.state_library_dir)})
    per_state = history[-1].get("w_distance_by_state", {})
    print(f"[train-primitive-wgan] final w_distance="
          f"{history[-1]['w_distance']:.4f} power_scale={power_scale:.1f} "
          f"w_distance_by_state={per_state} -> {args.output_dir}")


def train_primitive_diffusion_arm(args: argparse.Namespace) -> None:
    """B4DIFF arm: segment-level masked diffusion, per-state std scale."""
    waves, labels, _ = load_state_segments(Path(args.state_library_dir))
    n_states = max(labels) + 1
    power_scales = _state_std_power_scales(waves, labels)
    scaled = [w / power_scales[int(label)]
              for w, label in zip(waves, labels)]
    # Prereg §3 smoke check: per-state scaled std must sit near 1 or the
    # unit-variance training assumption is broken (job-4136 failure chain).
    scaled_std = {}
    for state, scale in power_scales.items():
        stds = [float(w.std()) for w, label in zip(scaled, labels)
                if int(label) == state and len(w)]
        if stds:
            scaled_std[state] = float(np.median(stds))
    print("[train-primitive-diffusion] per-state scales "
          + ", ".join(f"s{s}={v:.1f}W" for s, v in power_scales.items())
          + "; scaled median std "
          + ", ".join(f"s{s}={v:.3f}" for s, v in scaled_std.items()))
    conditions, bucketizer, normalizer = build_segment_conditions(
        waves, labels, n_states, sample_seconds=args.sample_seconds)
    bucket_length = bucketizer.bucket_length(bucketizer.n_buckets - 1)
    denoiser = DiffusionDenoiser(bucket_length, int(conditions.shape[1]),
                                 width=args.width)
    diffusion = GaussianDiffusion(n_steps=args.diffusion_steps)
    history = train_primitive_diffusion(
        denoiser, diffusion, scaled, conditions, bucketizer,
        epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, seed=args.seed,
        device=args.device)
    _save_checkpoint(
        Path(args.output_dir),
        {"denoiser": denoiser.state_dict(), "wave_length": bucket_length,
         "condition_dim": int(conditions.shape[1]), "width": args.width,
         "diffusion_steps": args.diffusion_steps,
         "bucketizer": bucketizer.to_dict(), "normalizer": normalizer,
         "n_states": n_states, "power_scale": 1.0,
         "power_scales": power_scales},
        history,
        {"route": "primitive-diffusion", "epochs": args.epochs,
         "seed": args.seed, "n_segments": len(waves), "n_states": n_states,
         "device": args.device, "power_scales": power_scales,
         "learning_rate": args.learning_rate,
         "diffusion_steps": args.diffusion_steps,
         "state_library_dir": str(args.state_library_dir)})
    print(f"[train-primitive-diffusion] final noise_mse="
          f"{history[-1]['noise_mse']:.5f} first="
          f"{history[0]['noise_mse']:.5f} -> {args.output_dir}")


def train_full_cycle_wgan(args: argparse.Namespace) -> None:
    waves = load_real_reference(Path(args.real_library_dir),
                                max_cycles=args.max_cycles)
    power_scale = _power_scale(waves)
    lengths = np.array([len(w) for w in waves])
    bucketizer = LengthBucketizer.fit(lengths, n_buckets=args.n_buckets)
    length_scale = bucketizer.bucket_length(bucketizer.n_buckets - 1)
    mean_scale = _mean_power_scale(waves)
    conditions = np.stack([
        condition_vector(len(w), w, args.sample_seconds, mean_scale,
                         length_scale) for w in waves])
    bucket_length = bucketizer.bucket_length(bucketizer.n_buckets - 1)
    generator = WGANGenerator(bucket_length, len(conditions[0]),
                              latent_dim=args.latent_dim, width=args.width)
    critic = WGANCritic(bucket_length, len(conditions[0]), width=args.width)
    history = train_wgan(generator, critic, [w / power_scale for w in waves],
                         conditions, bucketizer,
                         epochs=args.epochs, batch_size=args.batch_size,
                         learning_rate=args.learning_rate,
                         n_critic=args.n_critic, seed=args.seed,
                         device=args.device)
    _save_checkpoint(
        Path(args.output_dir),
        {"generator": generator.state_dict(), "critic": critic.state_dict(),
         "wave_length": bucket_length, "condition_dim": len(conditions[0]),
         "latent_dim": args.latent_dim, "width": args.width,
         "bucketizer": bucketizer.to_dict(),
         "length_scale": length_scale, "mean_power_scale": mean_scale,
         "power_scale": power_scale},
        history,
        {"route": "wgan", "epochs": args.epochs, "seed": args.seed,
         "n_waves": len(waves), "device": args.device,
         "power_scale": power_scale,
         "real_library_dir": str(args.real_library_dir)})
    print(f"[train-wgan] final w_distance={history[-1]['w_distance']:.4f} "
          f"power_scale={power_scale:.1f} -> {args.output_dir}")


def train_full_cycle_diffusion(args: argparse.Namespace) -> None:
    waves = load_real_reference(Path(args.real_library_dir),
                                max_cycles=args.max_cycles)
    power_scale = _std_power_scale(waves)
    lengths = np.array([len(w) for w in waves])
    bucketizer = LengthBucketizer.fit(lengths, n_buckets=args.n_buckets)
    length_scale = bucketizer.bucket_length(bucketizer.n_buckets - 1)
    mean_scale = _mean_power_scale(waves)
    conditions = np.stack([
        condition_vector(len(w), w, args.sample_seconds, mean_scale,
                         length_scale) for w in waves])
    bucket_length = bucketizer.bucket_length(bucketizer.n_buckets - 1)
    denoiser = DiffusionDenoiser(bucket_length, len(conditions[0]),
                                 width=args.width)
    diffusion = GaussianDiffusion(n_steps=args.diffusion_steps)
    history = train_diffusion(denoiser, diffusion,
                              [w / power_scale for w in waves], conditions,
                              bucketizer, epochs=args.epochs,
                              batch_size=args.batch_size,
                              learning_rate=args.learning_rate,
                              seed=args.seed, device=args.device)
    _save_checkpoint(
        Path(args.output_dir),
        {"denoiser": denoiser.state_dict(), "wave_length": bucket_length,
         "condition_dim": len(conditions[0]), "width": args.width,
         "diffusion_steps": args.diffusion_steps,
         "bucketizer": bucketizer.to_dict(),
         "length_scale": length_scale, "mean_power_scale": mean_scale,
         "power_scale": power_scale},
        history,
        {"route": "diffusion", "epochs": args.epochs, "seed": args.seed,
         "n_waves": len(waves), "device": args.device,
         "power_scale": power_scale,
         "real_library_dir": str(args.real_library_dir)})
    print(f"[train-diffusion] final noise_mse="
          f"{history[-1]['noise_mse']:.5f} -> {args.output_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--route",
                    choices=("cvae", "primitive", "wgan", "diffusion",
                             "primitive-wgan", "primitive-diffusion"),
                    required=True)
    ap.add_argument("--real-library-dir", default=None,
                    help="cvae route: real train cycle library")
    ap.add_argument("--state-library-dir", default=None,
                    help="primitive/primitive-wgan/primitive-diffusion "
                         "routes: frozen state library v1")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--learning-rate", type=float, default=None,
                    help="default: 1e-4 for primitive-wgan (prereg §3 "
                         "Adam(1e-4, betas 0.5/0.9)), 1e-3 otherwise")
    ap.add_argument("--latent-dim", type=int, default=16)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--n-critic", type=int, default=5,
                    help="wgan route: critic updates per generator update")
    ap.add_argument("--diffusion-steps", type=int, default=500,
                    help="diffusion route: noise schedule steps")
    ap.add_argument("--n-buckets", type=int, default=4)
    ap.add_argument("--max-cycles", type=int, default=None)
    ap.add_argument("--sample-seconds", type=int, default=6)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = ap.parse_args()

    if args.learning_rate is None:
        # Prereg §3 freezes the B4WGAN optimizer at Adam(1e-4, betas 0.5/0.9)
        # ("不另行调参"); other routes keep the historical 1e-3 default.
        # Route-dependent default (not a CLI constant) so the formal sbatch
        # cannot silently train the wrong lr.
        args.learning_rate = (1e-4 if args.route == "primitive-wgan"
                              else 1e-3)

    # Seed BEFORE any model construction: weight init draws from the global
    # torch RNG, which is process-entropy-seeded by default. Without this the
    # declared --seed only governed sampling inside train_* — same-seed reruns
    # silently produced different weights (caught via test flakiness).
    torch.manual_seed(args.seed)

    if args.route == "cvae":
        if not args.real_library_dir:
            raise SystemExit("--route cvae needs --real-library-dir")
        train_full_cycle_cvae(args)
    elif args.route == "wgan":
        if not args.real_library_dir:
            raise SystemExit("--route wgan needs --real-library-dir")
        train_full_cycle_wgan(args)
    elif args.route == "diffusion":
        if not args.real_library_dir:
            raise SystemExit("--route diffusion needs --real-library-dir")
        train_full_cycle_diffusion(args)
    elif args.route == "primitive-wgan":
        if not args.state_library_dir:
            raise SystemExit(
                "--route primitive-wgan needs --state-library-dir")
        train_primitive_wgan_arm(args)
    elif args.route == "primitive-diffusion":
        if not args.state_library_dir:
            raise SystemExit(
                "--route primitive-diffusion needs --state-library-dir")
        train_primitive_diffusion_arm(args)
    else:
        if not args.state_library_dir:
            raise SystemExit("--route primitive needs --state-library-dir")
        train_primitive_cvae(args)


if __name__ == "__main__":
    main()
