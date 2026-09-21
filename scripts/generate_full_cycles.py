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
from src.generation.full_cycle_cvae import (  # noqa: E402
    CVAESamplingGenerator,
    ConditionalWaveformCVAE,
    LengthBucketizer,
)
from src.generation.full_cycle_diffusion import (  # noqa: E402
    DiffusionDenoiser,
    GaussianDiffusion,
)
from src.generation.full_cycle_transform import (  # noqa: E402
    RealCycleTransformGenerator,
)
from src.generation.full_cycle_wgan import WGANGenerator  # noqa: E402
from src.validation.synthetic_quality import load_real_reference  # noqa: E402

ROUTE_CHOICES = ("transform", "cvae", "wgan", "diffusion")


def _load_cvae_checkpoint(checkpoint_dir: Path, device: str
                          ) -> tuple[ConditionalWaveformCVAE,
                                     LengthBucketizer, dict, float]:
    import torch

    payload = torch.load(Path(checkpoint_dir) / "model.pt",
                         map_location=device, weights_only=False)
    if payload.get("condition_dim") != 4:
        raise SystemExit("checkpoint is not a full-cycle CVAE (condition_dim)")
    model = ConditionalWaveformCVAE(
        payload["wave_length"], payload["condition_dim"],
        latent_dim=payload["latent_dim"], width=payload["width"])
    model.load_state_dict(payload["model"])
    bucketizer = LengthBucketizer.from_dict(payload["bucketizer"])
    normalizer = {"length_scale": payload["length_scale"],
                  "mean_power_scale": payload["mean_power_scale"]}
    return model, bucketizer, normalizer, float(payload.get("power_scale",
                                                            1.0))


def build_generator(args: argparse.Namespace) -> BaseGenerator:
    if args.route == "transform":
        return RealCycleTransformGenerator(
            args.real_library_dir,
            sample_seconds=args.sample_seconds,
            time_scale_range=(args.time_scale_min, args.time_scale_max),
            power_scale_range=(args.power_scale_min, args.power_scale_max))
    if args.route in ("cvae", "wgan", "diffusion"):
        if not args.checkpoint_dir:
            raise SystemExit(f"--route {args.route} needs --checkpoint-dir")
        import torch

        payload = torch.load(Path(args.checkpoint_dir) / "model.pt",
                             map_location=args.device, weights_only=False)
        normalizer = {"length_scale": payload["length_scale"],
                      "mean_power_scale": payload["mean_power_scale"]}
        power_scale = float(payload.get("power_scale", 1.0))
        donors = load_real_reference(Path(args.real_library_dir),
                                     max_cycles=200)
        if args.route == "cvae":
            model = ConditionalWaveformCVAE(
                payload["wave_length"], payload["condition_dim"],
                latent_dim=payload["latent_dim"], width=payload["width"])
            model.load_state_dict(payload["model"])
            name = "b3_cvae"
        elif args.route == "wgan":
            model = WGANGenerator(
                payload["wave_length"], payload["condition_dim"],
                latent_dim=payload["latent_dim"], width=payload["width"])
            model.load_state_dict(payload["generator"])
            name = "b3_wgan"
        else:
            model = DiffusionDenoiser(
                payload["wave_length"], payload["condition_dim"],
                width=payload["width"])
            model.load_state_dict(payload["denoiser"])
            name = "b3_diffusion"

        class _SamplingAdapter:
            """Expose non-CVAE models through the decode() interface.

            For diffusion, the incoming ``latent`` only seeds the sampler
            (deterministic given the seed); the real generation runs the
            full ancestral loop over the noise schedule.
            """

            def __init__(self, inner, diffusion: GaussianDiffusion | None,
                         wave_length: int):
                self.inner = inner
                self.diffusion = diffusion
                self.wave_length = wave_length
                # Diffusion denoisers have no latent space; the latent is
                # only a deterministic seed source.
                self.latent_dim = getattr(inner, "latent_dim", 8)

            def to(self, device: str) -> "_SamplingAdapter":
                self.inner.to(device)
                return self

            def eval(self) -> "_SamplingAdapter":
                self.inner.eval()
                return self

            def decode(self, latent: "torch.Tensor", cond: "torch.Tensor"
                       ) -> "torch.Tensor":
                if self.diffusion is None:
                    return self.inner(latent, cond)
                self.inner.eval()
                torch.manual_seed(
                    int(torch.round(latent.sum().detach().cpu()
                                    * 1e6).abs().item()) % (1 << 31) + 1)
                initial = torch.randn(cond.shape[0], 1, self.wave_length,
                                      device=cond.device)
                return self.diffusion.sample(self.inner, initial.shape,
                                             cond, cond.device)

        if args.route == "wgan":
            model = _SamplingAdapter(model, None, payload["wave_length"])
        if args.route == "diffusion":
            diffusion = GaussianDiffusion(
                n_steps=payload.get("diffusion_steps", 500))
            model = _SamplingAdapter(model, diffusion,
                                     payload["wave_length"])
        bucketizer = LengthBucketizer.from_dict(payload["bucketizer"])
        return CVAESamplingGenerator(
            model, bucketizer,
            length_scale=normalizer["length_scale"],
            mean_power_scale=normalizer["mean_power_scale"],
            donor_waves=donors, sample_seconds=args.sample_seconds,
            device=args.device, power_scale=power_scale, name=name)
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
    ap.add_argument("--checkpoint-dir", default=None,
                    help="cvae route: trained checkpoint directory")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = ap.parse_args()

    generator = build_generator(args)
    records = generator.generate_dataset(
        Path(args.output_dir), count=args.count, seed=args.seed,
        sample_seconds=args.sample_seconds)
    print(f"[generate] route={args.route} count={len(records)} "
          f"seed={args.seed} -> {args.output_dir}")


if __name__ == "__main__":
    main()
