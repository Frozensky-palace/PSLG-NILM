"""Tests for the B4WD segment-level WGAN/diffusion primitive backends.

Covers the preregistered B4WD invariants (reports/b4wd/
b4wd_protocol_prereg_v1.md §2/§7): masked training semantics per arm,
backend dispatch that leaves the numpy rng sequence untouched (same-seed
state paths and segment lengths equal across backends), per-state scaling
and peak caps, route labels, rng_seed persistence, and CLI wiring.
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.full_cycle_cvae import (  # noqa: E402
    ConditionalWaveformCVAE,
    LengthBucketizer,
    train_cvae,
)
from src.generation.full_cycle_diffusion import (  # noqa: E402
    DiffusionDenoiser,
    GaussianDiffusion,
    masked_noise_mse,
    train_primitive_diffusion,
)
from src.generation.full_cycle_wgan import (  # noqa: E402
    WGANCritic,
    WGANGenerator,
    apply_input_mask,
    train_primitive_wgan,
)
from src.generation.primitive_cvae import (  # noqa: E402
    PrimitiveComposer,
    build_segment_conditions,
    fit_path_model,
    load_state_segments,
)
from tests.test_cvae_family import _write_state_library  # noqa: E402
from tests.test_full_cycle_diffusion import _write_real_library  # noqa: E402


def _segment_dataset(count: int = 24, seed: int = 7
                     ) -> tuple[list[np.ndarray], np.ndarray, list[int],
                                LengthBucketizer]:
    rng = np.random.default_rng(seed)
    waves = [rng.normal(400, 30, int(n)).clip(min=0)
             for n in rng.integers(40, 60, count)]
    bucketizer = LengthBucketizer.fit(
        np.array([len(w) for w in waves]), n_buckets=1)
    conditions = np.zeros((count, 4), dtype=np.float32)
    labels = [int(i % 2) for i in range(count)]
    return waves, conditions, labels, bucketizer


class MaskHelperTests(unittest.TestCase):
    def test_apply_input_mask_zeros_padding(self) -> None:
        wave = torch.ones(2, 1, 8)
        mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
        masked = apply_input_mask(wave, mask)
        self.assertEqual(float(masked[0, 0, 2]), 1.0)
        self.assertEqual(float(masked[0, 0, 3]), 0.0)
        self.assertEqual(float(masked[1, 0, 7]), 1.0)

    def test_masked_noise_mse_averages_valid_points_only(self) -> None:
        predicted = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        noise = torch.tensor([[[0.0, 0.0, 0.0, 0.0]]])
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        # Only the first two points are valid: (1 + 4) / 2, ignoring 9 + 16.
        self.assertAlmostEqual(float(masked_noise_mse(predicted, noise, mask)),
                               2.5)


class TrainPrimitiveWganTests(unittest.TestCase):
    def test_critic_never_sees_padding(self) -> None:
        waves, conditions, labels, bucketizer = _segment_dataset()
        bucket_length = bucketizer.bucket_length(bucketizer.n_buckets - 1)
        max_len = max(len(w) for w in waves)
        self.assertLess(max_len, bucket_length,
                        "fixture must leave a zero tail in every bucket")
        seen: list[torch.Tensor] = []

        class RecordingCritic(WGANCritic):
            def forward(self, wave, cond):
                seen.append(wave.detach().clone())
                return super().forward(wave, cond)

        generator = WGANGenerator(bucket_length, 4, latent_dim=8, width=8)
        critic = RecordingCritic(bucket_length, 4, width=8)
        train_primitive_wgan(generator, critic,
                             [w / 800.0 for w in waves], conditions,
                             bucketizer, labels, epochs=1, batch_size=8,
                             n_critic=1, seed=1)
        self.assertTrue(seen)
        # Every critic input (masked real, masked fake, GP interpolates)
        # must be zero on the columns beyond the longest real segment.
        for tensor in seen:
            tail = tensor[:, :, max_len:]
            self.assertTrue(torch.equal(tail, torch.zeros_like(tail)),
                            "critic saw non-zero padding")

    def test_history_records_per_state_w_distance(self) -> None:
        waves, conditions, labels, bucketizer = _segment_dataset()
        bucket_length = bucketizer.bucket_length(bucketizer.n_buckets - 1)
        generator = WGANGenerator(bucket_length, 4, latent_dim=8, width=8)
        critic = WGANCritic(bucket_length, 4, width=8)
        history = train_primitive_wgan(generator, critic,
                                       [w / 800.0 for w in waves],
                                       conditions, bucketizer, labels,
                                       epochs=1, batch_size=8, n_critic=1,
                                       seed=1)
        self.assertEqual(len(history), 1)
        self.assertTrue(np.isfinite(history[-1]["w_distance"]))
        self.assertEqual(set(history[-1]["w_distance_by_state"]),
                         {"0", "1"})


class TrainPrimitiveDiffusionTests(unittest.TestCase):
    def test_training_reduces_masked_noise_mse(self) -> None:
        waves, conditions, _, bucketizer = _segment_dataset()
        bucket_length = bucketizer.bucket_length(bucketizer.n_buckets - 1)
        denoiser = DiffusionDenoiser(bucket_length, 4, width=8)
        diffusion = GaussianDiffusion(n_steps=40)
        history = train_primitive_diffusion(
            denoiser, diffusion, [w / 800.0 for w in waves], conditions,
            bucketizer, epochs=3, batch_size=8, seed=1)
        self.assertEqual(len(history), 3)
        self.assertLess(history[-1]["noise_mse"], history[0]["noise_mse"])

    @unittest.skipIf(not torch.cuda.is_available(), "cuda not available")
    def test_diffusion_trains_and_samples_on_cuda(self) -> None:
        # fd3a657 regression guard: the time embedding must follow the
        # timestep tensor's device for segment-level training too.
        waves, conditions, _, bucketizer = _segment_dataset()
        bucket_length = bucketizer.bucket_length(bucketizer.n_buckets - 1)
        denoiser = DiffusionDenoiser(bucket_length, 4, width=8)
        diffusion = GaussianDiffusion(n_steps=40)
        history = train_primitive_diffusion(
            denoiser, diffusion, [w / 800.0 for w in waves], conditions,
            bucketizer, epochs=1, batch_size=8, seed=1, device="cuda")
        self.assertTrue(np.isfinite(history[-1]["noise_mse"]))
        cond = torch.zeros(2, 4, device="cuda")
        samples = diffusion.sample(denoiser, (2, 1, bucket_length), cond,
                                   "cuda")
        self.assertTrue(torch.isfinite(samples).all())


def _trained_backends(tmp: Path
                      ) -> tuple[dict[str, PrimitiveComposer], dict, list]:
    """Tiny library + 1-epoch cvae/wgan/diffusion composers (same seed)."""
    library = _write_state_library(tmp)
    waves, labels, _ = load_state_segments(library)
    rows = list(csv.DictReader(open(
        library / "state_inventory.csv", encoding="utf-8")))
    path_model = fit_path_model(rows)
    n_states = 3
    conditions, bucketizer, normalizer = build_segment_conditions(
        waves, labels, n_states)
    bucket_length = bucketizer.bucket_length(bucketizer.n_buckets - 1)
    cond_dim = int(conditions.shape[1])

    # Seed before construction (as the CLI does): weight init draws from the
    # global torch RNG, which is process-entropy-seeded otherwise — the
    # source of the cross-process flakiness (unlucky inits NaN'd or
    # zero-clipped identically across backends).
    torch.manual_seed(1)
    cvae = ConditionalWaveformCVAE(bucket_length, condition_dim=cond_dim,
                                   latent_dim=8, width=16)
    generator = WGANGenerator(bucket_length, cond_dim, latent_dim=8,
                              width=16)
    critic = WGANCritic(bucket_length, cond_dim, width=16)
    denoiser = DiffusionDenoiser(bucket_length, cond_dim, width=16)
    diffusion = GaussianDiffusion(n_steps=40)
    # Train on scaled waves like the real routes (train_full_cycle_generator
    # always divides by a power scale first); tiny nets fed raw watts blow up
    # intermittently, and NaN decodes used to fail determinism via
    # nan != nan rather than the guard below.
    scale = 800.0
    scaled = [w / scale for w in waves]
    train_cvae(cvae, scaled, conditions, bucketizer, epochs=1, batch_size=8,
               seed=1)
    train_primitive_wgan(generator, critic, scaled, conditions, bucketizer,
                         labels, epochs=1, batch_size=8, n_critic=1, seed=1)
    train_primitive_diffusion(denoiser, diffusion, scaled, conditions,
                              bucketizer, epochs=1, batch_size=8, seed=1)

    composers = {
        "cvae": PrimitiveComposer(cvae, bucketizer, waves, labels,
                                  path_model, n_states, normalizer,
                                  power_scale=scale),
        "wgan": PrimitiveComposer(generator, bucketizer, waves, labels,
                                  path_model, n_states, normalizer,
                                  power_scale=scale,
                                  model_backend="wgan"),
        "diffusion": PrimitiveComposer(denoiser, bucketizer, waves, labels,
                                       path_model, n_states, normalizer,
                                       power_scale=scale,
                                       model_backend="diffusion",
                                       diffusion=diffusion),
    }
    return composers, waves, labels


class BackendEquivalenceTests(unittest.TestCase):
    """Same seed: only the waveform content differs across backends."""

    ROUTES = {"cvae": "B4", "wgan": "B4WGAN", "diffusion": "B4DIFF"}

    def test_paths_lengths_and_routes_match_across_backends(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            composers, _, _ = _trained_backends(Path(tmp))
            outputs = {}
            for name, composer in composers.items():
                outputs[name] = composer.generate_dataset(
                    Path(tmp) / f"out_{name}", count=5, seed=21)
            reference = outputs["cvae"]
            for record in reference:
                self.assertEqual(record.validate(), [])
            hashes = {name: {r.synthetic_cycle_id: r.waveform_sha256
                             for r in records}
                      for name, records in outputs.items()}
            for name, records in outputs.items():
                for ref, got in zip(reference, records):
                    self.assertEqual(got.route, self.ROUTES[name])
                    self.assertEqual(got.state_path, ref.state_path)
                    self.assertEqual(
                        [s.target_samples for s in got.segments],
                        [s.target_samples for s in ref.segments])
                    self.assertEqual(
                        got.conditions["rng_seeds"],
                        ref.conditions["rng_seeds"])
                    self.assertEqual(len(got.conditions["rng_seeds"]),
                                     len(got.state_path))
                    # The arms' point: decode content must differ pairwise.
                    for other in hashes:
                        if other != name:
                            self.assertNotEqual(
                                hashes[name][ref.synthetic_cycle_id],
                                hashes[other][ref.synthetic_cycle_id],
                                f"{name} vs {other} decoded identical waves")

    def test_backend_determinism(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            composers, _, _ = _trained_backends(Path(tmp))
            for name, composer in composers.items():
                runs = []
                for run in (1, 2):
                    records = composer.generate_dataset(
                        Path(tmp) / f"det_{name}_{run}", count=3, seed=21)
                    dicts = [record.to_dict() for record in records]
                    for entry in dicts:
                        entry.pop("created_utc")
                    runs.append(dicts)
                self.assertEqual(runs[0], runs[1], name)


class DiffusionScaleAndCapTests(unittest.TestCase):
    def test_per_state_scale_and_peak_caps_bound_segments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            composers, _, _ = _trained_backends(Path(tmp))
            composer = composers["diffusion"]
            n_states = composer.n_states
            power_scales = {state: 100.0 for state in range(n_states)}
            peak_caps = {state: 37.5 for state in range(n_states)}
            scaled = PrimitiveComposer(
                composer.model, composer.bucketizer, composer.donor_waves,
                composer.donor_labels, composer.path_model, n_states,
                composer.normalizer, model_backend="diffusion",
                power_scales=power_scales, peak_caps=peak_caps,
                diffusion=composer.diffusion)
            output = Path(tmp) / "capped"
            records = scaled.generate_dataset(output, count=4, seed=21)
            for record in records:
                with np.load(output / "cycles"
                             / f"{record.synthetic_cycle_id}.npz") as data:
                    power = data["appliance_w"].astype(np.float64)
                cursor = 0
                for segment in record.segments:
                    piece = power[cursor:cursor + segment.actual_samples]
                    cursor += segment.actual_samples
                    self.assertTrue((piece >= 0).all())
                    self.assertLessEqual(
                        float(piece.max()),
                        peak_caps[segment.state_label],
                        f"segment state {segment.state_label} over cap")


class ConstructorGuardTests(unittest.TestCase):
    def test_backend_guards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            composers, waves, labels = _trained_backends(Path(tmp))
            rows = list(csv.DictReader(open(
                Path(tmp) / "state_library" / "state_inventory.csv",
                encoding="utf-8")))
            path_model = fit_path_model(rows)
            with self.assertRaises(ValueError):
                PrimitiveComposer(composers["wgan"].model,
                                  composers["wgan"].bucketizer, waves,
                                  labels, path_model, 3, None,
                                  model_backend="bogus")
            with self.assertRaises(ValueError):
                PrimitiveComposer(composers["diffusion"].model,
                                  composers["diffusion"].bucketizer, waves,
                                  labels, path_model, 3, None,
                                  model_backend="diffusion",
                                  diffusion=None)
            with self.assertRaises(ValueError):
                PrimitiveComposer(None, None, waves, labels, path_model, 3,
                                  None, primitive_source="donor",
                                  model_backend="wgan")
            config = composers["diffusion"].config()
            self.assertEqual(config["model_backend"], "diffusion")


class PrimitiveBackendCliTests(unittest.TestCase):
    def _run_cli(self, source: str, route: str, tmp: Path) -> None:
        from scripts.train_full_cycle_generator import main as train_main
        from scripts.generate_primitive_cycles import main as gen_main
        library = _write_state_library(tmp)
        ckpt = tmp / f"ckpt_{source}"
        train_argv = ["train_full_cycle_generator.py", "--route", route,
                      "--state-library-dir", str(library),
                      "--output-dir", str(ckpt), "--seed", "3",
                      "--epochs", "1"]
        if source == "diffusion":
            train_argv += ["--diffusion-steps", "40"]
        else:
            train_argv += ["--n-critic", "1"]
        with patch.object(sys, "argv", train_argv):
            train_main()
        # Prereg §3 pins Adam(1e-4) for the wgan arm; the CLI default must
        # resolve per route and the effective lr must land in config.json.
        config = json.loads((ckpt / "config.json").read_text(encoding="utf-8"))
        expected_lr = {"wgan": 1e-4, "diffusion": 1e-3}[source]
        self.assertEqual(config["learning_rate"], expected_lr)
        gen_argv = ["generate_primitive_cycles.py",
                    "--primitive-source", source,
                    "--checkpoint-dir", str(ckpt),
                    "--state-library-dir", str(library),
                    "--output-dir", str(tmp / f"out_{source}"),
                    "--count", "3", "--seed", "21"]
        with patch.object(sys, "argv", gen_argv):
            gen_main()
        summary = json.loads((tmp / f"out_{source}" / "generation_summary.json"
                              ).read_text(encoding="utf-8"))
        routes = {"wgan": "B4WGAN", "diffusion": "B4DIFF"}
        self.assertTrue(all(record["route"] == routes[source]
                            for record in summary["records"]))
        if source == "diffusion":
            # Prereg §4 evidence: 贴上限占比 + 截断触发率 must be persisted.
            cap = json.loads((tmp / f"out_{source}" / "peak_cap_report.json"
                              ).read_text(encoding="utf-8"))
            self.assertEqual(cap["n_cycles"], 3)
            self.assertIn("cap_pinned_share", cap)
            self.assertIn("cap_trigger_rate", cap)
            self.assertTrue(0.0 <= cap["cap_pinned_share"] <= 1.0)
            self.assertTrue(0.0 <= cap["cap_trigger_rate"] <= 1.0)

    def test_cli_wgan_route_trains_and_generates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._run_cli("wgan", "primitive-wgan", Path(tmp))

    def test_cli_diffusion_route_trains_and_generates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._run_cli("diffusion", "primitive-diffusion", Path(tmp))

    def test_wrong_checkpoint_route_is_rejected(self) -> None:
        from scripts.generate_primitive_cycles import main as gen_main
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            library = _write_state_library(tmp)
            ckpt = tmp / "ckpt"
            train_argv = ["train_full_cycle_generator.py", "--route",
                          "primitive-wgan",
                          "--state-library-dir", str(library),
                          "--output-dir", str(ckpt), "--seed", "3",
                          "--epochs", "1", "--n-critic", "1"]
            with patch.object(sys, "argv", train_argv):
                from scripts.train_full_cycle_generator import (
                    main as train_main)
                train_main()
            gen_argv = ["generate_primitive_cycles.py",
                        "--primitive-source", "cvae",
                        "--checkpoint-dir", str(ckpt),
                        "--state-library-dir", str(library),
                        "--output-dir", str(tmp / "out"),
                        "--count", "3", "--seed", "21"]
            with patch.object(sys, "argv", gen_argv):
                with self.assertRaises(SystemExit):
                    gen_main()


class DiversityCoverageTests(unittest.TestCase):
    def test_diversity_max_cycles_overrides_default_cap(self) -> None:
        from src.generation.base_generator import ConstantGenerator
        from src.validation.synthetic_quality import (
            evaluate_synthetic_dataset)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real = _write_real_library(root, count=8)
            synth = root / "synth"
            ConstantGenerator(length_samples=50,
                              power_w=400.0).generate_dataset(
                synth, count=4, seed=1)
            default = evaluate_synthetic_dataset(synth, real)
            self.assertEqual(default["diversity"]["sampled_cycles"], 4)
            capped = evaluate_synthetic_dataset(synth, real,
                                                diversity_max_cycles=3)
            self.assertEqual(capped["diversity"]["sampled_cycles"], 3)


class FiniteGuardTests(unittest.TestCase):
    def test_non_finite_power_is_rejected(self) -> None:
        # NaN slips past the comparison gates (NaN < 0 and NaN > cap are
        # both False); generate_dataset must stop it loudly.
        from src.generation.base_generator import ConstantGenerator

        class NanGenerator(ConstantGenerator):
            def generate_cycle(self, synthetic_cycle_id, seed, rng):
                power, record = super().generate_cycle(
                    synthetic_cycle_id, seed, rng)
                power = power.copy()
                power[3] = float("nan")
                return power, record

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                NanGenerator(length_samples=10).generate_dataset(
                    Path(tmp) / "nan_out", count=1, seed=1)


if __name__ == "__main__":
    unittest.main()
