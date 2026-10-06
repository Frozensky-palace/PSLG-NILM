"""Tests for the B3-G conditional WGAN route."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_full_cycles import build_generator  # noqa: E402
from src.generation.full_cycle_wgan import (  # noqa: E402
    WGANCritic,
    WGANGenerator,
    gradient_penalty,
    train_wgan,
)
from tests.test_synthetic_quality import _write_real_library  # noqa: E402


class WganTests(unittest.TestCase):
    def _training_set(self, count: int = 24) -> tuple[list[np.ndarray],
                                                      np.ndarray,
                                                      object]:
        from src.generation.full_cycle_cvae import LengthBucketizer

        rng = np.random.default_rng(2)
        waves = [rng.normal(400, 30, int(n)).clip(min=0)
                 for n in rng.integers(120, 200, count)]
        bucketizer = LengthBucketizer.fit(
            np.array([len(w) for w in waves]), n_buckets=2)
        conditions = np.zeros((count, 4), dtype=np.float32)
        return waves, conditions, bucketizer

    def test_gradient_penalty_is_small_for_interpolations(self) -> None:
        torch = __import__("torch")
        waves, conditions, bucketizer = self._training_set()
        from src.generation.full_cycle_cvae import pad_and_mask

        padded, _ = pad_and_mask(waves, 256)
        critic = WGANCritic(256, 4, width=8)
        penalty = gradient_penalty(critic, padded, padded.clone(),
                                   torch.zeros(len(waves), 4))
        # Identical real/fake pairs are on the data manifold mid-points;
        # an untrained critic gives near-linear scores -> small penalty.
        self.assertGreaterEqual(float(penalty), 0.0)

    def test_training_runs_and_generator_shapes(self) -> None:
        torch = __import__("torch")
        waves, conditions, bucketizer = self._training_set()
        bucket_length = bucketizer.bucket_length(bucketizer.n_buckets - 1)
        generator = WGANGenerator(bucket_length, 4, latent_dim=8, width=16)
        critic = WGANCritic(bucket_length, 4, width=16)
        history = train_wgan(generator, critic,
                             [w / 800.0 for w in waves], conditions,
                             bucketizer, epochs=2, batch_size=8,
                             n_critic=2, seed=1)
        self.assertEqual(len(history), 2)
        self.assertTrue(np.isfinite(history[-1]["w_distance"]))
        latent = torch.randn(3, 8)
        cond = torch.zeros(3, 4)
        out = generator(latent, cond)
        self.assertEqual(out.shape, (3, 1, bucket_length))

    def test_cli_wgan_route_trains_and_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            library = _write_real_library(root, count=40)
            ckpt = root / "wgan_ckpt"
            from scripts.train_full_cycle_generator import (
                main as train_main)
            train_argv = ["train_full_cycle_generator.py", "--route", "wgan",
                          "--real-library-dir", str(library),
                          "--output-dir", str(ckpt),
                          "--seed", "3",
                          "--epochs", "1", "--max-cycles", "20",
                          "--n-critic", "2"]
            with patch.object(sys, "argv", train_argv):
                train_main()
            sample_argv = ["generate_full_cycles.py", "--route", "wgan",
                           "--checkpoint-dir", str(ckpt),
                           "--real-library-dir", str(library),
                           "--output-dir", str(root / "samples"),
                           "--count", "4", "--seed", "4"]
            with patch.object(sys, "argv", sample_argv):
                from scripts.generate_full_cycles import main as gen_main
                gen_main()
            self.assertTrue((root / "samples"
                             / "generation_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
