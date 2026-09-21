"""Tests for the B3-D conditional diffusion route."""
from __future__ import annotations

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

from src.generation.full_cycle_cvae import LengthBucketizer  # noqa: E402
from src.generation.full_cycle_diffusion import (  # noqa: E402
    DiffusionDenoiser,
    GaussianDiffusion,
    sinusoidal_embedding,
    train_diffusion,
)


def _write_real_library(root: Path, count: int = 30, seed: int = 5,
                        mean_w: float = 500.0) -> Path:
    """Minimal real-cycle library fixture (mirrors the batch-0 helper)."""
    import pandas as pd

    library = root / "real_library"
    (library / "cycles").mkdir(parents=True)
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(count):
        samples = int(rng.integers(250, 350))
        power = rng.normal(mean_w, 50.0, samples).clip(min=0.0)
        relative = f"cycles/cycle_{index:04d}.npz"
        np.savez_compressed(library / relative, appliance_w=power)
        rows.append({"cycle_id": f"c{index}", "partition": "train",
                     "path": relative})
    pd.DataFrame(rows).to_csv(library / "real_cycle_library.csv",
                              index=False)
    return library


class DiffusionTests(unittest.TestCase):
    def _setup(self, count: int = 24, steps: int = 100
               ) -> tuple[DiffusionDenoiser, GaussianDiffusion,
                          list[np.ndarray], np.ndarray, LengthBucketizer]:
        rng = np.random.default_rng(6)
        waves = [rng.normal(400, 30, int(n)).clip(min=0)
                 for n in rng.integers(120, 200, count)]
        bucketizer = LengthBucketizer.fit(
            np.array([len(w) for w in waves]), n_buckets=2)
        conditions = np.zeros((count, 4), dtype=np.float32)
        denoiser = DiffusionDenoiser(
            bucketizer.bucket_length(bucketizer.n_buckets - 1), 4, width=8)
        diffusion = GaussianDiffusion(n_steps=steps)
        return denoiser, diffusion, waves, conditions, bucketizer

    def test_sinusoidal_embedding_shape(self) -> None:
        embedding = sinusoidal_embedding(torch.tensor([0, 5, 10]), 16)
        self.assertEqual(embedding.shape, (3, 16))

    def test_add_noise_preserves_scale(self) -> None:
        denoiser, diffusion, waves, _, _ = self._setup()
        wave = torch.ones(4, 1, 256) * 0.5
        noisy, noise = diffusion.add_noise(wave, torch.tensor([0, 50, 99,
                                                               99]))
        self.assertTrue(torch.isfinite(noisy).all())
        self.assertEqual(noise.shape, noisy.shape)

    def test_training_reduces_noise_mse_and_samples_are_finite(self) -> None:
        denoiser, diffusion, waves, conditions, bucketizer = self._setup(
            steps=40)
        history = train_diffusion(denoiser, diffusion, waves, conditions,
                                  bucketizer, epochs=3, batch_size=8,
                                  seed=1)
        self.assertEqual(len(history), 3)
        self.assertLess(history[-1]["noise_mse"], history[0]["noise_mse"])
        wave_length = bucketizer.bucket_length(bucketizer.n_buckets - 1)
        cond = torch.zeros(2, 4)
        samples = diffusion.sample(denoiser, (2, 1, wave_length), cond,
                                   "cpu")
        self.assertEqual(samples.shape, (2, 1, wave_length))
        self.assertTrue(torch.isfinite(samples).all())

    def test_cli_route_trains_and_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            library = _write_real_library(root, count=30)
            from scripts.train_full_cycle_generator import (
                main as train_main)
            train_argv = ["train_full_cycle_generator.py",
                          "--route", "diffusion",
                          "--real-library-dir", str(library),
                          "--output-dir", str(root / "ckpt"),
                          "--seed", "2", "--epochs", "1",
                          "--max-cycles", "20", "--diffusion-steps", "40"]
            with patch.object(sys, "argv", train_argv):
                train_main()
            ckpt_payload_ok = (root / "ckpt" / "model.pt").exists()
            self.assertTrue(ckpt_payload_ok)
            from scripts.generate_full_cycles import main as gen_main
            gen_argv = ["generate_full_cycles.py", "--route", "diffusion",
                        "--checkpoint-dir", str(root / "ckpt"),
                        "--real-library-dir", str(library),
                        "--output-dir", str(root / "samples"),
                        "--count", "3", "--seed", "4"]
            with patch.object(sys, "argv", gen_argv):
                gen_main()
            self.assertTrue((root / "samples"
                             / "generation_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
