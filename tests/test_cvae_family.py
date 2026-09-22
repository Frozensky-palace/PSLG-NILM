"""Tests for the CVAE family: bucketizer, training, sampling and B4 compose."""
from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.full_cycle_cvae import (  # noqa: E402
    ConditionalWaveformCVAE,
    LengthBucketizer,
    pad_and_mask,
    sample_cvae,
    train_cvae,
)
from src.generation.primitive_cvae import (  # noqa: E402
    PrimitiveComposer,
    build_segment_conditions,
    fit_path_model,
    load_state_segments,
)


def _write_state_library(root: Path, n_states: int = 3,
                         blocks_per_state: int = 12) -> Path:
    """Tiny state library fixture with a connected transition structure."""
    import csv

    library = root / "state_library"
    library.mkdir(parents=True)
    rng = np.random.default_rng(4)
    rows = []
    power_parts: list[np.ndarray] = []
    offsets = [0]
    cycle_count = 8
    for block in range(n_states * blocks_per_state):
        state = block % n_states
        length = int(rng.integers(72, 200))
        base = 200.0 + 600.0 * state
        power_parts.append(rng.normal(base, 20, length).clip(min=0))
        offsets.append(offsets[-1] + length)
        cycle_id = f"c{block % cycle_count}"
        rows.append({
            "state_block_id": f"b{block:04d}", "state_label": state,
            "cycle_id": cycle_id, "start_sample": block * 10,
            "duration_seconds": length * 6,
            "waveform_offset_start": offsets[-2],
            "waveform_offset_end": offsets[-1],
            "partition": "train",
            "previous_state_label": (block - 1) % n_states
            if block % cycle_count else "",
            "next_state_label": (block + 1) % n_states
            if (block + 1) % cycle_count else "",
        })
    np.savez_compressed(
        library / "state_waveforms.npz",
        power_w=np.concatenate(power_parts),
        offsets=np.array(offsets),
        state_block_id=np.array([r["state_block_id"] for r in rows]))
    with open(library / "state_inventory.csv", "w", encoding="utf-8",
              newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return library


class LengthBucketizerTests(unittest.TestCase):
    def test_bucket_lengths_are_multiples_of_eight(self) -> None:
        bucketizer = LengthBucketizer.fit(
            np.array([100, 200, 333, 640, 900]), n_buckets=4)
        for boundary in bucketizer.boundaries:
            self.assertEqual(boundary % 8, 0)

    def test_encode_covers_all_lengths(self) -> None:
        bucketizer = LengthBucketizer.fit(
            np.array([100, 200, 333, 640, 900]), n_buckets=4)
        for length in range(1, 1000):
            self.assertTrue(0 <= bucketizer.encode(length)
                            < bucketizer.n_buckets)

    def test_roundtrip_through_dict(self) -> None:
        bucketizer = LengthBucketizer([128, 256, 512])
        restored = LengthBucketizer.from_dict(bucketizer.to_dict())
        self.assertEqual(restored.boundaries, [128, 256, 512])


class CvaeSmokeTests(unittest.TestCase):
    def _tiny_training_set(self, count: int = 32
                           ) -> tuple[list[np.ndarray], np.ndarray,
                                      LengthBucketizer]:
        rng = np.random.default_rng(0)
        waves = [rng.normal(400, 40, int(n)).clip(min=0)
                 for n in rng.integers(120, 240, count)]
        bucketizer = LengthBucketizer.fit(
            np.array([len(w) for w in waves]), n_buckets=2)
        conditions = np.stack([
            np.concatenate([np.array([len(w) / 256.0,
                                      float(w.mean()) / 500.0],
                                     dtype=np.float32),
                            np.zeros(2, dtype=np.float32)])
            for w in waves])
        return waves, conditions, bucketizer

    def test_training_reduces_loss(self) -> None:
        waves, conditions, bucketizer = self._tiny_training_set()
        model = ConditionalWaveformCVAE(
            bucketizer.bucket_length(bucketizer.n_buckets - 1),
            condition_dim=conditions.shape[1], latent_dim=8, width=16)
        history = train_cvae(model, waves, conditions, bucketizer,
                             epochs=8, batch_size=8, seed=1)
        self.assertLess(history[-1]["loss"], history[0]["loss"])

    def test_sampling_shapes_and_nonnegative(self) -> None:
        waves, conditions, bucketizer = self._tiny_training_set()
        model = ConditionalWaveformCVAE(
            bucketizer.bucket_length(bucketizer.n_buckets - 1),
            condition_dim=conditions.shape[1], latent_dim=8, width=16)
        train_cvae(model, waves, conditions, bucketizer,
                   epochs=3, batch_size=8, seed=1)
        lengths = [130, 210, 90]
        samples = sample_cvae(model, bucketizer, conditions[:3], lengths,
                              seed=3)
        for sample, length in zip(samples, lengths):
            self.assertEqual(len(sample), length)
            self.assertFalse((sample < 0).any())

    def test_pad_and_mask_shapes(self) -> None:
        padded, mask = pad_and_mask([np.ones(10), np.ones(4) * 2], 16)
        self.assertEqual(padded.shape, (2, 1, 16))
        self.assertEqual(int(mask[0].sum()), 10)
        self.assertEqual(int(mask[1].sum()), 4)


class PrimitivePipelineTests(unittest.TestCase):
    def test_load_fit_compose_is_schema_valid_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            library = _write_state_library(Path(tmp))
            waves, labels, _ = load_state_segments(library)
            self.assertEqual(len(waves), len(labels))
            rows = list(csv.DictReader(open(
                library / "state_inventory.csv", encoding="utf-8")))
            path_model = fit_path_model(rows)
            conditions, bucketizer, normalizer = (
                build_segment_conditions(waves, labels, 3))
            model = ConditionalWaveformCVAE(
                bucketizer.bucket_length(bucketizer.n_buckets - 1),
                condition_dim=conditions.shape[1], latent_dim=8, width=16)
            train_cvae(model, waves, conditions, bucketizer,
                       epochs=2, batch_size=8, seed=1)
            composer = PrimitiveComposer(model, bucketizer, waves, labels,
                                         path_model, 3, normalizer)
            outputs = []
            for run_seed in (21, 21):
                records = composer.generate_dataset(
                    Path(tmp) / f"out_{run_seed}", count=3, seed=run_seed)
                dicts = [r.to_dict() for r in records]
                for entry, record in zip(dicts, records):
                    self.assertEqual(record.route, "B4")
                    self.assertTrue(record.waveform_sha256)
                    self.assertEqual(record.validate(), [])
                    entry.pop("created_utc")  # wall-clock, not determinism
                outputs.append(dicts)
            self.assertEqual(outputs[0], outputs[1])


if __name__ == "__main__":
    unittest.main()
