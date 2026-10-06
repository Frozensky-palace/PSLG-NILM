"""Tests for the CVAE family: bucketizer, training, sampling and B4 compose."""
from __future__ import annotations

import csv
import json
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


class B4RealDonorModeTests(unittest.TestCase):
    """B4-real ablation: donor mode swaps primitive waveforms only."""

    def _build(self, tmp: Path
               ) -> tuple[PrimitiveComposer, PrimitiveComposer, list, list]:
        library = _write_state_library(Path(tmp))
        waves, labels, meta = load_state_segments(library)
        rows = list(csv.DictReader(open(
            library / "state_inventory.csv", encoding="utf-8")))
        path_model = fit_path_model(rows)
        conditions, bucketizer, normalizer = (
            build_segment_conditions(waves, labels, 3))
        model = ConditionalWaveformCVAE(
            bucketizer.bucket_length(bucketizer.n_buckets - 1),
            condition_dim=conditions.shape[1], latent_dim=8, width=16)
        train_cvae(model, waves, conditions, bucketizer,
                   epochs=1, batch_size=8, seed=1)
        real = PrimitiveComposer(
            None, None, waves, labels, path_model, 3, None,
            primitive_source="donor",
            donor_block_ids=meta["state_block_ids"],
            donor_cycle_ids=meta["cycle_ids"])
        generated = PrimitiveComposer(
            model, bucketizer, waves, labels, path_model, 3, normalizer)
        return real, generated, waves, rows

    def test_donor_mode_outputs_real_donors_with_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            real, _, waves, rows = self._build(Path(tmp))
            block_wave = {row["state_block_id"]: wave
                          for row, wave in zip(rows, waves)}
            output = Path(tmp) / "b4r"
            records = real.generate_dataset(output, count=4, seed=21)
            for record in records:
                self.assertEqual(record.route, "B4R")
                self.assertEqual(record.validate(), [])
                expected = np.concatenate([
                    block_wave[segment.donor_state_block_id]
                    for segment in record.segments]).astype(np.float32)
                with np.load(output / "cycles"
                             / f"{record.synthetic_cycle_id}.npz") as data:
                    np.testing.assert_array_equal(data["appliance_w"],
                                                  expected)
                for segment in record.segments:
                    self.assertIsNotNone(segment.donor_cycle_id)

    def test_same_seed_keeps_paths_and_lengths_identical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            real, generated, _, _ = self._build(Path(tmp))
            real_records = real.generate_dataset(
                Path(tmp) / "real", count=5, seed=21)
            gen_records = generated.generate_dataset(
                Path(tmp) / "gen", count=5, seed=21)
            for real_record, gen_record in zip(real_records, gen_records):
                self.assertEqual(real_record.state_path,
                                 gen_record.state_path)
                self.assertEqual(
                    [s.target_samples for s in real_record.segments],
                    [s.target_samples for s in gen_record.segments])
                # The ablation's point: waveform content must differ.
                self.assertNotEqual(real_record.waveform_sha256,
                                    gen_record.waveform_sha256)

    def test_donor_mode_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            real, _, _, _ = self._build(Path(tmp))
            outputs = []
            for run in (1, 2):
                records = real.generate_dataset(
                    Path(tmp) / f"out_{run}", count=3, seed=21)
                dicts = [record.to_dict() for record in records]
                for entry in dicts:
                    entry.pop("created_utc")
                outputs.append(dicts)
            self.assertEqual(outputs[0], outputs[1])

    def test_constructor_guards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            real, _, waves, rows = self._build(Path(tmp))
            path_model = fit_path_model(rows)
            labels = [0] * len(waves)
            with self.assertRaises(ValueError):
                PrimitiveComposer(None, None, waves, labels,
                                  path_model, 3, None,
                                  primitive_source="model")
            with self.assertRaises(ValueError):
                PrimitiveComposer(None, None, waves, labels,
                                  path_model, 3, None,
                                  primitive_source="bogus")
            config = real.config()
            self.assertEqual(config["primitive_source"], "donor")
            self.assertNotIn("buckets", config)


class PrimitiveRealCliTests(unittest.TestCase):
    def test_real_mode_runs_without_checkpoint(self) -> None:
        from scripts.generate_primitive_cycles import main as compose_main
        with tempfile.TemporaryDirectory() as tmp:
            library = _write_state_library(Path(tmp))
            argv = ["generate_primitive_cycles.py",
                    "--primitive-source", "real",
                    "--state-library-dir", str(library),
                    "--output-dir", str(Path(tmp) / "out"),
                    "--count", "3", "--seed", "21"]
            with patch.object(sys, "argv", argv):
                compose_main()
            summary = json.loads((Path(tmp) / "out" / "generation_summary.json")
                                 .read_text(encoding="utf-8"))
            self.assertEqual(len(summary["records"]), 3)
            self.assertTrue(all(record["route"] == "B4R"
                                for record in summary["records"]))

    def test_cvae_mode_requires_checkpoint(self) -> None:
        from scripts.generate_primitive_cycles import main as compose_main
        with tempfile.TemporaryDirectory() as tmp:
            library = _write_state_library(Path(tmp))
            argv = ["generate_primitive_cycles.py",
                    "--state-library-dir", str(library),
                    "--output-dir", str(Path(tmp) / "out"),
                    "--count", "3", "--seed", "21"]
            with patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit):
                    compose_main()


if __name__ == "__main__":
    unittest.main()
