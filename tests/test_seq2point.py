"""Unit tests for the Phase B Seq2Point and generation infrastructure."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.base_generator import ConstantGenerator  # noqa: E402
from src.generation.provenance import (  # noqa: E402
    append_registry_row,
    canonical_config_hash,
    environment_record,
    sha256_of_bytes,
    sha256_of_file,
)
from src.generation.schema import (  # noqa: E402
    StateSegmentRecord,
    SyntheticCycleRecord,
)
from src.nilm.checkpoint import (  # noqa: E402
    capture_rng_states,
    load_checkpoint,
    restore_rng_states,
    save_checkpoint,
)
from src.nilm.seq2point import Seq2PointCNN, parameter_count  # noqa: E402
from src.validation.statistics import (  # noqa: E402
    audit_identical_indices,
    multi_seed_summary,
    paired_bootstrap_ci,
    per_cycle_metrics,
)


class Seq2PointModelTests(unittest.TestCase):
    def test_output_shape_and_determinism(self):
        torch.manual_seed(0)
        model = Seq2PointCNN(window_length=64)
        window = torch.randn(4, 64)
        first = model(window)
        self.assertEqual(tuple(first.shape), (4,))
        second = model(window)
        torch.testing.assert_close(first, second)

    def test_parameter_count_in_documented_band(self):
        torch.manual_seed(0)
        count = parameter_count(Seq2PointCNN())
        self.assertGreater(count, 100_000)
        self.assertLess(count, 1_000_000)


class CheckpointTests(unittest.TestCase):
    def _tiny_setup(self, directory: Path):
        path = directory / "ckpt.pt"
        model = torch.nn.Linear(3, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        return path, model, optimizer

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, model, optimizer = self._tiny_setup(Path(temporary))
            save_checkpoint(path, model=model, optimizer=optimizer, epoch=3,
                            best_val_mae=1.5, config={"arm": "B0"},
                            rng_states=capture_rng_states())
            clone = torch.nn.Linear(3, 1)
            payload = load_checkpoint(path, model=clone, optimizer=optimizer)
            self.assertEqual(payload["epoch"], 3)
            self.assertEqual(payload["best_val_mae"], 1.5)
            for original, copied in zip(model.parameters(), clone.parameters()):
                torch.testing.assert_close(original, copied)

    def test_torch_generator_reproducible_after_restore(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, model, optimizer = self._tiny_setup(Path(temporary))
            save_checkpoint(path, model=model, optimizer=optimizer, epoch=1,
                            best_val_mae=0.0, config={},
                            rng_states=capture_rng_states())
            torch.rand(3)  # advance the global torch RNG
            restore_rng_states(load_checkpoint(path, model=model)["rng_states"])
            sequence_a = torch.rand(3)
            restore_rng_states(load_checkpoint(path, model=model)["rng_states"])
            sequence_b = torch.rand(3)
            torch.testing.assert_close(sequence_a, sequence_b)


class GenerationSchemaTests(unittest.TestCase):
    def _record(self, **overrides) -> SyntheticCycleRecord:
        segment = StateSegmentRecord(
            state_label=1, target_duration_seconds=600.0,
            actual_duration_seconds=600.0, target_samples=100,
            actual_samples=100, mean_power_w=200.0, energy_wh=33.3)
        payload = dict(
            synthetic_cycle_id="synthetic_0000", route="B4", seed=17,
            generator_name="test", generator_version="1",
            checkpoint_sha256=None, conditions={},
            state_path=[1], segments=[segment], boundary_treatment="none",
            sample_seconds=6, created_utc="2026-09-20T00:00:00Z",
            waveform_sha256="abc")
        payload.update(overrides)
        return SyntheticCycleRecord(**payload)

    def test_valid_record_passes(self):
        self.assertEqual(self._record().validate(), [])

    def test_invalid_route_rejected(self):
        errors = self._record(route="B9").validate()
        self.assertTrue(any("route" in error for error in errors))

    def test_duration_sample_mismatch_rejected(self):
        bad_segment = StateSegmentRecord(
            state_label=0, target_duration_seconds=60.0,
            actual_duration_seconds=60.0, target_samples=10,
            actual_samples=100, mean_power_w=1.0, energy_wh=1.0)
        errors = self._record(segments=[bad_segment]).validate()
        self.assertTrue(any("disagrees" in error for error in errors))

    def test_roundtrip_through_dict(self):
        clone = SyntheticCycleRecord.from_dict(self._record().to_dict())
        self.assertEqual(clone.validate(), [])
        self.assertEqual(clone.state_path, [1])

    def test_constant_generator_produces_valid_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "gen"
            records = ConstantGenerator(length_samples=120,
                                        power_w=800.0).generate_dataset(
                output, count=3, seed=17)
            self.assertEqual(len(records), 3)
            for index, record in enumerate(records):
                self.assertEqual(record.validate(), [])
                with np.load(output / "cycles" / f"synthetic_{index:04d}.npz"
                             ) as data:
                    power = data["appliance_w"]
                self.assertEqual(len(power), 120)
                self.assertGreaterEqual(float(power.min()), 0.0)
            summary = json.loads(
                (output / "generation_summary.json").read_text("utf-8"))
            self.assertEqual(summary["count"], 3)

    def test_negative_power_rejected(self):
        class Negative(ConstantGenerator):
            def __init__(self):
                super().__init__(length_samples=50, power_w=5.0)

            def generate_cycle(self, synthetic_cycle_id, seed, rng):
                power, record = super().generate_cycle(
                    synthetic_cycle_id, seed, rng)
                return power - 10.0, record

        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(RuntimeError):
                Negative().generate_dataset(Path(temporary) / "out",
                                            count=1, seed=1)


class ProvenanceTests(unittest.TestCase):
    def test_config_hash_order_insensitive(self):
        first = canonical_config_hash({"a": 1, "b": [1, 2]})
        second = canonical_config_hash({"b": [1, 2], "a": 1})
        self.assertEqual(first, second)

    def test_file_hash_matches_bytes_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "blob.bin"
            path.write_bytes(b"pslg")
            self.assertEqual(sha256_of_file(path), sha256_of_bytes(b"pslg"))

    def test_registry_append_and_replace(self):
        with tempfile.TemporaryDirectory() as temporary:
            registry = Path(temporary) / "registry.csv"
            append_registry_row(registry, {"run_id": "r1", "group": "B0",
                                           "seed": 17})
            append_registry_row(registry, {"run_id": "r2", "group": "B1"})
            append_registry_row(registry, {"run_id": "r1", "group": "B0",
                                           "seed": 42})
            frame = pd.read_csv(registry)
            self.assertEqual(len(frame), 2)
            row = frame[frame["run_id"] == "r1"].iloc[0]
            self.assertEqual(int(row["seed"]), 42)

    def test_environment_record_mentions_torch(self):
        self.assertIn("torch", environment_record())


class StatisticsTests(unittest.TestCase):
    def test_per_cycle_metrics_split(self):
        y_true = np.concatenate([np.full(50, 500.0), np.zeros(50)])
        y_pred = np.concatenate([np.full(50, 480.0), np.full(50, 5.0)])
        results = per_cycle_metrics(y_true, y_pred,
                                    cycle_spans=[(0, 50), (50, 100)])
        self.assertEqual(len(results), 2)
        self.assertAlmostEqual(results[0]["mae_w"], 20.0, places=5)

    def test_paired_bootstrap_detects_improvement(self):
        rng = np.random.default_rng(7)
        better = rng.normal(20.0, 1.0, size=200)     # arm A per-cycle MAE
        worse = better + rng.normal(2.0, 0.5, size=200)
        ci = paired_bootstrap_ci(better, worse, n_bootstraps=500, seed=17)
        self.assertLess(ci["statistic"], 0.0)        # A minus B is negative
        self.assertLess(ci["ci_upper"], 0.0)         # significantly better
        self.assertLess(ci["ci_lower"], ci["statistic"])
        self.assertGreater(ci["ci_upper"], ci["ci_lower"])

    def test_paired_bootstrap_no_difference(self):
        rng = np.random.default_rng(7)
        values = rng.normal(20.0, 1.0, size=200)
        ci = paired_bootstrap_ci(values, values, n_bootstraps=200, seed=17)
        self.assertAlmostEqual(ci["statistic"], 0.0)
        self.assertLessEqual(ci["ci_lower"], 0.0)
        self.assertGreaterEqual(ci["ci_upper"], 0.0)

    def test_multi_seed_summary(self):
        summary = multi_seed_summary([21.0, 22.0, 23.0])
        self.assertAlmostEqual(summary["mean"], 22.0)
        self.assertAlmostEqual(summary["std"], 1.0, places=6)
        self.assertEqual(summary["n_seeds"], 3)

    def test_audit_identical_indices(self):
        reference = np.arange(10)
        self.assertTrue(
            audit_identical_indices(reference, np.arange(10))["identical"])
        audit = audit_identical_indices(reference, np.arange(10)[::-1])
        self.assertFalse(audit["identical"])


if __name__ == "__main__":
    unittest.main()
