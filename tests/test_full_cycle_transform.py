"""Tests for the B3-T real-cycle transform generator and its CLI."""
from __future__ import annotations

import argparse
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

from scripts.generate_full_cycles import build_generator, main  # noqa: E402
from src.generation.full_cycle_transform import (  # noqa: E402
    RealCycleTransformGenerator,
)
from tests.test_synthetic_quality import _write_real_library  # noqa: E402


class TransformGeneratorTests(unittest.TestCase):
    def _generator(self, library: Path) -> RealCycleTransformGenerator:
        return RealCycleTransformGenerator(library, sample_seconds=6)

    def test_outputs_respect_scales_and_guards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            library = _write_real_library(Path(tmp))
            generator = self._generator(library)
            output = Path(tmp) / "out"
            records = generator.generate_dataset(output, count=20, seed=17)
            self.assertEqual(len(records), 20)
            for record in records:
                scales = record.conditions
                self.assertIn("donor_cycle_id", scales)
                self.assertTrue(
                    0.85 <= scales["time_scale"] <= 1.2)
                self.assertTrue(
                    0.9 <= scales["power_scale"] <= 1.1)
                self.assertGreater(record.waveform_sha256, "")
                donor_index = int(scales["donor_cycle_id"].split("_")[1])
                donor_samples = len(generator.donors[donor_index])
                segment = record.segments[0]
                self.assertEqual(segment.actual_samples,
                                 int(round(donor_samples
                                           * scales["time_scale"])))
                duration = segment.actual_duration_seconds
                self.assertGreaterEqual(
                    duration, generator.duration_bounds[0])
                self.assertLessEqual(duration, generator.duration_bounds[1])

    def test_no_negative_power_and_determinism(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            library = _write_real_library(Path(tmp))
            generator = self._generator(library)
            rng_a = np.random.default_rng(5)
            rng_b = np.random.default_rng(5)
            for index in range(8):
                power_a, record_a = generator.generate_cycle(
                    f"s{index}", 5, rng_a)
                power_b, record_b = generator.generate_cycle(
                    f"s{index}", 5, rng_b)
                self.assertFalse((power_a < 0).any())
                np.testing.assert_array_equal(power_a, power_b)
                self.assertEqual(record_a.conditions,
                                 record_b.conditions)

    def test_wide_scales_still_inside_envelope_via_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            library = _write_real_library(Path(tmp))
            generator = RealCycleTransformGenerator(
                library, sample_seconds=6,
                time_scale_range=(0.5, 2.0), power_scale_range=(0.5, 2.0))
            rng = np.random.default_rng(9)
            for index in range(10):
                power, record = generator.generate_cycle(f"w{index}", 9, rng)
                duration = len(power) * 6
                self.assertGreaterEqual(duration,
                                        generator.duration_bounds[0])
                self.assertLessEqual(duration, generator.duration_bounds[1])
                self.assertLessEqual(float(power.max()),
                                     generator.peak_limit)

    def test_impossible_envelope_fails_after_bounded_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            library = _write_real_library(Path(tmp), count=4, seed=1)
            generator = self._generator(library)
            # Make the envelope impossible: narrower than one sample width.
            generator.duration_bounds = (1800.0, 1800.0 + 1e-6)
            rng = np.random.default_rng(2)
            with self.assertRaises(RuntimeError):
                generator.generate_cycle("doomed", 2, rng)

    def test_unimplemented_route_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            library = _write_real_library(Path(tmp))
            argv = ["generate_full_cycles.py", "--route", "cvae",
                    "--real-library-dir", str(library),
                    "--output-dir", str(Path(tmp) / "out"),
                    "--count", "1", "--seed", "1"]
            with patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit):
                    build_generator(
                        argparse.Namespace(
                            route="cvae", real_library_dir=str(library),
                            sample_seconds=6, time_scale_min=0.85,
                            time_scale_max=1.2, power_scale_min=0.9,
                            power_scale_max=1.1, checkpoint_dir=None,
                            device="cpu"))

    def test_cli_end_to_end_and_quality_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            library = _write_real_library(root)
            output = root / "transform_out"
            argv = ["generate_full_cycles.py", "--route", "transform",
                    "--real-library-dir", str(library),
                    "--output-dir", str(output),
                    "--count", "10", "--seed", "17"]
            with patch.object(sys, "argv", argv):
                main()
            summary = json.loads((output / "generation_summary.json")
                                 .read_text(encoding="utf-8"))
            self.assertEqual(summary["generator"], "b3_transform")
            self.assertEqual(summary["count"], 10)
            # The batch-0 quality gate must accept transform output.
            from src.validation.synthetic_quality import (
                evaluate_synthetic_dataset)
            report = evaluate_synthetic_dataset(output, library)
            self.assertTrue(report["passed"], report["flags"])


if __name__ == "__main__":
    unittest.main()
