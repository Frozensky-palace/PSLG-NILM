"""Tests for the batch-0 quality gate and memorization audit."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_synthetic_quality import main as quality_main  # noqa: E402
from src.generation.base_generator import ConstantGenerator  # noqa: E402
from src.validation.memorization import (  # noqa: E402
    audit_memorization,
    real_baseline,
    unit_vector,
)
from src.validation.synthetic_quality import (  # noqa: E402
    diversity_index,
    evaluate_synthetic_dataset,
    physical_checks,
    resample_to_length,
)


def _write_real_library(root: Path, count: int = 30, seed: int = 5,
                        mean_w: float = 500.0) -> Path:
    """Build a minimal real-cycle library fixture."""
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


class SyntheticQualityTests(unittest.TestCase):
    def test_resample_to_length_exact(self) -> None:
        out = resample_to_length(np.ones(300) * 7.0, 128)
        self.assertEqual(len(out), 128)
        np.testing.assert_allclose(out, 7.0)

    def test_physical_checks_rejects_negative(self) -> None:
        power = np.array([100.0, -5.0, 100.0])
        checks = physical_checks(power, 6, np.array([1800.0]),
                                 np.array([250.0]), np.array([800.0]))
        self.assertTrue(checks["negative_power"])

    def test_physical_checks_flags_impossible_peak(self) -> None:
        power = np.full(300, 2000.0)
        checks = physical_checks(power, 6, np.array([1800.0]),
                                 np.array([250.0]), np.array([800.0]))
        self.assertTrue(checks["peak_exceeds_real_max_x1p05"])

    def test_diversity_separates_identical_from_distinct(self) -> None:
        rng = np.random.default_rng(0)
        identical = [np.full(64, 5.0) for _ in range(4)]
        self.assertGreater(diversity_index(identical)["identical_pairs"], 0)
        distinct = [rng.normal(500, 50, 64) for _ in range(4)]
        self.assertEqual(diversity_index(distinct)["identical_pairs"], 0)

    def test_end_to_end_constant_generator_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            library = _write_real_library(root)
            synthetic_dir = root / "synthetic"
            ConstantGenerator(length_samples=300, power_w=500.0
                              ).generate_dataset(synthetic_dir, count=5,
                                                 seed=1)
            report = evaluate_synthetic_dataset(synthetic_dir, library)
            self.assertTrue(report["passed"], report["flags"])
            self.assertEqual(report["flags"]["negative_power"], "PASS")
            self.assertEqual(report["n_synthetic"], 5)

    def test_end_to_end_impossible_peak_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            library = _write_real_library(root)
            synthetic_dir = root / "synthetic"
            ConstantGenerator(length_samples=300, power_w=5000.0
                              ).generate_dataset(synthetic_dir, count=3,
                                                 seed=1)
            report = evaluate_synthetic_dataset(synthetic_dir, library)
            self.assertFalse(report["passed"])
            self.assertEqual(report["flags"]["impossible_peak"], "FAIL")

    def test_cli_writes_report_and_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            library = _write_real_library(root)
            synthetic_dir = root / "synthetic"
            ConstantGenerator(length_samples=300, power_w=500.0
                              ).generate_dataset(synthetic_dir, count=4,
                                                 seed=2)
            output = root / "report.json"
            argv = ["evaluate_synthetic_quality.py",
                    "--synthetic-dir", str(synthetic_dir),
                    "--real-library-dir", str(library),
                    "--output", str(output)]
            with patch.object(sys, "argv", argv):
                quality_main()
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["passed"])
            # A clearly broken dataset must make the CLI fail loudly.
            broken = root / "broken"
            ConstantGenerator(length_samples=300, power_w=5000.0
                              ).generate_dataset(broken, count=2, seed=3)
            argv[-3:] = [str(broken)]
            with patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit):
                    quality_main()


class MemorizationTests(unittest.TestCase):
    def _real_powers(self, count: int = 60, seed: int = 7
                     ) -> list[np.ndarray]:
        rng = np.random.default_rng(seed)
        return [np.sin(np.linspace(0, 6, 256)) * 500 + 500
                + rng.normal(0, 5, 256) for _ in range(count)]

    def test_unit_vector_is_normalized(self) -> None:
        vector = unit_vector(np.full(256, 500.0))
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=6)

    def test_exact_copy_is_detected(self) -> None:
        real = self._real_powers()
        synth = [real[3].copy()]
        report = audit_memorization(synth, real, baseline_count=40)
        self.assertGreaterEqual(report["exact_duplicate_count"], 1)
        self.assertFalse(report["passed"])

    def test_unrelated_noise_is_not_replicated(self) -> None:
        rng = np.random.default_rng(11)
        real = self._real_powers()
        synth = [rng.normal(3000, 900, 256) for _ in range(5)]
        report = audit_memorization(synth, real, baseline_count=40)
        self.assertEqual(report["replicated_count"], 0)
        self.assertEqual(report["exact_duplicate_count"], 0)
        self.assertTrue(report["passed"])

    def test_baseline_excludes_self_so_no_zero_distances(self) -> None:
        vectors = np.stack([unit_vector(p) for p in self._real_powers(20)])
        baseline = real_baseline(vectors, 20, seed=0)
        self.assertTrue((baseline > 0).all())

    def test_empty_inputs_rejected(self) -> None:
        with self.assertRaises(ValueError):
            audit_memorization([], [np.ones(256)])


if __name__ == "__main__":
    unittest.main()
