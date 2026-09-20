"""Tests for B2 policy and boundary ablation helpers (Phase A3/A4)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pilot_composition import (  # noqa: E402
    apply_linear_crossfade,
    boundary_jumps,
    choose_matched_donor,
    correct_block_endpoints,
    filter_candidates_by_length_ratio,
)


class LengthRatioFilterTests(unittest.TestCase):
    def test_keeps_candidates_inside_limits(self):
        indices = np.asarray([10, 11, 12, 13])
        samples = np.asarray([50.0, 100.0, 200.0, 400.0])
        kept, ratios = filter_candidates_by_length_ratio(
            indices, samples, target_samples=100,
            ratio_min=0.5, ratio_max=2.0)
        self.assertEqual(sorted(kept.tolist()), [10, 11, 12])
        self.assertTrue(np.all(ratios >= 0.5) and np.all(ratios <= 2.0))

    def test_empty_result_allowed(self):
        kept, _ = filter_candidates_by_length_ratio(
            np.asarray([1]), np.asarray([1000.0]), 10, 0.5, 2.0)
        self.assertEqual(len(kept), 0)

    def test_rejects_invalid_limits(self):
        with self.assertRaises(ValueError):
            filter_candidates_by_length_ratio(
                np.asarray([1]), np.asarray([10.0]), 10, 2.0, 0.5)

    def test_rejects_zero_samples(self):
        with self.assertRaises(ValueError):
            filter_candidates_by_length_ratio(
                np.asarray([1]), np.asarray([0.0]), 10, 0.5, 2.0)


class EndpointCorrectionTests(unittest.TestCase):
    def test_endpoints_match_template(self):
        donor = np.full(100, 150.0)
        corrected, delta = correct_block_endpoints(
            donor, template_start_power_w=200.0, template_end_power_w=90.0)
        self.assertAlmostEqual(float(corrected[0]), 200.0, places=3)
        self.assertAlmostEqual(float(corrected[-1]), 90.0, places=3)
        self.assertEqual(len(corrected), 100)
        self.assertLess(delta, 0.0)  # ramp shifts most mass down

    def test_zero_correction_when_already_matched(self):
        donor = np.linspace(100.0, 200.0, 50)
        corrected, delta = correct_block_endpoints(donor, 100.0, 200.0)
        self.assertAlmostEqual(float(delta), 0.0, places=5)
        np.testing.assert_allclose(corrected, donor, atol=1e-4)

    def test_clips_at_zero_and_max(self):
        donor = np.zeros(10)
        corrected, _ = correct_block_endpoints(
            donor, template_start_power_w=-50.0, template_end_power_w=99_999.0)
        self.assertTrue((corrected >= 0).all())
        self.assertLessEqual(float(corrected.max()), 4096.0)

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            correct_block_endpoints(np.array([]), 0.0, 0.0)


class LinearCrossfadeTests(unittest.TestCase):
    def test_preserves_length_and_reduces_jump(self):
        waveform = np.concatenate([
            np.full(50, 200.0), np.full(50, 900.0), np.full(50, 100.0)])
        boundaries = [50, 100]
        before = boundary_jumps(waveform, boundaries)
        blended, delta = apply_linear_crossfade(
            waveform, boundaries, window_samples=10)
        self.assertEqual(len(blended), len(waveform))
        after = boundary_jumps(blended.astype(np.float64), boundaries)
        self.assertLess(after.max(), before.max())
        self.assertAlmostEqual(float(delta), float(
            blended.astype(np.float64).sum() - waveform.astype(np.float64).sum()),
            places=2)
        self.assertTrue((blended >= 0).all())

    def test_unaffected_away_from_boundaries(self):
        waveform = np.concatenate([
            np.full(50, 200.0), np.full(50, 900.0)])
        blended, _ = apply_linear_crossfade(waveform, [50], 8)
        np.testing.assert_array_equal(blended[:40], waveform[:40])
        np.testing.assert_array_equal(blended[60:], waveform[60:])

    def test_rejects_tiny_window_and_bad_boundary(self):
        with self.assertRaises(ValueError):
            apply_linear_crossfade(np.ones(10), [5], window_samples=1)
        with self.assertRaises(ValueError):
            apply_linear_crossfade(np.ones(10), [0], window_samples=4)
        with self.assertRaises(ValueError):
            apply_linear_crossfade(np.full(10, -1.0), [5], 4)


class FeatureWeightTests(unittest.TestCase):
    def test_duration_only_ignores_power(self):
        rng = np.random.default_rng(3)
        # Candidate A: nearly identical duration, wildly different power.
        # Candidate B: same duration cost class, but perfect power match.
        indices = np.asarray([0, 1])
        cycle_ids = np.asarray(["other_a", "other_b"])
        scores_call = lambda mean_weight, endpoint_weight: choose_matched_donor(
            indices, cycle_ids, template_cycle_id="template",
            target_samples=100, target_mean_power_w=1000.0,
            target_start_power_w=1000.0, target_end_power_w=1000.0,
            candidate_samples=np.asarray([101.0, 140.0]),
            candidate_mean_power_w=np.asarray([50.0, 1000.0]),
            candidate_start_power_w=np.asarray([50.0, 1000.0]),
            candidate_end_power_w=np.asarray([50.0, 1000.0]),
            rng=rng, top_k=1,
            mean_weight=mean_weight, endpoint_weight=endpoint_weight)
        duration_only, _ = scores_call(0.0, 0.0)
        full, _ = scores_call(0.25, 0.25)
        self.assertEqual(duration_only, 0)   # closest duration wins
        self.assertEqual(full, 1)            # power match dominates

    def test_weights_reject_negatives(self):
        with self.assertRaises(ValueError):
            choose_matched_donor(
                np.asarray([0]), np.asarray(["other"]),
                template_cycle_id="t", target_samples=10,
                target_mean_power_w=1.0, target_start_power_w=1.0,
                target_end_power_w=1.0,
                candidate_samples=np.asarray([10.0]),
                candidate_mean_power_w=np.asarray([1.0]),
                candidate_start_power_w=np.asarray([1.0]),
                candidate_end_power_w=np.asarray([1.0]),
                rng=np.random.default_rng(0), mean_weight=-1.0)


if __name__ == "__main__":
    unittest.main()
