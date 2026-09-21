"""Tests for the Phase F composition suite (Markov/HSMM/boundary/B5)."""
from __future__ import annotations

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

from scripts.fit_hsmm_composer import build_sequences, main as fit_main  # noqa: E402
from src.composition.boundary_handler import BoundaryHandler  # noqa: E402
from src.composition.constrained_composer import (  # noqa: E402
    ConstrainedComposer,
)
from src.composition.duration_model import StateDurationModel  # noqa: E402
from src.composition.hsmm_sequence import HSMMPathSampler  # noqa: E402
from src.composition.transition_model import MarkovChain  # noqa: E402
from tests.test_cvae_family import _write_state_library  # noqa: E402


class MarkovChainTests(unittest.TestCase):
    def test_fit_counts_transitions(self) -> None:
        chain = MarkovChain.fit([[0, 1, 2], [0, 1, 2], [0, 2]],
                                smoothing_alpha=0.0)
        outgoing = chain.outgoing(0)
        self.assertAlmostEqual(outgoing[1], 2 / 3, places=6)
        self.assertAlmostEqual(outgoing[2], 1 / 3, places=6)
        self.assertEqual(chain.outgoing(2), {})  # no observed successor

    def test_smoothing_stays_within_observed_options(self) -> None:
        chain = MarkovChain.fit([[0, 1], [0, 2]], smoothing_alpha=1.0)
        outgoing = chain.outgoing(0)
        self.assertAlmostEqual(sum(outgoing.values()), 1.0, places=6)
        self.assertEqual(sorted(outgoing), [1, 2])

    def test_sample_path_never_invents_transitions(self) -> None:
        chain = MarkovChain.fit([[0, 1], [0, 1], [0, 1]])
        rng = np.random.default_rng(0)
        for _ in range(20):
            path = chain.sample_path(rng, n_states=5)
            self.assertTrue(set(path) <= {0, 1})
            self.assertTrue(path[0] in (0, 1))

    def test_log_probability_rejects_unobserved_move(self) -> None:
        chain = MarkovChain.fit([[0, 1]])
        self.assertTrue(np.isfinite(chain.log_probability([0, 1])))
        self.assertEqual(chain.log_probability([1, 0]), -np.inf)

    def test_json_roundtrip(self) -> None:
        chain = MarkovChain.fit([[0, 1, 2]])
        restored = MarkovChain.from_dict(chain.to_dict())
        self.assertEqual(restored.transition_counts,
                         chain.transition_counts)
        self.assertEqual(restored.initial_counts, chain.initial_counts)


class DurationModelTests(unittest.TestCase):
    def test_empirical_sampling_stays_observed(self) -> None:
        model = StateDurationModel.fit([(0, 300), (0, 900), (1, 120)])
        rng = np.random.default_rng(0)
        for _ in range(30):
            self.assertIn(model.sample_duration(0, rng), (300, 900))
            self.assertEqual(model.sample_duration(1, rng), 120)

    def test_unknown_state_rejected(self) -> None:
        model = StateDurationModel.fit([(0, 300)])
        with self.assertRaises(KeyError):
            model.sample_duration(9, np.random.default_rng(0))

    def test_json_roundtrip(self) -> None:
        model = StateDurationModel.fit([(0, 300), (0, 900)])
        restored = StateDurationModel.from_dict(model.to_dict())
        self.assertEqual(restored.durations_by_state,
                         model.durations_by_state)


class HSMMTests(unittest.TestCase):
    def _hsmm(self) -> HSMMPathSampler:
        sequences = [[0, 3, 1], [0, 2, 1], [0, 3, 1], [0, 1]]
        markov = MarkovChain.fit(sequences)
        durations = StateDurationModel.fit([
            (0, 600), (3, 600), (2, 1200), (1, 3000), (1, 3300)])
        return HSMMPathSampler(markov, durations,
                               {3: 4, 2: 1})

    def test_sequence_respects_order_and_duration_ranges(self) -> None:
        rng = np.random.default_rng(1)
        sampler = self._hsmm()
        for _ in range(30):
            sequence = sampler.sample_sequence(rng, n_states=3)
            self.assertEqual(len(sequence), 3)
            for state, duration in sequence:
                self.assertIn(state, (0, 1, 2, 3))
                self.assertIn(duration,
                              sampler.durations.durations_by_state[state])

    def test_path_probability_recorded_and_finite(self) -> None:
        rng = np.random.default_rng(2)
        sampler = self._hsmm()
        sequence = sampler.sample_sequence(rng, n_states=3)
        self.assertTrue(np.isfinite(sampler.log_probability(sequence)))


class BoundaryHandlerTests(unittest.TestCase):
    def test_endpoint_match_selects_closest_start(self) -> None:
        handler = BoundaryHandler("endpoint_match")
        candidates = [(0, np.array([1800.0, 5.0])),
                      (1, np.array([305.0, 5.0])),
                      (2, np.array([900.0, 5.0]))]
        rng = np.random.default_rng(0)
        for _ in range(10):
            self.assertEqual(
                handler.select(candidates, previous_end_w=300.0, rng=rng),
                1)

    def test_none_mode_uses_rng(self) -> None:
        handler = BoundaryHandler("none")
        candidates = [(0, np.array([1.0])), (1, np.array([2.0]))]
        rng = np.random.default_rng(0)
        picked = {handler.select(candidates, 300.0, rng) for _ in range(20)}
        self.assertEqual(picked, {0, 1})

    def test_invalid_mode_rejected(self) -> None:
        with self.assertRaises(ValueError):
            BoundaryHandler("cross_fade")


class ConstrainedComposerTests(unittest.TestCase):
    def _composer(self, boundary_mode: str = "none",
                  use_durations: bool = True) -> ConstrainedComposer:
        sequences = [[0, 1, 2], [0, 1, 2], [0, 2, 1], [0, 1]]
        markov = MarkovChain.fit(sequences)
        durations = StateDurationModel.fit([
            (0, 180), (1, 300), (2, 240)])
        hsmm = HSMMPathSampler(markov, durations, {3: 4, 2: 1})
        rng = np.random.default_rng(8)
        pool_waves, pool_labels = [], []
        for state, base in ((0, 150.0), (1, 400.0), (2, 1800.0)):
            for _ in range(4):
                pool_waves.append(
                    rng.normal(base, 10, 60).clip(min=0))
                pool_labels.append(state)
        return ConstrainedComposer(hsmm, pool_waves, pool_labels,
                                   sample_seconds=6,
                                   boundary_mode=boundary_mode,
                                   use_hsmm_durations=use_durations)

    def test_records_are_schema_valid_and_deterministic(self) -> None:
        composer = self._composer()
        outputs = []
        for run_seed in (31, 31):
            records = composer.generate_dataset(
                Path(tempfile.mkdtemp()) / f"out_{run_seed}", count=3,
                seed=run_seed)
            dicts = [r.to_dict() for r in records]
            for entry, record in zip(dicts, records):
                self.assertEqual(record.route, "B5")
                self.assertTrue(record.waveform_sha256)
                self.assertEqual(record.validate(), [])
                for segment in record.segments:
                    # HSMM durations are enforced by resampling.
                    self.assertEqual(segment.actual_samples,
                                     segment.target_samples)
                entry.pop("created_utc")
            outputs.append(dicts)
        self.assertEqual(outputs[0], outputs[1])

    def test_ablation_rungs_change_behaviour(self) -> None:
        with_dur = self._composer(use_durations=True)
        without_dur = self._composer(use_durations=False)
        rng = np.random.default_rng(5)
        _, record_with = with_dur.generate_cycle("x", 1, rng)
        _, record_without = without_dur.generate_cycle("x", 1, rng)
        self.assertTrue(all(s.actual_samples == s.target_samples
                            for s in record_with.segments))
        self.assertTrue(any(s.actual_samples != s.target_samples
                            for s in record_without.segments))

    def test_endpoint_match_reduces_boundary_cost(self) -> None:
        costs = {}
        for mode in ("none", "endpoint_match"):
            composer = self._composer(boundary_mode=mode)
            rng = np.random.default_rng(12)
            costs[mode] = np.mean([
                composer.generate_cycle(f"c{i}", 2, rng)[1]
                .conditions["mean_boundary_cost_w"]
                for i in range(12)])
        self.assertLessEqual(costs["endpoint_match"], costs["none"])


class FitCliTests(unittest.TestCase):
    def test_fit_cli_writes_frozen_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            library = _write_state_library(Path(tmp))
            output = Path(tmp) / "hsmm.json"
            argv = ["fit_hsmm_composer.py",
                    "--state-library-dir", str(library),
                    "--output", str(output)]
            with patch.object(sys, "argv", argv):
                fit_main()
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["protocol"], "hsmm_model_v1")
            self.assertTrue(payload["markov"]["transition_counts"])
            sampler = HSMMPathSampler(
                MarkovChain.from_dict(payload["markov"]),
                StateDurationModel.from_dict(payload["durations"]),
                {int(k): v for k, v in payload["path_lengths"].items()})
            rng = np.random.default_rng(0)
            sequence = sampler.sample_sequence(rng, n_states=3)
            self.assertTrue(sequence)


if __name__ == "__main__":
    unittest.main()
