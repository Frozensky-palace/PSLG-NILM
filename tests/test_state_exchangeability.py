"""Tests for train-only state exchangeability diagnostics (Phase A2)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.validation.state_exchangeability import (  # noqa: E402
    EXAMPLE_CATEGORIES,
    exchangeability_verdict,
    load_state_library,
    multimodality_profile,
    replacement_boundary_jumps,
    select_example_blocks,
    state_metrics,
    transition_context_stats,
)


def _make_library(directory: Path, partition: str = "train") -> Path:
    """Two states across three cycles: state 0 low power, state 1 high power."""
    rng = np.random.default_rng(7)
    records, power_parts, offsets, timestamp_parts = [], [], [0], []
    block_id = 0
    for cycle_index in range(3):
        cycle_id = f"cycle_{cycle_index}"
        for state_label, base_power in ((0, 200.0), (1, 1800.0)):
            length = 40 + int(rng.integers(0, 20))
            waveform = base_power + rng.normal(0, 8.0, size=length)
            waveform = np.clip(waveform, 0.0, None).astype(np.float32)
            start_sample = sum(len(p) for p in power_parts) if power_parts else 0
            records.append({
                "state_block_id": block_id,
                "state_label": state_label,
                "source_partition": partition,
                "cycle_id": cycle_id,
                "start_sample": start_sample,
                "end_sample_exclusive": start_sample + length,
                "duration_seconds": length * 6,
                "samples": length,
                "mean_power_w": float(waveform.mean()),
                "std_power_w": float(waveform.std()),
                "energy_wh": float(waveform.sum() * 6 / 3600.0),
                "start_power_w": float(waveform[0]),
                "end_power_w": float(waveform[-1]),
                "mean_abs_slope_w_per_sample": float(
                    np.abs(np.diff(waveform)).mean()),
                "previous_state_label": None if state_label == 0 else 0,
                "next_state_label": 1 if state_label == 0 else None,
            })
            power_parts.append(waveform)
            offsets.append(offsets[-1] + length)
            timestamp_parts.append(
                np.arange(length, dtype=np.int64) * 6 + start_sample * 6)
            block_id += 1
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(directory / "state_inventory.csv", index=False)
    np.savez_compressed(
        directory / "state_waveforms.npz",
        power_w=np.concatenate(power_parts),
        timestamp=np.concatenate(timestamp_parts),
        offsets=np.asarray(offsets, dtype=np.int64),
    )
    return directory


class StateExchangeabilityTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.library = load_state_library(
            "test", _make_library(self.root / "lib"))
        self.library.sample_seconds = 6

    def tearDown(self):
        self._temporary.cleanup()

    def test_rejects_non_train_partition(self):
        non_train = _make_library(self.root / "bad", partition="validation")
        with self.assertRaises(ValueError):
            load_state_library("bad", non_train)

    def test_state_metrics_per_state(self):
        metrics = state_metrics(self.library, rng_seed=17)
        self.assertEqual(len(metrics), 2)
        row = metrics[metrics["state_label"] == 1].iloc[0]
        self.assertGreater(row["median_power_w"], 1000.0)
        for column in ("within_over_between", "nn_same_cycle_fraction",
                       "replaced_over_original", "multimodal_entropy"):
            self.assertTrue(np.isfinite(metrics[column]).all())

    def test_replacement_jumps_deterministic(self):
        first = replacement_boundary_jumps(
            self.library.inventory, self.library, 0,
            np.random.default_rng(17))
        second = replacement_boundary_jumps(
            self.library.inventory, self.library, 0,
            np.random.default_rng(17))
        self.assertEqual(first, second)
        self.assertGreater(first["blocks_with_neighbors"], 0)

    def test_example_selection_covers_categories(self):
        selections = select_example_blocks(self.library)
        self.assertEqual(set(selections), {0, 1})
        for state_label, categories in selections.items():
            self.assertEqual(set(categories), set(EXAMPLE_CATEGORIES))
            for row_index in categories.values():
                self.assertEqual(
                    int(self.library.inventory.iloc[row_index]["state_label"]),
                    state_label)

    def test_multimodality_entropy_bounded(self):
        result = multimodality_profile(self.library, 0)
        self.assertGreaterEqual(result["profile_entropy"], 0.0)
        self.assertLessEqual(result["profile_entropy"], 1.0)

    def test_transition_context_conditioned(self):
        context = transition_context_stats(self.library.inventory)
        state1 = context[context["state_label"] == 1]
        self.assertEqual(int(state1["previous_state_label"].iloc[0]), 0)

    def test_verdict_flags_long_multimodal_state(self):
        compact = pd.Series({
            "median_duration_s": 600.0, "multimodal_entropy": 0.5,
            "nn_same_cycle_fraction": 0.1, "replaced_over_original": 1.1})
        self.assertIn("可直接交换", exchangeability_verdict(compact))
        flagged = pd.Series({
            "median_duration_s": 4500.0, "multimodal_entropy": 0.95,
            "nn_same_cycle_fraction": 0.5, "replaced_over_original": 2.0})
        verdict = exchangeability_verdict(flagged)
        self.assertIn("需要条件化或继续细分", verdict)


if __name__ == "__main__":
    unittest.main()
