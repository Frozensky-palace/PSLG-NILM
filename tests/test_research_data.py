"""Tests for cycle inventory and leakage-free research split."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from tempfile import TemporaryDirectory

from src.data.cycle_inventory import CycleDetector
from src.data.cycle_audit import select_audit_cycles
from src.data.aligned_partitions import (
    aligned_power_frame,
    cycle_grid_counts,
    partition_intervals,
)
from src.data.research_split import (
    apply_protocol_filters,
    chronological_cycle_split,
    leakage_report,
)
from src.steps.time_segmentation import TimeSegmentationStep
from models.feature_extract.physical_stats import physical_stats
from src.data.pilot_composition import (
    boundary_jumps,
    choose_donor_indices,
    choose_matched_donor,
    matched_donor_scores,
    resample_waveform,
)
from src.data.event_placement import idle_runs, schedule_lengths
from src.nilm.window_dataset import valid_center_ranges
from src.nilm.metrics import nilm_metrics


class CycleInventoryTest(unittest.TestCase):
    def test_streaming_chunks_preserve_cycle_boundaries(self):
        detector = CycleDetector(
            threshold_w=20, max_inactive_seconds=30,
            min_duration_seconds=20, sample_seconds=10,
            dataset="test", building=1, appliance="wm",
            valid_start=0, valid_end=200,
        )
        detector.feed(
            np.array([0, 10, 20, 30, 40], dtype=float),
            np.array([0, 30, 0, 40, 0], dtype=float),
            row_offset=0,
        )
        detector.feed(
            np.array([50, 100, 110, 120], dtype=float),
            np.array([35, 0, 50, 55], dtype=float),
            row_offset=5,
        )
        records = detector.finish()
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["start_unix"], 10)
        self.assertEqual(records[0]["end_unix"], 50)
        self.assertTrue(records[0]["eligible"])
        self.assertFalse(records[1]["eligible"])
        self.assertEqual(records[1]["exclusion_reason"], "shorter_than_min_duration")

    def test_mains_coverage_marks_cycle_ineligible(self):
        detector = CycleDetector(
            threshold_w=20, max_inactive_seconds=30,
            min_duration_seconds=10, sample_seconds=10,
            dataset="test", building=1, appliance="wm",
            valid_start=15, valid_end=100,
        )
        detector.feed(np.array([0, 10, 20]), np.array([30, 40, 50]))
        record = detector.finish()[0]
        self.assertFalse(record["eligible"])
        self.assertIn("outside_mains_coverage", record["exclusion_reason"])


class ResearchSplitTest(unittest.TestCase):
    def make_inventory(self, n=10):
        return pd.DataFrame({
            "cycle_id": [f"c{i}" for i in range(n)],
            "start_unix": [i * 100 for i in range(n)],
            "end_unix": [i * 100 + 50 for i in range(n)],
            "eligible": [True] * n,
        })

    def test_split_keeps_whole_cycles_and_is_chronological(self):
        out = chronological_cycle_split(self.make_inventory())
        self.assertEqual((out["partition"] == "train").sum(), 6)
        self.assertEqual((out["partition"] == "validation").sum(), 2)
        self.assertEqual((out["partition"] == "test").sum(), 2)
        report = leakage_report(out)
        self.assertTrue(report["passed"])
        self.assertTrue(report["chronological_order_ok"])

    def test_ineligible_cycle_is_excluded(self):
        frame = self.make_inventory()
        frame.loc[4, "eligible"] = False
        out = chronological_cycle_split(frame)
        self.assertEqual(out.loc[out["cycle_id"] == "c4", "partition"].iloc[0],
                         "excluded")
        self.assertTrue(leakage_report(out)["passed"])

    def test_cross_partition_overlap_is_reported(self):
        out = chronological_cycle_split(self.make_inventory())
        first_val = out.index[out["partition"] == "validation"][0]
        last_train = out.index[out["partition"] == "train"][-1]
        out.loc[first_val, "start_unix"] = out.loc[last_train, "end_unix"]
        report = leakage_report(out)
        self.assertFalse(report["passed"])
        self.assertEqual(report["problems"][0]["type"],
                         "cross_partition_time_overlap")

    def test_device_instance_filter_preserves_audit_rows(self):
        frame = self.make_inventory(8)
        frame["device_instance"] = [1, 1, 1, 1, 2, 2, 2, 2]
        selected = apply_protocol_filters(frame, device_instance=1)
        self.assertTrue(selected.loc[:3, "eligible"].all())
        self.assertFalse(selected.loc[4:, "eligible"].any())
        self.assertTrue(selected["inventory_eligible"].all())
        self.assertTrue(selected.loc[4:, "exclusion_reason"]
                        .str.contains("different_device_instance").all())


class CycleAuditTest(unittest.TestCase):
    def test_selection_covers_each_partition_without_duplicates(self):
        rows = []
        for partition in ("train", "validation", "test"):
            for idx in range(10):
                rows.append({
                    "cycle_id": f"{partition}_{idx}",
                    "partition": partition,
                    "start_unix": idx * 100,
                    "duration_seconds": 100 + idx,
                    "active_energy_wh_approx": 10 + idx * 2,
                    "source_row_start": idx * 20,
                    "source_row_end": idx * 20 + 10,
                })
        selected = select_audit_cycles(pd.DataFrame(rows), per_partition=6, seed=7)
        self.assertEqual(len(selected), 18)
        self.assertEqual(selected["cycle_id"].nunique(), 18)
        self.assertEqual(selected["partition"].value_counts().to_dict(), {
            "train": 6, "validation": 6, "test": 6,
        })
        reasons = set("+".join(selected["audit_reason"]).split("+"))
        self.assertTrue({"shortest", "median_duration", "longest",
                         "lowest_energy", "highest_energy"}.issubset(reasons))


class AlignedPartitionTest(unittest.TestCase):
    def test_partition_intervals_do_not_overlap(self):
        frame = pd.DataFrame({
            "partition": ["train", "train", "validation", "test"],
            "start_unix": [100, 200, 400, 700],
            "end_unix": [150, 250, 500, 800],
        })
        intervals = partition_intervals(frame)
        self.assertEqual(intervals["train"], (100, 400))
        self.assertEqual(intervals["validation"], (400, 700))
        self.assertEqual(intervals["test"], (700, 801))

    def test_alignment_keeps_identity_and_marks_long_gap(self):
        index = pd.to_datetime([0, 6, 12, 30, 36], unit="s", utc=True)
        branch = pd.Series([10, 20, 30, 40, 50], index=index)
        mains = pd.Series([110, 120, 130, 140, 150], index=index)
        frame, quality = aligned_power_frame(
            branch, mains, start_unix=0, end_unix=42,
            sample_seconds=6, max_interpolate_seconds=6)
        self.assertEqual(frame["timestamp"].tolist(), [0, 6, 12, 30, 36])
        self.assertEqual(frame["segment_id"].tolist(), [0, 0, 0, 1, 1])
        np.testing.assert_allclose(
            frame["mains_w"],
            frame["appliance_w"] + frame["background_signed_w"])
        self.assertEqual(quality["missing_rows"], 2)

    def test_cycle_counts_use_inclusive_cycle_end(self):
        cycles = pd.DataFrame({
            "cycle_id": ["a", "b"],
            "start_unix": [6, 24],
            "end_unix": [18, 30],
        })
        counts = cycle_grid_counts(cycles, np.array([0, 6, 12, 18, 30]))
        self.assertEqual(counts, {"a": 3, "b": 1})

    def test_padding_can_fill_a_gap_at_target_boundary(self):
        index = pd.to_datetime([-6, 6, 12], unit="s", utc=True)
        branch = pd.Series([10, 30, 40], index=index)
        mains = pd.Series([110, 130, 140], index=index)
        frame, quality = aligned_power_frame(
            branch, mains, start_unix=0, end_unix=18,
            sample_seconds=6, max_interpolate_seconds=6)
        self.assertEqual(frame["timestamp"].tolist(), [0, 6, 12])
        self.assertEqual(quality["missing_rows"], 0)


class StateDiscoveryInputTest(unittest.TestCase):
    def test_explicit_cycle_directory_has_priority(self):
        with TemporaryDirectory() as explicit, TemporaryDirectory() as fallback:
            step = TimeSegmentationStep(input_dir=explicit)
            context = {"input_root": fallback, "log_root": fallback}
            self.assertEqual(step._resolve_input_dir(context), explicit)

    def test_physical_features_ignore_padding(self):
        data = np.zeros((2, 5, 4), dtype=float)
        data[0, :3, 0] = [10, 20, 30]
        data[0, 3:, 0] = 9999
        data[1, :3, 0] = [10, 20, 30]
        features, history = physical_stats(data, {"lengths": [[3], [3]]})
        np.testing.assert_allclose(features[0], features[1])
        self.assertEqual(history["epochs_trained"], 0)


class PilotCompositionTest(unittest.TestCase):
    def test_resample_has_exact_target_length_and_endpoints(self):
        result = resample_waveform(np.array([10, 20, 40]), 7)
        self.assertEqual(len(result), 7)
        self.assertEqual(float(result[0]), 10.0)
        self.assertEqual(float(result[-1]), 40.0)

    def test_donor_prefers_another_cycle(self):
        rng = np.random.default_rng(17)
        chosen = choose_donor_indices(
            np.array([4, 5, 6]), np.array(["a", "a", "b"]), "a", rng)
        self.assertEqual(chosen, 6)

    def test_matched_donor_chooses_best_cross_cycle_candidate(self):
        donor_id, score = choose_matched_donor(
            np.array([10, 11, 12]),
            np.array(["template", "other_a", "other_b"]),
            template_cycle_id="template",
            target_samples=100,
            target_mean_power_w=200,
            target_start_power_w=20,
            target_end_power_w=30,
            candidate_samples=np.array([100, 102, 300]),
            candidate_mean_power_w=np.array([200, 205, 900]),
            candidate_start_power_w=np.array([20, 22, 800]),
            candidate_end_power_w=np.array([30, 28, 700]),
            rng=np.random.default_rng(17),
            top_k=1,
        )
        self.assertEqual(donor_id, 11)
        self.assertLess(score, 0.1)

    def test_matched_donor_is_deterministic_for_same_seed(self):
        kwargs = dict(
            template_cycle_id="template",
            target_samples=100,
            target_mean_power_w=200,
            target_start_power_w=20,
            target_end_power_w=30,
            candidate_samples=np.array([95, 100, 105]),
            candidate_mean_power_w=np.array([190, 200, 210]),
            candidate_start_power_w=np.array([18, 20, 22]),
            candidate_end_power_w=np.array([28, 30, 32]),
            top_k=3,
        )
        first = choose_matched_donor(
            np.array([1, 2, 3]), np.array(["a", "b", "c"]),
            rng=np.random.default_rng(73), **kwargs)
        second = choose_matched_donor(
            np.array([1, 2, 3]), np.array(["a", "b", "c"]),
            rng=np.random.default_rng(73), **kwargs)
        self.assertEqual(first, second)

    def test_matched_donor_rejects_same_cycle_only_pool(self):
        with self.assertRaisesRegex(ValueError, "another cycle"):
            choose_matched_donor(
                np.array([1]), np.array(["template"]),
                template_cycle_id="template",
                target_samples=100,
                target_mean_power_w=200,
                target_start_power_w=20,
                target_end_power_w=30,
                candidate_samples=np.array([100]),
                candidate_mean_power_w=np.array([200]),
                candidate_start_power_w=np.array([20]),
                candidate_end_power_w=np.array([30]),
                rng=np.random.default_rng(17),
            )

    def test_matched_score_penalizes_extreme_duration(self):
        scores = matched_donor_scores(
            target_samples=100,
            target_mean_power_w=200,
            target_start_power_w=20,
            target_end_power_w=30,
            candidate_samples=np.array([100, 1000]),
            candidate_mean_power_w=np.array([200, 200]),
            candidate_start_power_w=np.array([20, 20]),
            candidate_end_power_w=np.array([30, 30]),
        )
        self.assertEqual(float(scores[0]), 0.0)
        self.assertGreater(float(scores[1]), 2.0)

    def test_boundary_jump_uses_adjacent_samples(self):
        jumps = boundary_jumps(np.array([1, 2, 10, 12]), [2])
        np.testing.assert_allclose(jumps, [8])

    def test_common_schedule_is_idle_continuous_and_nonoverlapping(self):
        timestamps = np.arange(0, 240, 6, dtype=np.int64)
        appliance = np.zeros(len(timestamps))
        appliance[18:20] = 100
        runs = idle_runs(timestamps, appliance, sample_seconds=6)
        starts = schedule_lengths(
            runs, np.array([5, 4]), total_rows=len(timestamps),
            guard_samples=1, seed=17)
        occupied = set()
        for start, length in zip(starts, [5, 4]):
            chosen = set(range(int(start), int(start + length)))
            self.assertFalse(chosen & occupied)
            self.assertTrue(np.all(appliance[start:start + length] == 0))
            occupied |= chosen


class NILMWindowTest(unittest.TestCase):
    def test_windows_do_not_cross_gaps_or_exclusions(self):
        timestamps = np.array([0, 6, 12, 18, 24, 60, 66, 72, 78, 84, 90])
        ranges = valid_center_ranges(
            timestamps, window_length=3, sample_seconds=6, stride=1,
            excluded_intervals=[(72, 72)])
        centers = []
        for first, stop, stride, _ in ranges:
            centers.extend(range(first, stop, stride))
        self.assertEqual(centers, [1, 2, 3, 9])
        for center in centers:
            self.assertEqual(timestamps[center + 1] - timestamps[center - 1], 12)

    def test_nilm_metrics(self):
        result = nilm_metrics([0, 100, 100, 0], [0, 100, 0, 100], threshold_w=20)
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["fn"], 1)
        self.assertAlmostEqual(result["f1"], 0.5)
        self.assertAlmostEqual(result["mae_w"], 50.0)


if __name__ == "__main__":
    unittest.main()
