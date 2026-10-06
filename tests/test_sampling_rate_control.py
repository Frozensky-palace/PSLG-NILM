"""Sampling-rate control tests; NumPy/Pandas only, no neural frameworks."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.pairwise_sampling import coarsen_arrays, expected_count
from src.nilm.metrics import nilm_metrics
from scripts.downsample_aligned_pairs import convert, sha256
from scripts.compare_sampling_predictions import compare
from scripts.export_paired_generation_summaries import export as export_paired
from scripts.build_b1_b2_pilot_cycles import _save_cycle
from src.nilm.window_dataset import ShardedWindowDataset


def arrays(ts, power=None, segments=None):
    ts = np.asarray(ts, dtype=np.int64)
    p = np.asarray(power if power is not None else np.arange(len(ts)) * 10, dtype=float)
    return {"timestamp": ts, "appliance_w": p, "mains_w": p + 100,
            "background_signed_w": np.full(len(ts), 100.),
            "background_clipped_w": np.full(len(ts), 100.),
            "segment_id": np.asarray(segments if segments is not None else np.zeros(len(ts)), dtype=np.int64)}


class PairAveragingTests(unittest.TestCase):
    def test_mean_not_sum_and_energy_conservation(self):
        out, report = coarsen_arrays(arrays([0, 6, 12, 18], [10, 30, 50, 70]))
        np.testing.assert_array_equal(out["appliance_w"], [20, 60])
        np.testing.assert_array_equal(out["timestamp"], [0, 12])
        np.testing.assert_array_equal(out["source_left_row"], [0, 2])
        self.assertEqual(report["discarded_input_rows"], 0)
        self.assertAlmostEqual(report["energy"]["appliance_w"]["retained_energy_error_wh"], 0)

    def test_odd_tail_is_accounted_for(self):
        _, report = coarsen_arrays(arrays([0, 6, 12], [10, 30, 90]))
        self.assertEqual(report["discarded_input_rows"], 1)
        self.assertAlmostEqual(report["energy"]["appliance_w"]["discarded_input_wh"], 0.15)

    def test_gaps_and_segment_boundaries_are_not_crossed(self):
        out, _ = coarsen_arrays(arrays([0, 6, 18, 24, 30, 36, 42],
                                      segments=[0, 0, 1, 1, 1, 2, 3]))
        np.testing.assert_array_equal(out["timestamp"], [0, 24])
        np.testing.assert_array_equal(out["segment_id"], [0, 1])

    def test_epoch_phase_is_explicit(self):
        out, report = coarsen_arrays(arrays([6, 12, 18, 24]))
        np.testing.assert_array_equal(out["timestamp"], [12])
        self.assertEqual(report["discarded_input_rows"], 2)

    def test_on_off_boundary_does_not_create_artificial_gap(self):
        out, report = coarsen_arrays(arrays([0, 6, 12, 18], [0, 100, 100, 100]), intervals=[(6, 18)])
        np.testing.assert_array_equal(out["timestamp"], [0, 12])
        np.testing.assert_array_equal(out["source_cycle_owner"], [-2, 0])
        self.assertEqual(report["bins_crossing_cycle_boundary"], 1)

    def test_clip_after_signed_average(self):
        data = arrays([0, 6], [10, 10])
        data["mains_w"] = np.array([0., 20.])
        data["background_signed_w"] = np.array([-10., 10.])
        data["background_clipped_w"] = np.array([0., 10.])
        out, _ = coarsen_arrays(data)
        self.assertEqual(out["background_clipped_w"][0], 0)
        self.assertEqual(out["background_signed_w"][0], 0)

    def test_factor_one_retains_all_source_powers(self):
        source = arrays([6, 12, 18], [20, 30, 40])
        out, report = coarsen_arrays(source, factor=1)
        for key in source:
            np.testing.assert_array_equal(source[key], out[key])
        self.assertEqual(report["discarded_input_rows"], 0)

    def test_reject_invalid_inputs(self):
        for ts in ([0, 6, 6], [6, 0], [1, 7]):
            with self.assertRaises(ValueError):
                coarsen_arrays(arrays(ts))
        data = arrays([0, 6])
        data["appliance_w"][0] = np.nan
        with self.assertRaises(ValueError):
            coarsen_arrays(data)
        with self.assertRaises(ValueError):
            coarsen_arrays({**arrays([0, 6]), "label": np.array([0, 1])})
        with self.assertRaises(ValueError):
            coarsen_arrays(arrays([0, 6]), intervals=[(0, 6), (6, 12)])

    def test_complete_support_count(self):
        self.assertEqual(expected_count(6, 18, 6, 2), 1)
        self.assertEqual(expected_count(0, 12, 6, 2), 1)
        self.assertEqual(expected_count(0, 5, 6, 2), 0)


def fixture(root):
    source = root / "source"
    source.mkdir()
    manifest = {"sample_seconds": 6, "partitions": {}}
    coverage = []
    for index, part in enumerate(("train", "validation")):
        offset = index * 1200
        relative = Path(part) / "raw.npz"
        (source / part).mkdir()
        np.savez_compressed(source / relative, **arrays(offset + np.arange(200) * 6))
        manifest["partitions"][part] = {
            "interval_start_unix": offset, "interval_end_unix_exclusive": offset + 1200,
            "shards": [{"path": relative.as_posix(), "sha256": sha256(source / relative)}]}
        coverage.append({"cycle_id": part + "_1", "partition": part, "start_unix": offset + 6,
                         "end_unix": offset + 1188, "analysis_eligible": True,
                         "expected_grid_samples": 198})
    # Unavailable test arrays must not be read by the train/validation command.
    manifest["partitions"]["test"] = {"shards": [{"path": "test/never_read.npz"}]}
    (source / "aligned_partition_manifest.json").write_text(json.dumps(manifest))
    pd.DataFrame(coverage).to_csv(source / "cycle_alignment_coverage.csv", index=False)
    return source


class ConversionTests(unittest.TestCase):
    def test_conversion_and_inherited_library_export(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = fixture(root)
            originals = {str(p): sha256(p) for p in source.rglob("*") if p.is_file()}
            target = root / "coarse"
            report = convert(source, target)
            self.assertEqual(report["status"], "completed")
            self.assertFalse(report["test_transformed"])
            for p, fingerprint in originals.items():
                self.assertEqual(sha256(p), fingerprint)
            coverage = pd.read_csv(target / "cycle_alignment_coverage.csv")
            self.assertEqual(list(coverage.start_unix), [12, 1212])
            self.assertTrue(coverage.analysis_eligible.all())
            library = root / "library"
            subprocess.run([sys.executable, str(ROOT / "scripts/build_real_cycle_library.py"),
                            "--aligned-dir", str(target), "--output-dir", str(library)], check=True,
                           capture_output=True, text=True)
            subprocess.run([sys.executable, str(ROOT / "scripts/export_cycle_library_segments.py"),
                            "--library-dir", str(library), "--output-dir", str(root / "segments")],
                           check=True, capture_output=True, text=True)
            lib = json.loads((library / "real_cycle_library_manifest.json").read_text())
            self.assertEqual(lib["sample_seconds"], 12)
            self.assertEqual(lib["cycle_count"], 1)
            self.assertTrue((root / "segments/segment_source_map.csv").is_file())
            with self.assertRaises(FileExistsError):
                convert(source, target)

    def test_hash_failure_leaves_failed_record(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = fixture(root)
            (source / "train/raw.npz").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                convert(source, root / "bad")
            audit = json.loads((root / "bad/downsampling_audit.json").read_text())
            self.assertEqual(audit["status"], "failed")
            self.assertFalse((root / "bad/aligned_partition_manifest.json").exists())

    def test_nested_output_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            source = fixture(Path(folder))
            with self.assertRaises(ValueError):
                convert(source, source / "nested")


class ComparisonTests(unittest.TestCase):
    def test_common_bins_and_truth_mismatch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fields = {"arm": np.asarray("B0"), "partition": np.asarray("validation"), "seed": np.asarray(17)}
            np.savez(root / "native.npz", timestamp=[0, 6, 12, 18], y_true=[0., 40., 60., 80.],
                     y_pred=[10., 30., 50., 90.], sample_seconds=6, **fields)
            np.savez(root / "coarse.npz", timestamp=[0, 12], y_true=[20., 70.], y_pred=[25., 65.],
                     sample_seconds=12, **fields)
            report = compare(root / "native.npz", root / "coarse.npz", root / "result")
            self.assertEqual(report["common_bins"], 2)
            self.assertEqual(report["native_6s_model_on_common_12s"]["mae_w"], 0)
            self.assertEqual(report["retrained_12s_model_on_common_12s"]["mae_w"], 5)
            np.savez(root / "coarse.npz", timestamp=[0, 12], y_true=[200., 70.], y_pred=[25., 65.],
                     sample_seconds=12, **fields)
            with self.assertRaisesRegex(ValueError, "targets disagree"):
                compare(root / "native.npz", root / "coarse.npz", root / "invalid")

    def test_metrics_reject_nonfinite_and_use_sample_threshold(self):
        self.assertEqual(nilm_metrics([20, 0], [20, 0])["f1"], 1.)
        for value in (np.nan, np.inf, -np.inf):
            with self.assertRaises(ValueError):
                nilm_metrics([value], [0])

    def test_bad_time_grid_and_unknown_seed_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fields = {"arm": np.asarray("B0"), "partition": np.asarray("validation"),
                      "seed": np.asarray(-1)}
            np.savez(root / "native.npz", timestamp=[0, 6], y_true=[0., 40.],
                     y_pred=[10., 30.], sample_seconds=6, **fields)
            np.savez(root / "coarse.npz", timestamp=[0], y_true=[20.], y_pred=[25.],
                     sample_seconds=12, **fields)
            with self.assertRaisesRegex(ValueError, "seed is unknown"):
                compare(root / "native.npz", root / "coarse.npz", root / "result")
            np.savez(root / "coarse.npz", timestamp=[0.5], y_true=[20.], y_pred=[25.],
                     sample_seconds=12, **fields)
            with self.assertRaisesRegex(ValueError, "epoch-grid"):
                compare(root / "native.npz", root / "coarse.npz", root / "result")


class DownstreamInterfaceTests(unittest.TestCase):
    def test_center_metadata_matches_window_targets_across_shards(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            sources = []
            for shard in range(2):
                path = root / f"shard_{shard}.npz"
                np.savez(path, **arrays(np.arange(12) * 12 + shard * 240))
                sources.append({"path": str(path), "mains_field": "mains_w",
                                "appliance_field": "appliance_w"})
            (root / "dataset_manifest.json").write_text(json.dumps({
                "sample_seconds": 12, "window_length": 3,
                "sources": {"B0": {"validation": sources}}}))
            (root / "normalization.json").write_text(json.dumps({
                name: {"mean": 0, "std": 1} for name in ("mains_w", "appliance_w")}))
            pd.DataFrame([
                {"partition": "validation", "shard_index": 0, "first_center": 1, "stride": 2, "count": 3},
                {"partition": "validation", "shard_index": 0, "first_center": 8, "stride": 1, "count": 2},
                {"partition": "validation", "shard_index": 1, "first_center": 2, "stride": 3, "count": 3},
            ]).to_csv(root / "window_ranges.csv", index=False)
            dataset = ShardedWindowDataset(root, arm="B4", partition="validation")
            try:
                indices = np.array([7, 0, 4, 3, 5])
                info = dataset.center_metadata(indices)
                np.testing.assert_array_equal(info["shard_index"], [1, 0, 0, 0, 1])
                np.testing.assert_array_equal(info["center_row"], [8, 1, 9, 8, 2])
                np.testing.assert_array_equal(info["timestamp"], [336, 12, 108, 96, 264])
                np.testing.assert_array_equal([dataset[int(i)][1] for i in indices],
                                              info["center_row"] * 10)
                self.assertEqual(len(dataset.center_metadata([])["timestamp"]), 0)
                for invalid in ([-1], [len(dataset)], [1.5], [[1]]):
                    with self.assertRaises(ValueError):
                        dataset.center_metadata(invalid)
            finally:
                dataset.close()

    def test_paired_adapter_checks_both_arms_before_writing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            row = {"template_cycle_id": "real_train_1", "samples": 4}
            for arm in ("B1", "B2"):
                relative = Path(arm) / "cycles/synthetic_0000.npz"
                _save_cycle(root / relative, np.arange(4), 12)
                row[arm.lower() + "_path"] = relative.as_posix()
                row[arm.lower() + "_sha256"] = sha256(root / relative)
            pd.DataFrame([row]).to_csv(root / "paired_cycle_plan.csv", index=False)
            with self.assertRaisesRegex(ValueError, "rate/length"):
                export_paired(root, 6)
            self.assertFalse((root / "B1/generation_summary.json").exists())
            export_paired(root, 12)
            for arm in ("B1", "B2"):
                result = json.loads((root / arm / "generation_summary.json").read_text())
                self.assertEqual(result["count"], 1)
                self.assertEqual(result["records"][0]["synthetic_cycle_id"], "synthetic_0000")
            with self.assertRaises(FileExistsError):
                export_paired(root, 12)


if __name__ == "__main__":
    unittest.main()
