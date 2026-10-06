import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.downsample_aligned_pairs import sha256
from scripts.prepare_sampling_control import prepare, verify_inputs
from tests.test_sampling_rate_control import fixture


def prep_fixture(root, short_cycle=False):
    source = fixture(root)
    inventory = root / "cycle_inventory_with_split.csv"
    coverage = source / "cycle_alignment_coverage.csv"
    frame = pd.read_csv(coverage)
    if short_cycle:
        train = frame.partition == "train"
        frame.loc[train, "end_unix"] = 1176
        frame.loc[train, "expected_grid_samples"] = 196
        frame = pd.concat([frame, pd.DataFrame([{
            "cycle_id": "short_train", "partition": "train", "start_unix": 1194, "end_unix": 1194,
            "expected_grid_samples": 1, "analysis_eligible": True}])], ignore_index=True)
        frame.to_csv(coverage, index=False)
    frame[["cycle_id", "partition", "start_unix", "end_unix"]].to_csv(inventory, index=False)
    manifest_path = source / "aligned_partition_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_inventory_sha256"] = sha256(inventory)
    manifest["source_inventory"] = "D:\\old\\cycle_inventory_with_split.csv"
    manifest_path.write_text(json.dumps(manifest))
    anchor = root / "dataset_manifest.json"
    anchor.write_text(json.dumps({"sample_seconds": 6, "source_hashes": {"aligned_manifest": sha256(manifest_path)}}))
    spec = {"source_aligned_dir": str(source), "source_inventory_path": str(inventory),
            "anchor_dataset_manifest": str(anchor), "partitions": ["train", "validation"],
            "expected_eligible_cycles": {"train": 2 if short_cycle else 1, "validation": 1},
            "expected_hashes": {"aligned_manifest": sha256(manifest_path), "coverage": sha256(coverage),
                                "inventory": sha256(inventory), "anchor_dataset_manifest": sha256(anchor)}}
    return {"preparation": spec}


class SamplingPreparationTests(unittest.TestCase):
    def test_preparation_passes_continuous_shard_policy_and_restores_cycle(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = prep_fixture(root)
            spec = config["preparation"]
            spec["pair_across_continuous_storage_shards"] = True
            source = Path(spec["source_aligned_dir"])
            manifest_path = source / "aligned_partition_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["protocol"] = "aligned_research_partitions_v1"
            with np.load(source / "train/raw.npz") as data:
                arrays = {name: data[name] for name in data.files}
            shards = []
            for index, interval in enumerate((slice(0, 101), slice(101, None))):
                relative = f"train/split_{index}.npz"
                np.savez(source / relative, **{name: values[interval] for name, values in arrays.items()})
                shards.append({"path": relative, "sha256": sha256(source / relative)})
            manifest["partitions"]["train"]["shards"] = shards
            manifest_path.write_text(json.dumps(manifest))
            spec["expected_hashes"]["aligned_manifest"] = sha256(manifest_path)
            anchor = Path(spec["anchor_dataset_manifest"])
            anchor.write_text(json.dumps({"sample_seconds": 6, "source_hashes": {
                "aligned_manifest": sha256(manifest_path)}}))
            spec["expected_hashes"]["anchor_dataset_manifest"] = sha256(anchor)
            output = root / "run"
            prepare(config, output, require_slurm=False)
            summary = json.loads((output / "preparation_summary.json").read_text())
            train = summary["rates"]["12s"]["train"]
            self.assertEqual((train["input_rows"], train["output_rows"], train["discarded_input_rows"]), (200, 100, 0))
            self.assertEqual(train["eligible_cycles"], 1)
            self.assertEqual(train["cross_shard_pair_count"], 1)
            self.assertEqual(summary["rates"]["6s"]["train"]["cross_shard_pair_count"], 0)
            self.assertEqual(summary["eligibility_changes"]["train"]["lost_eligible_cycle_ids"], [])
            audit = json.loads((output / "12s/aligned/downsampling_audit.json").read_text())
            self.assertTrue(audit["pair_across_continuous_storage_shards"])
            self.assertTrue(json.loads((output / "source_verification_after.json").read_text())["passed"])

    def test_full_data_stage_and_small_diagnostic_archive(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = prep_fixture(root)
            source = Path(config["preparation"]["source_aligned_dir"])
            originals = {str(p): sha256(p) for p in source.rglob("*") if p.is_file()}
            # The fixture has a test reference but no test array. Every
            # subprocess must succeed without trying to load or hash it.
            output = root / "new_run"
            run = prepare(config, output, require_slurm=False)
            self.assertEqual(run["status"], "completed")
            self.assertEqual(len(run["steps_completed"]), 8)
            self.assertFalse(run["model_training_performed"])
            summary = json.loads((output / "preparation_summary.json").read_text())
            self.assertEqual(summary["rates"]["6s"]["train"]["output_rows"], 200)
            self.assertEqual(summary["rates"]["12s"]["train"]["output_rows"], 100)
            for part in ("train", "validation"):
                self.assertEqual(summary["eligibility_changes"][part]["lost_eligible_cycle_ids"], [])
            comparison = pd.read_csv(output / "cycle_comparison.csv")
            self.assertTrue((comparison.energy_delta_wh < 0).all())
            self.assertTrue((comparison.left_edge_shift_seconds == 6).all())
            for path, fingerprint in originals.items():
                self.assertEqual(sha256(path), fingerprint)
            self.assertFalse((output / "12s/validation_segments").exists())
            with tarfile.open(output / "diagnostics.tar.gz") as archive:
                names = archive.getnames()
                self.assertIn("preparation_summary.json", names)
                self.assertIn("source_verification_after.json", names)
                self.assertIn("12s/train_segments/segment_source_map.csv", names)
                self.assertFalse(any(n.endswith(".npz") or "/cycle_0000.csv" in n for n in names))
            with self.assertRaises(FileExistsError):
                prepare(config, output, require_slurm=False)

    def test_changed_waveform_fails_before_outputs_and_keeps_failure_package(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = prep_fixture(root)
            (root / "source/train/raw.npz").write_bytes(b"changed")
            output = root / "failed_run"
            with self.assertRaisesRegex(ValueError, "shard hash mismatch"):
                prepare(config, output, require_slurm=False)
            self.assertFalse((output / "6s").exists())
            self.assertEqual(json.loads((output / "run_manifest.json").read_text())["status"], "failed")
            self.assertFalse(json.loads((output / "source_verification.json").read_text())["passed"])
            self.assertTrue((output / "diagnostics.tar.gz").is_file())

    def test_changed_metadata_and_partition_selection_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = prep_fixture(root)
            config["preparation"]["expected_hashes"]["coverage"] = "not_the_frozen_hash"
            with self.assertRaisesRegex(ValueError, "metadata hash mismatch"):
                verify_inputs(config)
            config["preparation"]["partitions"].append("test")
            with self.assertRaisesRegex(ValueError, "restricted"):
                verify_inputs(config)

    def test_login_guard_and_source_overlap(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = prep_fixture(root)
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "Slurm"):
                    prepare(config, root / "no_login_work")
            self.assertFalse((root / "no_login_work").exists())
            with self.assertRaisesRegex(ValueError, "overlap"):
                prepare(config, root / "source/new", require_slurm=False)

    def test_commit_drift_leaves_failed_record(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = prep_fixture(root)
            with self.assertRaisesRegex(ValueError, "checkout changed"):
                prepare(config, root / "run", require_slurm=False, expected_commit="not_this_commit")
            self.assertTrue((root / "run/diagnostics.tar.gz").exists())

    def test_code_drift_during_a_step_stops_the_job(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = prep_fixture(root)
            with patch("scripts.prepare_sampling_control.check_frozen_code",
                       side_effect=[None, None, ValueError("checkout changed")]), \
                 patch("scripts.prepare_sampling_control.run_recorded", return_value=0) as command:
                with self.assertRaisesRegex(ValueError, "checkout changed"):
                    prepare(config, root / "run", require_slurm=False, expected_commit="frozen")
            self.assertEqual(command.call_count, 1)
            self.assertEqual(json.loads((root / "run/run_manifest.json").read_text())["status"], "failed")

    def test_packaging_failure_does_not_hide_primary_error(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = prep_fixture(root)
            (root / "source/train/raw.npz").write_bytes(b"changed")
            with patch("scripts.prepare_sampling_control.package_diagnostics", side_effect=OSError("archive failed")):
                with self.assertRaisesRegex(ValueError, "shard hash mismatch"):
                    prepare(config, root / "run", require_slurm=False)
            manifest = json.loads((root / "run/run_manifest.json").read_text())
            self.assertEqual(manifest["diagnostics_error"], "archive failed")

    def test_short_cycle_exclusion_is_reported_not_silently_matched(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = prep_fixture(root, short_cycle=True)
            prepare(config, root / "run", require_slurm=False)
            summary = json.loads((root / "run/preparation_summary.json").read_text())
            self.assertEqual(summary["eligibility_changes"]["train"]["lost_eligible_cycle_ids"], ["short_train"])
            self.assertEqual(summary["rates"]["6s"]["train"]["minimum_library_samples"], 1)
            self.assertEqual(summary["rates"]["12s"]["train"]["eligible_cycles"], 1)


if __name__ == "__main__":
    unittest.main()
