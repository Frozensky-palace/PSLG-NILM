import copy
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch

import pandas as pd
import yaml

from scripts.downsample_aligned_pairs import sha256, write_json
from scripts.run_sampling_state_smoke import (
    METADATA_NAMES, ROOT, package_diagnostics, rate_config, run,
    validate_state_outputs, verify_prepared,
)
from scripts.train_state_discovery import seed_tensorflow_runtime


def fixture(root):
    source = root / "accepted"
    source.mkdir()
    write_json(source / "run_manifest.json", {
        "status": "completed", "test_arrays_read": False, "model_training_performed": False,
        "environment": {"repo": {"commit": {"value": "accepted"}}}})
    write_json(source / "preparation_summary.json", {"eligibility_changes": {
        p: {"lost_eligible_cycle_ids": [], "gained_eligible_cycle_ids": []}
        for p in ("train", "validation")}})
    for name in ("source_verification.json", "source_verification_after.json"):
        write_json(source / name, {"passed": True})
    for rate in ("6s", "12s"):
        lib, segments = source / rate / "train_library", source / rate / "train_segments"
        (lib / "cycles").mkdir(parents=True)
        segments.mkdir()
        rows, mapping = [], []
        for i in range(8):
            cycle, name, npz = f"cycle_{i}", f"cycle_{i:04d}.csv", f"cycles/cycle_{i:04d}.npz"
            (lib / npz).write_bytes(f"test-only waveform {rate} {i}".encode())
            (segments / name).write_text("timestamp,power\n0,20\n")
            rows.append({"cycle_id": cycle, "partition": "train", "path": npz, "samples": 8})
            mapping.append({"cycle_id": cycle, "partition": "train", "csv_idx": i,
                            "filename": name, "source_npz": npz, "samples": 8})
        pd.DataFrame(rows).to_csv(lib / "real_cycle_library.csv", index=False)
        pd.DataFrame(mapping).to_csv(segments / "segment_source_map.csv", index=False)
        write_json(lib / "real_cycle_library_manifest.json", {"sample_seconds": int(rate[:-1])})
        write_json(segments / "segment_export_manifest.json", {"partition": "train"})
        for kind, folder in (("train_library", lib), ("train_segments", segments)):
            record = source / f"records/{rate}_{kind}/run_manifest.json"
            record.parent.mkdir(parents=True)
            write_json(record, {"status": "completed", "outputs": [
                {"path": str(p), "sha256": sha256(p)} for p in sorted(folder.rglob("*")) if p.is_file()]})
    return {"prepared_dir": str(source), "preparation_commit": "accepted", "expected_train_cycles": 8,
            "smoke_cycles": 8, "smoke_epochs": 2, "runtime_seed": 42,
            "base_config": "config/experiments/core_wm_state_discovery_detsec_pc.yaml",
            "python": {"torch": sys.executable, "tensorflow": sys.executable},
            "expected_metadata_hashes": {name: sha256(source / name) for name in METADATA_NAMES}}


def repin(config, relative):
    """Only test deliberately accepted-but-inconsistent metadata."""
    source = Path(config["prepared_dir"])
    if relative.endswith("segment_source_map.csv"):
        record_relative = f"records/{relative.split('/')[0]}_train_segments/run_manifest.json"
        record = json.loads((source / record_relative).read_text())
        for entry in record["outputs"]:
            if entry["path"] == str(source / relative):
                entry["sha256"] = sha256(source / relative)
        write_json(source / record_relative, record)
        relative = record_relative
    config["expected_metadata_hashes"][relative] = sha256(source / relative)


class SamplingStateSmokeTests(unittest.TestCase):
    def test_accepted_data_checks_all_train_files_without_validation_or_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = fixture(root)
            report = verify_prepared(config, root / "verified.json")
            self.assertTrue(report["passed"])
            self.assertEqual(report["verified_train_files"], 40)
            self.assertEqual(report["selected_cycle_ids"], [f"cycle_{i}" for i in range(8)])
            self.assertFalse(report["test_arrays_read"])

    def test_changed_metadata_waveform_and_extra_files_fail(self):
        for kind in ("metadata", "waveform", "extra"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = fixture(root)
                relative = {"metadata": "run_manifest.json", "waveform": "6s/train_library/cycles/cycle_0000.npz",
                            "extra": "12s/train_segments/unrecorded.csv"}[kind]
                (Path(config["prepared_dir"]) / relative).write_text("changed")
                with self.assertRaises(ValueError):
                    verify_prepared(config, root / "failed.json")
                self.assertFalse(json.loads((root / "failed.json").read_text())["passed"])

    def test_missing_pins_and_inconsistent_paired_order_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = fixture(root)
            bad = copy.deepcopy(config)
            del bad["expected_metadata_hashes"]["source_verification.json"]
            with self.assertRaisesRegex(ValueError, "eight"):
                verify_prepared(bad, root / "missing.json")
            relative = "12s/train_segments/segment_source_map.csv"
            path = Path(config["prepared_dir"]) / relative
            mapping = pd.read_csv(path)
            mapping[["cycle_id", "source_npz"]] = mapping[["cycle_id", "source_npz"]].iloc[::-1].to_numpy()
            mapping.to_csv(path, index=False)
            repin(config, relative)
            with self.assertRaisesRegex(ValueError, "ordered cycle IDs"):
                verify_prepared(config, root / "mismatch.json")

    def test_file_paths_cannot_escape_prepared_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = fixture(root)
            source = Path(config["prepared_dir"])
            relative = "records/6s_train_library/run_manifest.json"
            record = json.loads((source / relative).read_text())
            record["outputs"][0]["path"] = str(root / "outside.npz")
            write_json(source / relative, record)
            repin(config, relative)
            with self.assertRaisesRegex(ValueError, "escapes"):
                verify_prepared(config, root / "failed.json")

    def test_rate_configuration_preserves_model_and_sample_count_parameters(self):
        base = yaml.safe_load((ROOT / "config/experiments/core_wm_state_discovery_detsec_pc.yaml").read_text())
        snapshot = copy.deepcopy(base)
        for label, seconds in (("6s", 6), ("12s", 12)):
            result = rate_config(base, Path("/accepted"), label, Path("/fresh"))
            self.assertEqual(result["state_discovery"]["sample_seconds"], seconds)
            self.assertEqual(result["temporal_state_merge"]["fs"], 1 / seconds)
            self.assertEqual(result["time_segmentation"], base["time_segmentation"])
            self.assertEqual(result["time_clustering"], base["time_clustering"])
            self.assertEqual(result["feature_extract"], {**base["feature_extract"], "cache": False})
            self.assertEqual(result["temporal_state_merge"]["min_block_seconds"], 90)
        self.assertEqual(base, snapshot)

    def test_runtime_seed_is_explicitly_forwarded_to_keras(self):
        with patch.dict(sys.modules, {"tensorflow": types.SimpleNamespace(
                keras=types.SimpleNamespace(utils=types.SimpleNamespace(set_random_seed=lambda s: seeds.append(s))))}):
            seeds = []
            seed_tensorflow_runtime(42)
            self.assertEqual(seeds, [42])

    def test_backend_failure_stops_training_and_packages_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = fixture(root)
            with patch.dict(os.environ, {"SLURM_JOB_ID": "test_sampling_failure"}), \
                 patch("scripts.run_sampling_state_smoke.run_recorded", return_value=1) as launch:
                with self.assertRaisesRegex(RuntimeError, "environment_torch failed"):
                    run(config, root / "run", require_slurm=False)
            self.assertEqual(launch.call_count, 1)
            report = json.loads((root / "run/run_manifest.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertFalse((root / "run/6s").exists())
            self.assertTrue((root / "run/diagnostics.tar.gz").exists())

    def test_complete_runner_orchestration_with_mocked_gpu_and_training(self):
        # Exercises paths/arguments/auditing only, NOT evidence of GPU training.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = fixture(root)
            config["base_config"] = str(ROOT / config["base_config"])
            calls = []

            def fake_child(command, record_dir, *, inputs, outputs):
                calls.append(command)
                record_dir.mkdir(parents=True)
                (record_dir / "execution.log").write_text("mock successful child")
                if "train_state_discovery.py" not in command[1]:
                    write_json(outputs[0], {"all_passed": True})
                    return 0
                self.assertEqual(command[command.index("--smoke-cycles") + 1], "8")
                self.assertEqual(command[command.index("--epochs-override") + 1], "2")
                self.assertEqual(command[command.index("--runtime-seed") + 1], "42")
                cfg = yaml.safe_load(Path(command[command.index("--config") + 1]).read_text())
                self.assertFalse(cfg["feature_extract"]["cache"])
                seconds = cfg["state_discovery"]["sample_seconds"]
                self.assertEqual(cfg["temporal_state_merge"]["fs"], 1 / seconds)
                result = outputs[0]
                for path in outputs:
                    path.mkdir(parents=True)
                write_json(result / "discovery_summary.json", {
                    "status": "smoke", "train_only": True, "test_accessed": False})
                subset = result / "segments_smoke_subset"
                subset.mkdir()
                mapping = pd.read_csv(Path(cfg["paths"]["segments_dir"]) / "segment_source_map.csv")
                mapping.to_csv(subset / "segment_source_map.csv", index=False)
                for k in (3, 4, 5):
                    library = result / f"state_library_k{k}"
                    library.mkdir()
                    write_json(library / "state_library_summary.json", {
                        "cycle_coverage_exact": True, "all_records_train": True, "states": {"0": {}}})
                    pd.DataFrame([{"cycle_id": cycle, "source_partition": "train", "start_sample": 0,
                        "end_sample_exclusive": 8, "samples": 8, "duration_seconds": 8 * seconds,
                        "mean_power_w": 10, "energy_wh": 80*seconds/3600}
                        for cycle in mapping.cycle_id]).to_csv(library / "state_inventory.csv", index=False)
                return 0

            with patch.dict(os.environ, {"SLURM_JOB_ID": "test_complete"}), \
                 patch("scripts.run_sampling_state_smoke.ROOT", root / "repo"), \
                 patch("scripts.run_sampling_state_smoke.run_recorded", side_effect=fake_child):
                result = run(config, root / "run", require_slurm=False)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(len(calls), 6)
            summary = json.loads((root / "run/state_smoke_summary.json").read_text())
            self.assertEqual(summary["status"], "paired_smoke_passed_not_formal_result")
            self.assertEqual(summary["rates"]["12s"]["4"]["cycles"], 8)
            self.assertTrue((root / "run/input_verification_after.json").is_file())

    def test_code_drift_prevents_launch_and_retains_failure_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = fixture(root)
            with patch("scripts.run_sampling_state_smoke.check_frozen_code",
                       side_effect=ValueError("checkout changed")), \
                 patch("scripts.run_sampling_state_smoke.run_recorded") as launch:
                with self.assertRaisesRegex(ValueError, "checkout changed"):
                    run(config, root / "run", require_slurm=False, expected_commit="frozen")
            launch.assert_not_called()
            self.assertTrue((root / "run/diagnostics.tar.gz").is_file())

    def test_login_and_overlap_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = fixture(root)
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "Slurm"):
                    run(config, root / "run")
            with self.assertRaisesRegex(ValueError, "overlap"):
                run(config, Path(config["prepared_dir"]) / "nested", require_slurm=False)

    def test_package_excludes_waveforms_and_includes_partial_workflow_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output, workflow = root / "run", root / "workflow"
            (output / "6s/segments_smoke_subset").mkdir(parents=True)
            (output / "6s/segments_smoke_subset/cycle_0000.csv").write_text("raw power")
            (output / "6s/segments_smoke_subset/segment_source_map.csv").write_text("provenance")
            (output / "6s/features.npy").write_bytes(b"large")
            (output / "6s/state_inventory.csv").write_text("statistics")
            workflow.mkdir()
            (workflow / "training_history.json").write_text("{}")
            (workflow / "X.npy").write_bytes(b"large")
            archive = package_diagnostics(output, {"6s": workflow})
            with tarfile.open(archive) as bundle:
                self.assertEqual(set(bundle.getnames()), {
                    "6s/state_inventory.csv", "6s/segments_smoke_subset/segment_source_map.csv",
                    "workflows/6s/training_history.json"})
            with self.assertRaises(FileExistsError):
                package_diagnostics(output, {})

    def test_state_coverage_gate_detects_missing_cycles_and_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / "discovery_summary.json", {"status": "smoke", "train_only": True, "test_accessed": False})
            (root / "segments_smoke_subset").mkdir()
            pd.DataFrame([{"csv_idx": 0, "cycle_id": "c", "samples": 8}]).to_csv(
                root / "segments_smoke_subset/segment_source_map.csv", index=False)
            for k in (3, 4, 5):
                library = root / f"state_library_k{k}"
                library.mkdir()
                write_json(library / "state_library_summary.json", {
                    "cycle_coverage_exact": True, "all_records_train": True, "states": {"0": {}}})
                pd.DataFrame([{"cycle_id": "c", "source_partition": "train", "start_sample": 0,
                               "end_sample_exclusive": 8, "samples": 8, "duration_seconds": 96,
                               "mean_power_w": 10, "energy_wh": 96*10/3600}]).to_csv(
                    library / "state_inventory.csv", index=False)
            self.assertTrue(validate_state_outputs(root, ["c"], 12)["4"]["exact_coverage_verified"])
            path = root / "state_library_k4/state_inventory.csv"
            frame = pd.read_csv(path)
            frame.loc[0, "start_sample"] = 1
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "reconstruct"):
                validate_state_outputs(root, ["c"], 12)
            frame.loc[0, "cycle_id"] = "not-selected"
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "coverage"):
                validate_state_outputs(root, ["c"], 12)


if __name__ == "__main__":
    unittest.main()
