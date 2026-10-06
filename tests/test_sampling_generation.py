import copy
import hashlib
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from scripts import run_sampling_generation as runner
from scripts.collect_sampling_generation import collect, compare_primitive_plans
from scripts.downsample_aligned_pairs import write_json
from scripts.run_recorded_command import fingerprints
from scripts.run_sampling_state_smoke import verify_prepared
from src.generation.provenance import canonical_config_hash
from tests.test_sampling_state_smoke import fixture

ROOT = runner.ROOT


def settings():
    return json.loads((ROOT / "config/server/zzz_sampling_generation.json").read_text())


def waveform_fixture(directory, count=4, seed=17, seconds=6):
    (directory / "cycles").mkdir(parents=True)
    records = []
    for i in range(count):
        power = np.array([0, 10, 20, 30, 0], dtype=np.float32) + i
        cid = f"synthetic_{i:04d}"
        np.savez(directory / f"cycles/{cid}.npz", appliance_w=power,
                 relative_time_s=np.arange(5) * seconds)
        records.append({"synthetic_cycle_id": cid, "route": "B3", "seed": seed,
            "generator_name": "test", "generator_version": "1", "checkpoint_sha256": None,
            "conditions": {}, "state_path": [0], "segments": [{"state_label": 0,
                "target_duration_seconds": 5 * seconds, "actual_duration_seconds": 5 * seconds,
                "target_samples": 5, "actual_samples": 5, "mean_power_w": float(power.mean()),
                "energy_wh": float(power.sum() * seconds / 3600)}], "boundary_treatment": "none",
            "sample_seconds": seconds, "created_utc": "test",
            "waveform_sha256": hashlib.sha256(power.tobytes()).hexdigest()})
    write_json(directory / "generation_summary.json", {"count": count, "seed": seed,
               "sample_seconds": seconds, "records": records})


def reports(count=4, real=8):
    q = {"passed": True, "n_synthetic": count, "n_real_reference": real,
         "flags": {"duration_distribution": "WARN", "diversity": "PASS"},
         "diversity": {"sampled_cycles": count, "identical_pairs": 0,
                       "min_pairwise_distance": 0.2}}
    m = {"passed": True, "n_synthetic": count, "n_real_reference": real,
         "exact_duplicate_count": 0, "replication_rate": 0.0}
    return q, m


def handoff_fixture(root, config, inputs):
    root.mkdir()
    for rate in ("6s", "12s"):
        folder = root / rate / "state_library_k4"
        folder.mkdir(parents=True)
        (folder / "state_inventory.csv").write_text("test state inventory")
    ready = {"status": "ready_for_generation", "git_commit": "frozen",
        "config_hash": canonical_config_hash(config), "input_config_hash": canonical_config_hash(inputs),
        "states": {r: {} for r in ("6s", "12s")}, "selected_k": 4,
        "selected_cycle_ids": [f"cycle_{i}" for i in range(8)],
        "state_files": {r: fingerprints([root / r]) for r in ("6s", "12s")}}
    write_json(root / "state_ready.json", ready)


class SamplingGenerationTests(unittest.TestCase):
    def test_frozen_matrix_and_budgets(self):
        config, inputs = runner.load_config(ROOT / "config/server/zzz_sampling_generation.json")
        cells = runner.matrix(config)
        self.assertEqual(len(cells), 14)
        self.assertEqual({(c["arm"], c["rate"]) for c in cells},
                         {(arm, rate) for arm in runner.ARM_ROUTES for rate in ("6s", "12s")})
        self.assertEqual(config["generation_count"], 246)
        self.assertEqual(inputs["expected_train_cycles"], 493)
        self.assertEqual([c["epochs"] for c in cells[:7]], [0, 200, 150, 300, 200, 150, 300])
        self.assertEqual(cells[5]["learning_rate"], 0.0001)
        self.assertEqual(cells[2]["learning_rate"], 0.001)

    def test_all_routes_forward_true_sampling_units_and_inherited_scripts(self):
        config = settings()
        for cell in runner.matrix(config):
            commands = runner.command_specs(config, cell, Path("/real"), Path("/state"), Path("/out"))
            names = [c[0] for c in commands]
            self.assertEqual("train" in names, cell["arm"] != "B3T")
            for name, script, argv, _, _ in commands:
                args = list(map(str, argv))
                self.assertTrue((ROOT / "scripts" / script).is_file())
                if name != "memorization":
                    self.assertEqual(args[args.index("--sample-seconds") + 1], cell["rate"][:-1])
                if name == "generate":
                    self.assertEqual(args[args.index("--count") + 1], "246")
                    self.assertEqual(args[args.index("--seed") + 1], "17")
                if name == "quality_all_cycles":
                    self.assertEqual(args[args.index("--diversity-max-cycles") + 1], "246")
                if name == "quality":
                    cap = "246" if cell["arm"] in {"B4WGAN", "B4DIFF"} else "200"
                    self.assertEqual(args[args.index("--diversity-max-cycles") + 1], cap)

    def test_full_selection_is_not_smoke_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = fixture(root)
            inputs["smoke_cycles"] = 5
            self.assertEqual(len(verify_prepared(inputs, root / "verify.json", selection_cycles=8)["selected_cycle_ids"]), 8)

    def test_generated_integrity_and_mutations(self):
        for problem in (None, "interval", "power", "time", "duplicate", "segment", "hash"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                waveform_fixture(root, seconds=12)
                summary = json.loads((root / "generation_summary.json").read_text())
                if problem == "interval": summary["sample_seconds"] = 6
                if problem == "duplicate": summary["records"][1] = summary["records"][0]
                if problem == "segment": summary["records"][0]["segments"][0]["actual_samples"] = 4
                if problem == "hash": summary["records"][0]["waveform_sha256"] = "wrong"
                if problem in {"power", "time"}:
                    np.savez(root / "cycles/synthetic_0000.npz", appliance_w=np.array([0, np.nan if problem == "power" else 10, 20, 30, 0], dtype=np.float32), relative_time_s=np.arange(5) * (6 if problem == "time" else 12))
                write_json(root / "generation_summary.json", summary)
                if problem:
                    with self.assertRaises(ValueError): runner.validate_generated(root, 4, 17, 12)
                else:
                    self.assertTrue(runner.validate_generated(root, 4, 17, 12)["passed"])

    def test_memorization_json_passed_does_not_bypass_rate_cap(self):
        q, m = reports()
        m["replication_rate"] = 0.1
        result = runner.summarize_gates({"arm": "B3CVAE"}, q, q, m, {"memorization": 1})
        self.assertEqual(result["result"], "quality_rejected")
        self.assertFalse(result["memorization_passed_with_route_cap"])
        self.assertTrue(runner.summarize_gates({"arm": "B3T"}, q, q, m, {"memorization": 0})["all_gates_passed"])

    def test_b4wd_collapse_gate_and_nonfinite_metrics(self):
        q, m = reports()
        self.assertTrue(runner.summarize_gates({"arm": "B4DIFF"}, q, q, m, {})["all_gates_passed"])
        q["diversity"]["identical_pairs"] = 1
        self.assertFalse(runner.summarize_gates({"arm": "B4WGAN"}, q, q, m, {})["all_gates_passed"])
        m["replication_rate"] = float("nan")
        with self.assertRaises(ValueError): runner.summarize_gates({"arm": "B4"}, q, q, m, {})

    def test_handoff_rejects_changed_state_commit_and_configuration(self):
        for mutation in ("state", "commit", "config", "empty"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                inputs, config = fixture(root), settings()
                states = root / "states"
                handoff_fixture(states, config, inputs)
                runner.verify_handoff(states, config, inputs, "frozen")
                ready = json.loads((states / "state_ready.json").read_text())
                if mutation == "state": (states / "6s/state_library_k4/state_inventory.csv").write_text("changed")
                if mutation == "commit": ready["git_commit"] = "other"
                if mutation == "config": ready["config_hash"] = "other"
                if mutation == "empty": ready["state_files"] = {}
                write_json(states / "state_ready.json", ready)
                with self.assertRaises(ValueError): runner.verify_handoff(states, config, inputs, "frozen")

    def test_complete_generation_orchestration_and_negative_result(self):
        # Mock only GPU/training children; real preparation hashes, waveform
        # integrity, records, packaging and negative-result logic are exercised.
        for reject in (False, True):
            with self.subTest(reject=reject), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                inputs, config = fixture(root), settings()
                config["generation_count"] = 4
                config["arms"][1]["epochs"] = 2
                states = root / "states"
                handoff_fixture(states, config, inputs)
                calls = []
                def fake_child(command, record_dir, *, inputs, outputs):
                    name = record_dir.name
                    calls.append(name)
                    record_dir.mkdir(parents=True)
                    (record_dir / "execution.log").write_text("mock child")
                    if name in {"environment", "gpu"}: write_json(outputs[0], {"all_passed": True})
                    elif name == "train":
                        outputs[0].mkdir()
                        write_json(outputs[0] / "history.json", [{"loss": 2.}, {"loss": 1.}])
                        (outputs[0] / "model.pt").write_bytes(b"mock model")
                    elif name == "generate": waveform_fixture(outputs[0])
                    else:
                        q, m = reports()
                        if reject: m["replication_rate"] = 0.1
                        write_json(outputs[0], m if name == "memorization" else q)
                        if name == "memorization" and reject: return 1
                    return 0
                with patch("scripts.run_sampling_generation.check_frozen_code"), \
                     patch("scripts.run_sampling_generation.run_recorded", side_effect=fake_child):
                    report = runner.run(config, inputs, root / "run", stage="generator", expected_commit="frozen",
                                        state_root=states, task_index=1, require_slurm=False, mirror=False)
                self.assertEqual(report["status"], "completed")
                self.assertEqual(report["result"], "quality_rejected" if reject else "gates_passed_pending_review")
                self.assertEqual(calls, ["environment", "gpu", "train", "generate", "quality", "quality_all_cycles", "memorization"])
                with tarfile.open(root / "run/diagnostics.tar.gz") as archive:
                    self.assertIn("generation_result.json", archive.getnames())
                    self.assertFalse(any(n.endswith((".npz", ".pt")) for n in archive.getnames()))

    def test_formal_states_are_not_smoke_and_emit_verified_handoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs, config = fixture(root), settings()
            inputs["base_config"] = str(ROOT / inputs["base_config"])
            config["generation_count"] = 4
            def fake_child(command, record_dir, *, inputs, outputs):
                record_dir.mkdir(parents=True)
                (record_dir / "execution.log").write_text("mock child")
                if record_dir.name in {"environment", "gpu"}:
                    write_json(outputs[0], {"all_passed": True})
                    return 0
                self.assertNotIn("--smoke-cycles", command)
                cfg = yaml.safe_load(Path(command[command.index("--config") + 1]).read_text())
                self.assertEqual(cfg["feature_extract"]["epochs"], 50)
                for path in outputs: path.mkdir(parents=True)
                seconds = cfg["state_discovery"]["sample_seconds"]
                write_json(outputs[0] / "discovery_summary.json", {"status": "formal_candidate", "train_only": True, "test_accessed": False})
                history = outputs[1] / "FeatureExtract_detsec_pc_on_prim-glr"
                history.mkdir()
                write_json(history / "training_history.json", {"epochs_trained": 7, "loss": [1., 0.5]})
                for k in (3, 4, 5):
                    lib = outputs[0] / f"state_library_k{k}"
                    lib.mkdir()
                    write_json(lib / "state_library_summary.json", {"cycle_coverage_exact": True,
                               "all_records_train": True, "states": {"0": {}}})
                    pd.DataFrame([{"cycle_id": f"cycle_{i}", "source_partition": "train", "start_sample": 0,
                        "end_sample_exclusive": 8, "samples": 8, "duration_seconds": 8 * seconds,
                        "mean_power_w": 10, "energy_wh": 1.0} for i in range(8)]).to_csv(lib / "state_inventory.csv", index=False)
                return 0
            with patch("scripts.run_sampling_generation.ROOT", root / "repo"), \
                 patch("scripts.run_sampling_generation.check_frozen_code"), \
                 patch("scripts.run_sampling_generation.run_recorded", side_effect=fake_child):
                report = runner.run(config, inputs, root / "run", stage="states", expected_commit="frozen", require_slurm=False, mirror=False)
            self.assertEqual(report["status"], "completed")
            ready = runner.verify_handoff(root / "run", config, inputs, "frozen")
            self.assertEqual(len(ready["selected_cycle_ids"]), 8)

    def test_bad_state_handoff_stops_before_gpu_or_training_and_leaves_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs, config = fixture(root), settings()
            with patch("scripts.run_sampling_generation.check_frozen_code"), \
                 patch("scripts.run_sampling_generation.run_recorded") as launch:
                with self.assertRaises(FileNotFoundError):
                    runner.run(config, inputs, root / "run", stage="generator", expected_commit="frozen",
                               state_root=root / "missing", task_index=0, require_slurm=False, mirror=False)
            launch.assert_not_called()
            self.assertTrue((root / "run/diagnostics.tar.gz").is_file())

    def test_login_overlap_and_existing_output_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs, config = fixture(root), settings()
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(RuntimeError):
                    runner.run(config, inputs, root / "run", stage="states", expected_commit="frozen")
            with self.assertRaises(ValueError):
                runner.run(config, inputs, Path(inputs["prepared_dir"]) / "nested", stage="states", expected_commit="frozen", require_slurm=False)
            existing = root / "already_run"
            existing.mkdir()
            with self.assertRaises(FileExistsError):
                runner.run(config, inputs, existing, stage="states", expected_commit="frozen", require_slurm=False)

    def test_mirror_never_overwrites_previous_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, dest = root / "source", root / "log/dest"
            source.write_bytes(b"diagnostics")
            runner.mirror_diagnostics(source, dest)
            self.assertEqual(dest.read_bytes(), source.read_bytes())
            with self.assertRaises(FileExistsError): runner.mirror_diagnostics(source, dest)

    def test_collector_records_missing_cells_and_does_not_claim_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report, archive = collect(settings(), {}, root / "artifacts", root / "log", "1", "2")
            self.assertEqual(len(report["cells"]), 14)
            self.assertFalse(report["matrix_complete"])
            with tarfile.open(archive) as bundle:
                self.assertEqual(bundle.getnames(), ["matrix_summary.json"])

    def test_primitive_pairing_rejects_different_paths_lengths_and_rng_seeds(self):
        base = {"records": [{"synthetic_cycle_id": "synthetic_0000", "state_path": [1],
                "segments": [{"target_samples": 20}], "conditions": {"rng_seeds": [123]}}]}
        summaries = {(r, a): copy.deepcopy(base) for r in ("6s", "12s") for a in ("B4", "B4WGAN", "B4DIFF")}
        self.assertEqual(compare_primitive_plans(summaries)["6s"]["status"], "passed")
        summaries[("12s", "B4")]["records"][0]["segments"][0]["target_samples"] = 21
        self.assertEqual(compare_primitive_plans(summaries)["12s"]["status"], "mismatch")

    def test_collector_complete_matrix_still_rejects_primitive_pairing_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, inputs = settings(), {}
            artifacts = root / "artifacts"
            for cell in runner.matrix(config):
                folder = artifacts / f"sampling_generation_2_{cell['index']}"
                folder.mkdir(parents=True)
                write_json(folder / "run_manifest.json", {"cell": cell, "status": "completed",
                    "git_commit": "frozen", "state_root": str(artifacts / "sampling_states_1")})
                write_json(folder / "config_snapshot.json", {"experiment": config, "inputs": inputs})
                write_json(folder / "generation_result.json", {"gates": {"all_gates_passed": True}})
                if cell["arm"] in {"B4", "B4WGAN", "B4DIFF"}:
                    (folder / "cycles").mkdir()
                    write_json(folder / "cycles/generation_summary.json", {"records": [{
                        "synthetic_cycle_id": "synthetic_0000", "state_path": [0],
                        "segments": [{"target_samples": 10 if cell["arm"] == "B4" else 11}],
                        "conditions": {"rng_seeds": [17]}}]})
            report, _ = collect(config, inputs, artifacts, root / "log", "1", "2")
            self.assertTrue(report["matrix_complete"])
            self.assertFalse(report["all_gates_passed"])
            self.assertEqual(report["primitive_plan_pairing"]["6s"]["status"], "mismatch")

    def test_slurm_specs_preserve_activation_gpu_allocation_and_serial_array(self):
        for name in ("sampling_generation_states.sbatch", "sampling_generation_array.sbatch"):
            text = (ROOT / "slurm" / name).read_text()
            self.assertIn("#SBATCH --nodelist=h104-slurm-a", text)
            self.assertIn("conda activate pslg-nilm", text)
            self.assertNotIn("export CUDA_VISIBLE_DEVICES", text)
            self.assertIn("PSLG_SAMPLING_COMMIT", text)
        self.assertIn("#SBATCH --array=0-13%1", (ROOT / "slurm/sampling_generation_array.sbatch").read_text())


if __name__ == "__main__":
    unittest.main()
