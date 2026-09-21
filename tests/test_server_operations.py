from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.collect_run_artifacts import collect_missing
from scripts.finalize_server_run import build_artifact_manifest
from scripts.freeze_protocol_before_test import inspect_run
from scripts.query_slurm_registry import job_ids_from_registry, parse_pipe_rows
from scripts.prepare_server_matrices import materialize
from scripts.package_manifest_bundle import validated_files
from scripts.submit_slurm_matrix import (
    append_registry,
    build_command,
    load_matrix,
    normalize_exports,
)
from src.nilm.trainer import update_early_stopping


class SubmitMatrixTests(unittest.TestCase):
    def test_load_matrix_requires_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "matrix.yaml"
            path.write_text("template: job.sbatch\njobs: []\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_matrix(path)

    def test_normalize_exports_rejects_bad_name(self) -> None:
        with self.assertRaises(ValueError):
            normalize_exports({"bad-name": "x"})

    def test_normalize_exports_rejects_delimiters(self) -> None:
        with self.assertRaises(ValueError):
            normalize_exports({"GOOD": "a,b"})

    def test_build_command_is_argument_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            template = Path(tmp) / "job.sbatch"
            template.write_text("#!/bin/bash\n", encoding="utf-8")
            command = build_command(template, {"B": "2", "A": "1"},
                                    "afterok:123")
            self.assertEqual(command[0:2], ["sbatch", "--parsable"])
            self.assertIn("--export=ALL,A=1,B=2", command)
            self.assertIn("--dependency=afterok:123", command)

    def test_append_registry_preserves_existing_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.json"
            append_registry(path, [{"job_id": "2"}])
            append_registry(path, [{"job_id": "3"}])
            records = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual([item["job_id"] for item in records], ["2", "3"])

    def test_build_command_places_extra_before_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            template = Path(tmp) / "job.sbatch"
            template.write_text("#!/bin/bash\n", encoding="utf-8")
            command = build_command(template, {"A": "1"}, None,
                                    ["-w", "h104-slurm-a"])
            self.assertEqual(command[-3:], ["-w", "h104-slurm-a",
                                            str(template)])
            self.assertEqual(command[0:2], ["sbatch", "--parsable"])


class MatrixMaterializationTests(unittest.TestCase):
    def test_custom_clone_directory_propagates_to_all_matrices(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            project = base / "PSLG-NILM-c1"
            templates = project / "config" / "server"
            templates.mkdir(parents=True)
            for name in ("c1_gpu_smoke_matrix.yaml", "c2_detsec_matrix.yaml",
                         "c3_seq2point_matrix.yaml"):
                (templates / name).write_text(
                    "template: ../../slurm/job.sbatch\ncommon: {}\njobs:\n  - {}\n",
                    encoding="utf-8")
            output = base / "matrices"
            with patch("scripts.prepare_server_matrices.git_commit",
                       return_value="abc123"):
                paths = materialize(
                    project, output, base / "manifests", base / "artifacts",
                    project / "reports" / "inputs", ratio="1p0")
            self.assertEqual(paths[-1].name,
                             "c3_seq2point_r1p0.server.yaml")
            for path in paths:
                content = path.read_text(encoding="utf-8")
                self.assertIn(str(project), content)
            c3 = load_matrix(paths[-1])
            self.assertEqual(c3["common"]["PSLG_RATIO"], "1p0")
            self.assertEqual(
                c3["template"], str(project / "slurm" / "c3_seq2point.sbatch"))


class ArtifactTests(unittest.TestCase):
    def test_manifest_excludes_itself(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("a", encoding="utf-8")
            (root / "artifact_manifest.json").write_text("old", encoding="utf-8")
            records = build_artifact_manifest(root)
            self.assertEqual([item["path"] for item in records], ["a.txt"])

    def test_nilm_validation_metrics_name_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            required = (
                "config.json", "git_commit.txt", "data_manifest.json",
                "environment.json", "stdout.log", "stderr.log",
                "history.json", "run_summary.md", "best_checkpoint.pt",
                "validation_predictions.npz", "validation_metrics.json",
            )
            for name in required:
                (root / name).write_text(name, encoding="utf-8")
            self.assertEqual(collect_missing(root), [])

    def test_bundle_rejects_parent_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "files": [{"path": "../secret", "bytes": 0,
                           "sha256": "0" * 64}],
            }), encoding="utf-8")
            with self.assertRaises(ValueError):
                validated_files(manifest, root)


class RegistryQueryTests(unittest.TestCase):
    def test_job_ids_are_valid_unique_and_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.json"
            path.write_text(json.dumps([
                {"job_id": "10"}, {"job_id": "2"}, {"job_id": "10"}
            ]), encoding="utf-8")
            self.assertEqual(job_ids_from_registry(path), ["2", "10"])

    def test_parse_pipe_rows(self) -> None:
        rows = parse_pipe_rows("12|RUNNING|00:01|node-a\n",
                               ["id", "state", "elapsed", "node"])
        self.assertEqual(rows[0]["state"], "RUNNING")


class ProtocolFreezeTests(unittest.TestCase):
    def _run_dir(self, root: Path, test_accessed: bool) -> Path:
        root.mkdir()
        (root / "training_summary.json").write_text(json.dumps({
            "arm": "B0", "test_accessed": test_accessed,
        }), encoding="utf-8")
        for name in ("config.json", "validation_metrics.json",
                     "best_checkpoint.pt"):
            (root / name).write_text(name, encoding="utf-8")
        return root

    def test_inspect_run_accepts_validation_only_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = self._run_dir(Path(tmp) / "run", False)
            record = inspect_run(run)
            self.assertEqual(record["group"], "B0")
            self.assertFalse(record["test_accessed"])

    def test_inspect_run_rejects_test_access(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = self._run_dir(Path(tmp) / "run", True)
            with self.assertRaises(ValueError):
                inspect_run(run)


class TrainingControlTests(unittest.TestCase):
    def test_new_best_resets_early_stopping_counter(self) -> None:
        improved, best, misses = update_early_stopping(4.0, 5.0, 3)
        self.assertTrue(improved)
        self.assertEqual(best, 4.0)
        self.assertEqual(misses, 0)

    def test_resume_state_keeps_best_epoch(self) -> None:
        from src.nilm.trainer import resume_training_state

        payload = {"epoch": 7, "best_val_mae": 84.99,
                   "config": {"_best_epoch": 5, "_history": [
                       {"epoch": 5, "val_mae_w": 84.99}],
                       "arm": "B0"}}
        start_epoch, best_val_mae, best_epoch, history = (
            resume_training_state(payload))
        self.assertEqual(start_epoch, 8)
        self.assertEqual(best_val_mae, 84.99)
        self.assertEqual(best_epoch, 5)
        self.assertEqual(history[0]["epoch"], 5)

    def test_resume_state_defaults_without_metadata(self) -> None:
        from src.nilm.trainer import resume_training_state

        start_epoch, best_val_mae, best_epoch, history = (
            resume_training_state({"epoch": 2, "best_val_mae": 10.0,
                                   "config": {}}))
        self.assertEqual((start_epoch, best_val_mae, best_epoch, history),
                         (3, 10.0, 0, []))

    def test_deterministic_cuda_env_setdefault(self) -> None:
        import src.nilm.trainer as trainer

        with patch.dict(os.environ):
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            trainer.ensure_deterministic_cuda_env()
            self.assertEqual(os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                             ":4096:8")
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
            trainer.ensure_deterministic_cuda_env()
            self.assertEqual(os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                             ":16:8")


class GpuFrameworkSmokeTests(unittest.TestCase):
    def test_output_writes_report_json(self) -> None:
        import scripts.gpu_framework_smoke as smoke

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "nested" / "smoke.json"
            with patch.object(sys, "argv", [
                    "gpu_framework_smoke", "--framework", "torch",
                    "--require-gpu", "false", "--output", str(output)]):
                with self.assertRaises(SystemExit) as exited:
                    smoke.main()
                self.assertEqual(exited.exception.code, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["all_passed"])
            self.assertFalse(report["require_gpu"])
            self.assertEqual(report["results"][0]["framework"], "torch")
            self.assertFalse(report["results"][0]["gpu_detected"])


if __name__ == "__main__":
    unittest.main()
