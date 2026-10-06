"""Standard-library-only tests for the initial server inventory."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "inspect_server_setup.py"
SPEC = importlib.util.spec_from_file_location("setup_inventory", SCRIPT)
inventory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inventory)


class ServerSetupInventoryTests(unittest.TestCase):
    def test_package_metadata_including_pyyaml_and_missing_packages(self):
        def version(name):
            if name == "PyYAML":
                return "6.0.2"
            raise inventory.metadata.PackageNotFoundError(name)

        with patch.object(inventory.metadata, "version", side_effect=version):
            versions = inventory.package_versions()
        self.assertEqual(versions["PyYAML"]["version"], "6.0.2")
        self.assertEqual(versions["torch"]["status"], "not_installed")

    def test_broken_metadata_is_reported_without_aborting(self):
        with patch.object(inventory.metadata, "version", side_effect=ValueError("broken")):
            self.assertEqual(inventory.package_versions()["torch"]["status"], "metadata_error")

    def test_paths_are_metadata_only_and_missing_is_not_a_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            wave = Path(directory) / "channel_5.dat"
            wave.write_bytes(b"not a waveform")
            self.assertEqual(inventory.path_info(wave)["bytes"], 14)
            self.assertEqual(inventory.path_info(directory)["kind"], "directory")
            self.assertEqual(inventory.path_info(Path(directory) / "absent")["status"], "missing")
            self.assertEqual(wave.read_bytes(), b"not a waveform")

    def test_inaccessible_path_is_distinct_from_missing(self):
        with patch.object(Path, "stat", side_effect=PermissionError("denied")):
            self.assertEqual(inventory.path_info("somewhere")["status"], "inaccessible")

    def test_git_timeout_is_reported(self):
        with patch.object(inventory.subprocess, "run", side_effect=subprocess.TimeoutExpired("git", 10)):
            self.assertEqual(inventory.git_value(Path.cwd(), "status")["status"], "timeout")

    def test_inventory_whitelists_environment_and_runs_no_gpu_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"CONDA_DEFAULT_ENV": "test-env", "GITHUB_TOKEN": "not-for-report"}, clear=True), \
                    patch.object(inventory, "package_versions", return_value={}), \
                    patch.object(inventory, "git_value", return_value={"status": "ok", "value": ""}) as git, \
                    patch.object(inventory.shutil, "which", return_value=None), \
                    patch.object(inventory.subprocess, "run") as run:
                report = inventory.inspect_setup(directory, [Path(directory) / "missing"])
            self.assertEqual(report["environment"], {"CONDA_DEFAULT_ENV": "test-env"})
            self.assertNotIn("not-for-report", json.dumps(report))
            self.assertEqual(report["status"], "inventory_only_not_training_readiness")
            self.assertEqual(report["data_candidates"][0]["root"]["status"], "missing")
            self.assertEqual(git.call_count, 4)
            run.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_reports_get_unique_names_without_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            first = inventory.save_report({"value": 1}, directory)
            second = inventory.save_report({"value": 2}, directory)
            self.assertNotEqual(first, second)
            self.assertEqual(json.loads(first.read_text())["value"], 1)
            self.assertEqual(json.loads(second.read_text())["value"], 2)


if __name__ == "__main__":
    unittest.main()
