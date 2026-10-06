import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.inspect_sampling_control import inspect_raw, locate_manifests
from scripts.run_recorded_command import run_recorded


class SamplingServerToolsTests(unittest.TestCase):
    def test_raw_inspection_does_not_guess_channel(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            house = root / "house_1"
            house.mkdir()
            (house / "labels.dat").write_text("5 washing_machine\n54 aggregate\n1 boiler\n")
            (house / "channel_5.dat").write_text("100 20\n106 40\n112 60\n118 80\n")
            report = inspect_raw({"data_root": folder, "building": 1})
            self.assertEqual(report["channel_selection"], "not_automatically_selected")
            self.assertEqual([x["channel"] for x in report["label_matches"]], [5, 54])
            sample = next(x for x in report["candidate_heads"] if x["path"].endswith("channel_5.dat"))
            self.assertEqual(len(sample["first_lines"]), 3)

    def test_bounded_manifest_discovery(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "aligned_partition_manifest.json").write_text("{}")
            (root / ".git").mkdir()
            (root / ".git/dataset_manifest.json").write_text("{}")
            result = locate_manifests([root])
            self.assertEqual(len(result["files"]), 1)
            self.assertTrue(locate_manifests([root], max_directories=0)["truncated"])

    def test_success_and_failure_have_records(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch("scripts.run_recorded_command.inspect_setup", return_value={}):
                rc = run_recorded([sys.executable, "-c", "print('diagnostic')"], root / "success")
                failed = run_recorded([sys.executable, "-c", "raise SystemExit(7)"], root / "failure")
            self.assertEqual(rc, 0)
            self.assertEqual(failed, 7)
            record = json.loads((root / "failure/run_manifest.json").read_text())
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["returncode"], 7)
            self.assertEqual((root / "success/execution.log").read_text().strip(), "diagnostic")
            with self.assertRaises(FileExistsError):
                run_recorded([sys.executable, "-c", "pass"], root / "success")

    def test_missing_output_and_existing_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch("scripts.run_recorded_command.inspect_setup", return_value={}):
                rc = run_recorded([sys.executable, "-c", "pass"], root / "record",
                                  outputs=[root / "absent"])
            self.assertNotEqual(rc, 0)
            self.assertEqual(json.loads((root / "record/run_manifest.json").read_text())["status"], "failed")
            with self.assertRaises(FileExistsError):
                run_recorded([sys.executable, "-c", "pass"], root / "record2", outputs=[root / "record"])

    def test_slurm_guard_prevents_login_execution(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "Slurm"):
                run_recorded([sys.executable, "-c", "pass"], "must-not-exist", require_slurm=True)


if __name__ == "__main__":
    unittest.main()
