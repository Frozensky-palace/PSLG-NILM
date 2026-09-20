"""Tests for the C1 server bundle, preflight and audit tools (guide §5.2)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_no_test_access import audit_manifest, audit_run  # noqa: E402
from scripts.build_c1_server_bundle import (  # noqa: E402
    collect_state_discovery_paths,
    looks_like_test_path,
)
from scripts.collect_run_artifacts import collect_missing  # noqa: E402
from scripts.server_preflight import check_manifest  # noqa: E402


def _write_fake_library(root: Path, partitions=("train", "train")) -> Path:
    library = root / "library"
    segments = library / "segments"
    segments.mkdir(parents=True)
    pd.DataFrame({
        "csv_idx": range(len(partitions)),
        "filename": [f"cycle_{i:04d}.csv" for i in range(len(partitions))],
        "cycle_id": [f"c{i}" for i in range(len(partitions))],
        "partition": list(partitions),
        "source_npz": [f"cycles/cycle_{i:04d}.npz"
                       for i in range(len(partitions))],
        "start_unix": 0, "end_unix": 10, "samples": 2,
    }).to_csv(segments / "segment_source_map.csv", index=False)
    (segments / "segment_export_manifest.json").write_text("{}", encoding="utf-8")
    pd.DataFrame({
        "cycle_id": [f"c{i}" for i in range(len(partitions))],
        "partition": list(partitions),
        "path": [f"cycles/cycle_{i:04d}.npz" for i in range(len(partitions))],
    }).to_csv(library / "real_cycle_library.csv", index=False)
    (library / "real_cycle_library_manifest.json").write_text("{}", encoding="utf-8")
    for index in range(len(partitions)):
        pd.DataFrame({"timestamp": [1], "power": [1.0]}).to_csv(
            segments / f"cycle_{index:04d}.csv", index=False)
    return library


class StateDiscoveryBundleTests(unittest.TestCase):
    def test_collect_paths_train_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            library = _write_fake_library(root)
            paths = collect_state_discovery_paths(root, library)
            relatives = {path.as_posix() for path in paths}
            self.assertIn(
                "config/experiments/core_wm_state_discovery_detsec_pc.yaml",
                relatives)
            self.assertIn("scripts/train_state_discovery.py", relatives)
            self.assertTrue(any(p.endswith("cycle_0000.csv") and
                                "segments" in p for p in relatives))
            for relative in paths:
                self.assertFalse(looks_like_test_path(relative))

    def test_rejects_mixed_partitions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            library = _write_fake_library(root, partitions=("train", "test"))
            with self.assertRaises(SystemExit):
                collect_state_discovery_paths(root, library)

    def test_rejects_map_row_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            library = _write_fake_library(root)
            (library / "segments" / "cycle_9.csv").write_text(
                "timestamp,power\n1,2.0\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                collect_state_discovery_paths(root, library)


class PreflightManifestTests(unittest.TestCase):
    def _manifest(self, root: Path, good_sha: bool) -> Path:
        import hashlib

        data_file = root / "data" / "blob.bin"
        data_file.parent.mkdir(parents=True, exist_ok=True)
        payload = b"pslg"
        data_file.write_bytes(payload)
        sha = hashlib.sha256(payload).hexdigest()
        record = {"path": "data/blob.bin", "bytes": len(payload),
                  "sha256": sha if good_sha else "0" * 64}
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({
            "protocol": "test", "file_count": 1, "files": [record]}),
            encoding="utf-8")
        return manifest

    def test_valid_manifest_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            info = check_manifest(self._manifest(root, good_sha=True), root)
            self.assertEqual(info["files"], 1)
            self.assertEqual(info["test_path_count"], 0)

    def test_bad_sha_raises(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(RuntimeError):
                check_manifest(self._manifest(root, good_sha=False), root)

    def test_test_path_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._manifest(root, good_sha=True)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["files"][0]["path"] = "aligned_partitions_v2/test/shard.npz"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                check_manifest(manifest, root)


class RunArtifactTests(unittest.TestCase):
    def test_complete_run_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            for name in ("config.json", "git_commit.txt", "data_manifest.json",
                         "environment.json", "stdout.log", "stderr.log",
                         "history.json", "metrics.json", "run_summary.md",
                         "best_checkpoint.pt", "validation_predictions.npz"):
                (run / name).write_text("x", encoding="utf-8")
            self.assertEqual(collect_missing(run), [])

    def test_incomplete_run_lists_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing = collect_missing(Path(temporary))
            self.assertIn("config.json", missing)
            self.assertTrue(any("best_checkpoint.pt" in name
                                for name in missing))


class NoTestAuditTests(unittest.TestCase):
    def test_manifest_with_test_path_flagged(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "m.json"
            manifest.write_text(json.dumps({"files": [
                {"path": "aligned_partitions_v2/test/shard.npz"}]}),
                encoding="utf-8")
            audit = audit_manifest(manifest)
            self.assertFalse(audit["clean"])
            self.assertEqual(audit["test_path_count"], 1)

    def test_run_with_test_access_flagged(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "training_summary.json").write_text(
                json.dumps({"test_accessed": True}), encoding="utf-8")
            np.savez_compressed(run / "validation_predictions.npz",
                                y_true=np.zeros(3), y_pred=np.zeros(3),
                                partition=np.asarray("validation"))
            audit = audit_run(run)
            self.assertFalse(audit["clean"])

    def test_clean_run_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "training_summary.json").write_text(
                json.dumps({"test_accessed": False}), encoding="utf-8")
            self.assertTrue(audit_run(run)["clean"])


if __name__ == "__main__":
    unittest.main()
