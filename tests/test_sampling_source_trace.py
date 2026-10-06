import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.inspect_sampling_sources import (
    MetadataReader, candidate_repo_roots, resolve_reference, trace_sources,
)


def put_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class SourceTraceTests(unittest.TestCase):
    def test_find_old_project_and_match_metadata_without_reading_waveforms(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            current = root / "new_repo"
            current.mkdir()
            old = root / "projects/PSLG-NILM-c1"
            (old / ".git").mkdir(parents=True)
            aligned = old / "reports/aligned_partitions_v2"
            (aligned / "train").mkdir(parents=True)
            wave = aligned / "train/shard_000.npz"
            # Intentionally not an NPZ: reading waveform content would fail.
            wave.write_bytes(b"metadata-only check; never load this")
            inventory = old / "reports/cycle_inventory_with_split.csv"
            inventory.write_text("cycle_id,partition\nc1,train\n")
            put_json(aligned / "aligned_partition_manifest.json", {
                "sample_seconds": 6, "source_inventory": str(inventory),
                "source_inventory_sha256": hashlib.sha256(inventory.read_bytes()).hexdigest(),
                "partitions": {"train": {"shards": [{"path": "train/shard_000.npz"}]}}})
            (aligned / "cycle_alignment_coverage.csv").write_text("cycle_id,partition\nc1,train\n")
            aligned_hash = hashlib.sha256((aligned / "aligned_partition_manifest.json").read_bytes()).hexdigest()
            dataset = root / "artifacts/de_inputs_r0p5/dataset_manifest.json"
            put_json(dataset, {"sample_seconds": 6, "source_hashes": {"aligned_manifest": aligned_hash},
                               "sources": {"B0": {"train": [{"path": "reports/aligned_partitions_v2/train/shard_000.npz"}],
                                                  "test": [{"path": "reports/aligned_partitions_v2/test/not_transferred.npz"}]}}})
            config = {"project_root": str(current), "discovery_roots": [str(current), str(root / "projects"),
                                                                       str(root / "artifacts")]}
            original = wave.read_bytes()
            original_open = open
            def guarded_open(file, *args, **kwargs):
                if str(file).endswith(".npz"):
                    raise AssertionError("must not open arrays")
                return original_open(file, *args, **kwargs)
            with patch("builtins.open", guarded_open):
                result = trace_sources(config)
            data = result["datasets"][0]
            self.assertEqual(data["b0_references"][0]["status"], "unique")
            self.assertEqual(data["b0_references"][1]["status"], "unresolved")
            self.assertEqual(data["hash_matching_aligned_manifests"],
                             [str((aligned / "aligned_partition_manifest.json").resolve())])
            self.assertEqual(result["aligned_candidates"][0]["coverage_status"], "read")
            self.assertTrue(result["aligned_candidates"][0]["inventory_reference"]["matches"][0]["inventory_hash_matches"])
            self.assertEqual(wave.read_bytes(), original)
            # Path existence is not waveform verification or training readiness.
            self.assertEqual(result["status"], "source_metadata_only_not_training_readiness")

    def test_ambiguous_relative_references_are_not_selected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            roots = [root / "repo1", root / "repo2"]
            for path in roots:
                path.mkdir()
                (path / "data.npz").write_bytes(b"wave")
            reader = MetadataReader(roots)
            result = resolve_reference("data.npz", roots, reader)
            self.assertEqual(result["status"], "ambiguous")
            self.assertEqual(len(result["matches"]), 2)

    def test_path_escape_windows_and_symlink_are_not_followed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            scope = root / "allowed"
            scope.mkdir()
            outside = root / "dataset_manifest.json"
            put_json(outside, {"secret": "never include"})
            (scope / "dataset_manifest.json").symlink_to(outside)
            reader = MetadataReader([scope])
            self.assertEqual(reader.read(scope / "dataset_manifest.json")["status"], "outside_metadata_scope")
            self.assertEqual(resolve_reference("../dataset_manifest.json", [scope], reader)["status"],
                             "parent_reference_rejected")
            self.assertEqual(resolve_reference("D:\\old\\data.npz", [scope], reader)["status"],
                             "non_posix_reference")
            self.assertEqual(resolve_reference(str(outside), [scope], reader)["status"], "unresolved")

    def test_size_limits_and_invalid_json_are_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / "dataset_manifest.json"
            data.write_text("{" + " " * 25)
            with patch("scripts.inspect_sampling_sources.MAX_FILE_BYTES", 10):
                self.assertEqual(MetadataReader([root]).read(data)["status"], "metadata_size_limit")
            reader = MetadataReader([root])
            self.assertEqual(reader.read(data)["status"], "read_error")
            data.write_text("{}")
            with patch("scripts.inspect_sampling_sources.MAX_TOTAL_BYTES", 1):
                self.assertEqual(MetadataReader([root]).read(data)["status"], "metadata_size_limit")

    def test_missing_roots_and_files_are_honest(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "absent"
            config = {"project_root": str(path), "discovery_roots": [str(path)]}
            self.assertEqual(candidate_repo_roots(config), [path.resolve()])
            report = trace_sources(config)
            self.assertEqual(report["datasets"], [])
            self.assertEqual(report["aligned_candidates"], [])
            self.assertFalse(report["discovery"]["truncated"])

    def test_hash_mismatch_is_not_promoted_to_match(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            put_json(root / "dataset_manifest.json", {"source_hashes": {"aligned_manifest": "wrong"},
                                                       "sources": {"B0": {}}})
            put_json(root / "aligned_partition_manifest.json", {"sample_seconds": 6})
            report = trace_sources({"project_root": str(root), "discovery_roots": [str(root)]})
            self.assertEqual(report["datasets"][0]["hash_matching_aligned_manifests"], [])
            self.assertEqual(report["aligned_candidates"][0]["coverage_status"], "missing")


if __name__ == "__main__":
    unittest.main()
