"""Boundary-pair regressions with physical row provenance, without neural deps."""
from contextlib import redirect_stdout
import io
from itertools import combinations
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.downsample_aligned_pairs import convert, sha256
from src.data.pairwise_sampling import POWER_FIELDS, coarsen_arrays


def _arrays(timestamps, segments=None):
    timestamps = np.asarray(timestamps, dtype=np.int64)
    power = (timestamps % 101).astype(np.float64) + 17
    signed = np.linspace(-3., 7., len(timestamps))
    return {
        "timestamp": timestamps, "appliance_w": power,
        "mains_w": power + signed, "background_signed_w": signed,
        "background_clipped_w": np.maximum(signed, 0),
        "segment_id": np.asarray(segments if segments is not None else
                                 np.zeros(len(timestamps)), dtype=np.int64),
    }


def _split(data, cuts):
    """Emulate independently numbered, file-local continuous segments."""
    blocks = []
    for index, (start, stop) in enumerate(zip((0, *cuts), (*cuts, len(data["timestamp"])))):
        block = {key: values[start:stop].copy() for key, values in data.items()}
        block["segment_id"][:] = 100 + index
        blocks.append(block)
    return blocks


def _fixture(root, train_shards, validation_shards=None, protocol="aligned_research_partitions_v1"):
    source = root / "source"
    source.mkdir()
    parts = {"train": train_shards,
             "validation": validation_shards if validation_shards is not None else
             [_arrays([1200, 1206, 1212, 1218])]}
    manifest = {"protocol": protocol, "sample_seconds": 6, "partitions": {}}
    coverage = []
    for partition, blocks in parts.items():
        (source / partition).mkdir()
        entries = []
        all_ts = np.concatenate([block["timestamp"] for block in blocks])
        for index, block in enumerate(blocks):
            relative = f"{partition}/source_{index}.npz"
            np.savez_compressed(source / relative, **block)
            entries.append({"path": relative, "sha256": sha256(source / relative)})
        manifest["partitions"][partition] = {
            "shards": entries, "interval_start_unix": int(all_ts[0]),
            "interval_end_unix_exclusive": int(all_ts[-1] + 6),
        }
        coverage.append({"cycle_id": f"{partition}_cycle", "partition": partition,
                         "start_unix": int(all_ts[0]), "end_unix": int(all_ts[-1]),
                         "analysis_eligible": True})
    # The default conversion must never try to read these unavailable arrays.
    manifest["partitions"]["test"] = {"shards": [{"path": "test/do_not_read.npz"}]}
    (source / "aligned_partition_manifest.json").write_text(json.dumps(manifest))
    pd.DataFrame(coverage).to_csv(source / "cycle_alignment_coverage.csv", index=False)
    return source, parts, manifest


def _run(source, target, **kwargs):
    with redirect_stdout(io.StringIO()):
        return convert(source, target, **kwargs)


def _read_partition(target, partition):
    manifest = json.loads((target / "aligned_partition_manifest.json").read_text())
    blocks = []
    for entry in manifest["partitions"][partition]["shards"]:
        with np.load(target / entry["path"], allow_pickle=False) as data:
            blocks.append({key: data[key] for key in data.files})
    return {key: np.concatenate([block[key] for block in blocks]) for key in blocks[0]}


class SamplingShardBoundaryTests(unittest.TestCase):
    def assert_accounting(self, report, partition, source_blocks, output):
        records = [record for record in report["shards"] if record["partition"] == partition]
        input_rows = sum(len(block["timestamp"]) for block in source_blocks)
        self.assertEqual(sum(record["source_file_rows"] for record in records), input_rows)
        self.assertEqual(sum(record["input_rows"] for record in records), input_rows)
        dropped = sum(record["discarded_input_rows"] for record in records)
        self.assertEqual(dropped + 2 * len(output["timestamp"]), input_rows)
        for field in POWER_FIELDS:
            original_wh = sum(block[field].sum() for block in source_blocks) * 6 / 3600
            recorded_wh = sum(record["energy"][field]["input_wh"] for record in records)
            retained_wh = sum(record["energy"][field]["retained_input_wh"] for record in records)
            dropped_wh = sum(record["energy"][field]["discarded_input_wh"] for record in records)
            output_wh = sum(record["energy"][field]["output_wh"] for record in records)
            self.assertAlmostEqual(original_wh, recorded_wh)
            self.assertAlmostEqual(original_wh, retained_wh + dropped_wh)
            self.assertAlmostEqual(retained_wh, output_wh)
            self.assertAlmostEqual(output_wh, output[field].sum() * 12 / 3600)

    def assert_physical_provenance(self, output, source_blocks):
        used = []
        for index, timestamp in enumerate(output["timestamp"]):
            rows = []
            for side in ("left", "right"):
                shard = int(output[f"source_{side}_shard_index"][index])
                row = int(output[f"source_{side}_row"][index])
                self.assertGreaterEqual(shard, 0)
                self.assertLess(shard, len(source_blocks))
                self.assertGreaterEqual(row, 0)
                self.assertLess(row, len(source_blocks[shard]["timestamp"]))
                used.append((shard, row))
                rows.append({key: values[row] for key, values in source_blocks[shard].items()})
            self.assertEqual(rows[0]["timestamp"], timestamp)
            self.assertEqual(rows[1]["timestamp"], timestamp + 6)
            self.assertEqual(timestamp % 12, 0)
            for field in POWER_FIELDS:
                self.assertAlmostEqual(output[field][index], (rows[0][field] + rows[1][field]) / 2)
        self.assertEqual(len(used), len(set(used)), "each physical source point is used once at most")
        self.assertTrue(np.all(np.diff(output["timestamp"]) > 0))

    def test_arbitrary_two_and_three_way_splits_match_whole_input(self):
        data = _arrays(np.arange(8) * 6)
        expected, _ = coarsen_arrays(data)
        splits = [(cut,) for cut in range(1, 8)] + list(combinations(range(1, 8), 2))
        # Alternating singleton/exhausted logical blocks exercise repeated consumption.
        splits.append(tuple(range(1, 8)))
        for cuts in splits:
            with self.subTest(cuts=cuts), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                source, parts, manifest = _fixture(root, _split(data, cuts))
                hashes = {entry["path"]: entry["sha256"] for part in ("train", "validation")
                          for entry in manifest["partitions"][part]["shards"]}
                report = _run(source, root / "output", pair_across_shards=True)
                result = _read_partition(root / "output", "train")
                for field in ("timestamp", *POWER_FIELDS, "background_clipped_w"):
                    np.testing.assert_array_equal(result[field], expected[field])
                self.assert_physical_provenance(result, parts["train"])
                self.assert_accounting(report, "train", parts["train"], result)
                self.assertEqual(len(result["timestamp"]) * 2, len(data["timestamp"]))
                for relative, fingerprint in hashes.items():
                    self.assertEqual(sha256(source / relative), fingerprint)
                for record in report["shards"]:
                    for pair in record["boundary_pairs"]:
                        self.assertEqual(pair["left_source_sha256"], hashes[pair["left_source_path"]])
                        self.assertEqual(pair["right_source_sha256"], hashes[pair["right_source_path"]])
                self.assertFalse(report["source_shard_boundaries_preserved"])
                self.assertFalse(report["test_transformed"])

    def test_boundary_cycle_coverage_restored_and_protocol_is_versioned(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, _, _ = _fixture(root, _split(_arrays([0, 6, 12, 18, 24, 30]), (3,)))
            _run(source, root / "legacy")
            _run(source, root / "fixed", pair_across_shards=True)
            legacy = pd.read_csv(root / "legacy/cycle_alignment_coverage.csv").set_index("partition")
            fixed = pd.read_csv(root / "fixed/cycle_alignment_coverage.csv").set_index("partition")
            self.assertFalse(legacy.loc["train", "analysis_eligible"])
            self.assertEqual(legacy.loc["train", "observed_aligned_samples"], 2)
            self.assertTrue(fixed.loc["train", "analysis_eligible"])
            self.assertEqual(fixed.loc["train", "observed_aligned_samples"], 3)
            self.assertEqual(fixed.loc["train", "expected_grid_samples"], 3)
            manifest = json.loads((root / "fixed/aligned_partition_manifest.json").read_text())
            self.assertEqual(manifest["protocol"], "aligned_pairwise_sampling_v2")

    def test_real_162_second_gap_is_never_bridged(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, parts, _ = _fixture(root, [_arrays([0, 6, 12]), _arrays([174, 180, 186])])
            report = _run(source, root / "output", pair_across_shards=True)
            result = _read_partition(root / "output", "train")
            np.testing.assert_array_equal(result["timestamp"], [0, 180])
            self.assertEqual(sum(record["cross_shard_pair_count"] for record in report["shards"]), 0)
            self.assert_accounting(report, "train", parts["train"], result)
            self.assert_physical_provenance(result, parts["train"])

    def test_internal_segment_change_remains_a_barrier(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, parts, _ = _fixture(root, [_arrays([0, 6, 12, 18, 24], [0, 0, 0, 1, 1]),
                                              _arrays([30, 36, 42], [0, 0, 0])])
            report = _run(source, root / "output", pair_across_shards=True)
            result = _read_partition(root / "output", "train")
            np.testing.assert_array_equal(result["timestamp"], [0, 24, 36])
            self.assertEqual(sum(record["cross_shard_pair_count"] for record in report["shards"]), 1)
            self.assert_accounting(report, "train", parts["train"], result)
            self.assert_physical_provenance(result, parts["train"])

    def test_partition_boundary_is_not_borrowed_even_when_continuous(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, parts, _ = _fixture(root, [_arrays([0, 6, 12])], [_arrays([18, 24, 30])])
            report = _run(source, root / "output", pair_across_shards=True)
            for partition, expected in (("train", [0]), ("validation", [24])):
                result = _read_partition(root / "output", partition)
                np.testing.assert_array_equal(result["timestamp"], expected)
                self.assert_accounting(report, partition, parts[partition], result)
            self.assertEqual(sum(record["cross_shard_pair_count"] for record in report["shards"]), 0)

    def test_empty_shards_are_audited_without_losing_neighboring_complete_bins(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            blocks = [_arrays([]), _arrays([0, 6]), _arrays([]), _arrays([12, 18]), _arrays([])]
            source, parts, _ = _fixture(root, blocks)
            report = _run(source, root / "output", pair_across_shards=True)
            result = _read_partition(root / "output", "train")
            np.testing.assert_array_equal(result["timestamp"], [0, 12])
            empty_records = [record for record in report["shards"]
                             if record["partition"] == "train" and record["source_file_rows"] == 0]
            self.assertEqual(len(empty_records), 3)
            self.assertTrue(all(record["output_path"] is None for record in empty_records))
            self.assert_accounting(report, "train", parts["train"], result)
            self.assert_physical_provenance(result, parts["train"])

    def test_empty_intermediary_is_conservatively_not_bridged(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            blocks = [_arrays([0, 6, 12]), _arrays([]), _arrays([18, 24, 30])]
            source, parts, _ = _fixture(root, blocks)
            report = _run(source, root / "output", pair_across_shards=True)
            result = _read_partition(root / "output", "train")
            np.testing.assert_array_equal(result["timestamp"], [0, 24])
            self.assertEqual(sum(record["cross_shard_pair_count"] for record in report["shards"]), 0)
            self.assert_accounting(report, "train", parts["train"], result)
            self.assert_physical_provenance(result, parts["train"])

    def test_epoch_phase_six_discards_only_unpaired_outer_edges(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = _arrays(np.arange(1, 9) * 6)
            source, parts, _ = _fixture(root, _split(data, (2, 4)))
            report = _run(source, root / "output", pair_across_shards=True)
            result = _read_partition(root / "output", "train")
            expected, _ = coarsen_arrays(data)
            np.testing.assert_array_equal(result["timestamp"], [12, 24, 36])
            for field in POWER_FIELDS:
                np.testing.assert_array_equal(result[field], expected[field])
            records = [record for record in report["shards"] if record["partition"] == "train"]
            self.assertEqual(sum(record["discarded_input_rows"] for record in records), 2)
            self.assertEqual(sum(record["cross_shard_pair_count"] for record in records), 2)
            self.assert_accounting(report, "train", parts["train"], result)
            self.assert_physical_provenance(result, parts["train"])

    def test_overlapping_physical_source_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, _, _ = _fixture(root, [_arrays([0, 6, 12]), _arrays([12, 18, 24, 30])])
            with self.assertRaisesRegex(ValueError, "overlapping or unordered source shards"):
                _run(source, root / "output", pair_across_shards=True)
            report = json.loads((root / "output/downsampling_audit.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertFalse((root / "output/aligned_partition_manifest.json").exists())

    def test_invalid_borrowed_first_row_is_validated_before_consumption(self):
        for kind in ("nonfinite", "bad_grid", "unexpected_field"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                blocks = _split(_arrays([0, 6, 12, 18, 24, 30]), (3,))
                if kind == "nonfinite":
                    blocks[1]["appliance_w"][0] = np.nan
                elif kind == "bad_grid":
                    blocks[1]["timestamp"][0] = 19
                else:
                    blocks[1]["unknown_label"] = np.arange(3)
                source, _, _ = _fixture(root, blocks)
                with self.assertRaises(ValueError):
                    _run(source, root / "output", pair_across_shards=True)
                report = json.loads((root / "output/downsampling_audit.json").read_text())
                self.assertEqual(report["status"], "failed")
                self.assertFalse((root / "output/aligned_partition_manifest.json").exists())

    def test_borrowed_file_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            blocks = _split(_arrays([0, 6, 12, 18, 24, 30]), (3,))
            source, _, _ = _fixture(root, blocks)
            blocks[1]["appliance_w"][0] += 1
            np.savez_compressed(source / "train/source_1.npz", **blocks[1])
            with self.assertRaisesRegex(ValueError, "source hash mismatch"):
                _run(source, root / "output", pair_across_shards=True)
            report = json.loads((root / "output/downsampling_audit.json").read_text())
            self.assertEqual(report["status"], "failed")

    def test_unknown_protocol_and_factor_one_cannot_enable_boundary_reinterpretation(self):
        for protocol, factor in (("unknown", 2), ("aligned_research_partitions_v1", 1)):
            with self.subTest(protocol=protocol, factor=factor), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                source, _, _ = _fixture(root, [_arrays([0, 6])], protocol=protocol)
                with self.assertRaisesRegex(ValueError, "cross-shard pairing requires"):
                    _run(source, root / "output", factor=factor, pair_across_shards=True)
                self.assertFalse((root / "output").exists())

    def test_factor_one_without_boundary_option_preserves_every_source_power(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = _arrays([0, 6, 12, 18, 24, 30])
            source, _, _ = _fixture(root, _split(data, (1, 3)))
            report = _run(source, root / "output", factor=1)
            result = _read_partition(root / "output", "train")
            for key in ("timestamp", *POWER_FIELDS, "background_clipped_w"):
                np.testing.assert_array_equal(result[key], data[key])
            self.assertTrue(report["source_shard_boundaries_preserved"])
            self.assertEqual(sum(record["discarded_input_rows"] for record in report["shards"]), 0)


if __name__ == "__main__":
    unittest.main()
