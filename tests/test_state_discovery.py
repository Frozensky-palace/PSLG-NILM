"""Tests for the formal state discovery entry helpers (Phase C2)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_state_discovery import (  # noqa: E402
    build_smoke_segments,
    load_segment_map,
)


def _write_segments(directory: Path, partitions=("train", "train", "train")):
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, partition in enumerate(partitions):
        filename = f"cycle_{index:04d}.csv"
        pd.DataFrame({
            "timestamp": [1, 2],
            "power": [1.0, 2.0],
            "datetime": ["a", "b"],
            "cycle_id": [f"c{index}"] * 2,
        }).to_csv(directory / filename, index=False)
        rows.append({
            "csv_idx": index,
            "filename": filename,
            "cycle_id": f"c{index}",
            "partition": partition,
            "source_npz": f"cycles/cycle_{index:04d}.npz",
            "start_unix": 0,
            "end_unix": 10,
            "samples": 2,
        })
    map_path = directory / "segment_source_map.csv"
    pd.DataFrame(rows).to_csv(map_path, index=False)
    return map_path


class StateDiscoveryHelperTests(unittest.TestCase):
    def test_train_only_guard_accepts_train(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            map_path = _write_segments(root / "segments")
            mapping, resolved = load_segment_map(root / "segments", None)
            self.assertEqual(resolved, map_path)
            self.assertEqual(len(mapping), 3)

    def test_train_only_guard_rejects_mixed_partitions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_segments(root / "segments",
                            partitions=("train", "validation", "train"))
            with self.assertRaises(SystemExit):
                load_segment_map(root / "segments", None)

    def test_missing_map_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(SystemExit):
                load_segment_map(Path(temporary), None)

    def test_smoke_subset_copies_first_n_and_map(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            segments = root / "segments"
            _write_segments(segments)
            mapping, _ = load_segment_map(segments, None)
            subset_map = build_smoke_segments(
                segments, root / "subset", 2, mapping)
            subset = pd.read_csv(subset_map)
            self.assertEqual(len(subset), 2)
            self.assertEqual(sorted(subset["csv_idx"].tolist()), [0, 1])
            for index in range(2):
                self.assertTrue((root / "subset" / f"cycle_{index:04d}.csv"
                                 ).exists())
            self.assertFalse((root / "subset" / "cycle_0002.csv").exists())

    def test_smoke_subset_rejects_oversized_request(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            segments = root / "segments"
            _write_segments(segments)
            mapping, _ = load_segment_map(segments, None)
            with self.assertRaises(SystemExit):
                build_smoke_segments(segments, root / "subset", 10, mapping)


if __name__ == "__main__":
    unittest.main()
