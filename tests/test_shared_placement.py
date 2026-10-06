"""Tests for the shared frozen placement schedule builder."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_shared_placement_schedule import (  # noqa: E402
    main as schedule_main,
    sha256_of_file,
)


def _write_aligned_fixture(root: Path, rows: int = 3000) -> Path:
    """Two train shards: long continuous idle runs with a few busy blocks."""
    aligned = root / "aligned_partitions_v2"
    (aligned / "train").mkdir(parents=True)
    rng = np.random.default_rng(3)
    shards = []
    start_unix = 1_363_876_806
    for shard_index in range(2):
        appliance = np.zeros(rows, dtype=np.float32)
        for block_start in (400, 1600, 2800):
            appliance[block_start:block_start + 15] = rng.normal(
                500, 50, 15).clip(min=0)
        timestamps = (start_unix + np.arange(rows) * 6).astype(np.int64)
        relative = f"train/shard_{shard_index:03d}.npz"
        np.savez_compressed(aligned / relative, timestamp=timestamps,
                            appliance_w=appliance)
        shards.append({"path": relative})
        start_unix += rows * 6
    manifest = {"partitions": {"train": {"shards": shards}}}
    (aligned / "aligned_partition_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8")
    return aligned


class SharedPlacementScheduleTests(unittest.TestCase):
    def test_schedule_is_deterministic_and_overlap_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            aligned = _write_aligned_fixture(root)
            output_a = root / "sched_a"
            output_b = root / "sched_b"
            for output in (output_a, output_b):
                argv = ["build_shared_placement_schedule.py",
                        "--aligned-dir", str(aligned),
                        "--count", "5", "--samples-per-cycle", "100",
                        "--sample-seconds", "6",
                        "--guard-seconds", "30",
                        "--seed", "17", "--output-dir", str(output)]
                with patch.object(sys, "argv", argv):
                    schedule_main()
            a = (output_a / "placement_schedule.csv").read_text()
            b = (output_b / "placement_schedule.csv").read_text()
            self.assertEqual(a, b)  # same seed -> identical frozen schedule
            summary = json.loads(
                (output_a / "placement_schedule_summary.json")
                .read_text(encoding="utf-8"))
            self.assertEqual(summary["count"], 5)
            self.assertEqual(summary["envelope_overlap_rows"], 0)
            self.assertEqual(summary["schedule_sha256"],
                             sha256_of_file(
                                 output_a / "placement_schedule.csv"))
            self.assertEqual(summary["schedule_sha256"],
                             sha256_of_file(
                                 output_b / "placement_schedule.csv"))

    def test_different_seed_moves_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            aligned = _write_aligned_fixture(root)
            starts = []
            for seed in (17, 18):
                output = root / f"sched_{seed}"
                argv = ["build_shared_placement_schedule.py",
                        "--aligned-dir", str(aligned),
                        "--count", "3", "--samples-per-cycle", "50",
                        "--guard-seconds", "30",
                        "--seed", str(seed), "--output-dir", str(output)]
                with patch.object(sys, "argv", argv):
                    schedule_main()
                import pandas as pd
                starts.append(pd.read_csv(
                    output / "placement_schedule.csv")
                    ["start_global_index"].tolist())
            self.assertNotEqual(starts[0], starts[1])

    def test_impossible_count_fails_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            aligned = _write_aligned_fixture(root)
            argv = ["build_shared_placement_schedule.py",
                    "--aligned-dir", str(aligned),
                    "--count", "100000", "--samples-per-cycle", "500",
                    "--guard-seconds", "300",
                    "--seed", "17", "--output-dir", str(root / "sched")]
            with patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit):
                    schedule_main()


class PlacementEnvelopePinTests(unittest.TestCase):
    """Pinning the envelope must reproduce the shared slot schedule."""

    def _aligned_with_mains(self, root: Path, rows: int = 4000) -> Path:
        aligned = root / "aligned_partitions_v2"
        (aligned / "train").mkdir(parents=True)
        rng = np.random.default_rng(3)
        shards = []
        start_unix = 1_363_876_806
        for shard_index in range(2):
            appliance = np.zeros(rows, dtype=np.float32)
            for block_start in (400, 1600, 2800):
                appliance[block_start:block_start + 15] = rng.normal(
                    500, 50, 15).clip(min=0)
            timestamps = (start_unix + np.arange(rows) * 6).astype(np.int64)
            relative = f"train/shard_{shard_index:03d}.npz"
            np.savez_compressed(aligned / relative, timestamp=timestamps,
                                mains_w=appliance + 50.0,
                                appliance_w=appliance)
            shards.append({"path": relative})
            start_unix += rows * 6
        manifest = {"partitions": {"train": {"shards": shards}}}
        (aligned / "aligned_partition_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8")
        return aligned

    def _write_synthetic(self, root: Path, name: str,
                         lengths: list[int]) -> Path:
        directory = root / name
        (directory / "cycles").mkdir(parents=True)
        records = []
        for index, length in enumerate(lengths):
            cycle_id = f"synthetic_{index:04d}"
            np.savez_compressed(
                directory / "cycles" / f"{cycle_id}.npz",
                appliance_w=np.full(length, 300.0, dtype=np.float32))
            records.append({"synthetic_cycle_id": cycle_id})
        (directory / "generation_summary.json").write_text(
            json.dumps({"records": records}), encoding="utf-8")
        return directory

    def _place(self, place_main, aligned: Path, routes: list[tuple[str, Path]],
               output: Path, envelope: int | None = None) -> None:
        argv = ["place_synthetics_on_background.py",
                "--aligned-dir", str(aligned),
                "--output-dir", str(output),
                "--seed", "17", "--guard-seconds", "30"]
        for label, path in routes:
            argv += ["--synthetic-dir", f"{label}={path}"]
        if envelope is not None:
            argv += ["--envelope-samples", str(envelope)]
        with patch.object(sys, "argv", argv):
            place_main()

    def test_pinned_envelope_reproduces_shared_slots(self) -> None:
        from scripts.place_synthetics_on_background import (
            main as place_main,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            aligned = self._aligned_with_mains(root)
            short = self._write_synthetic(root, "short", [80, 60, 90])
            long = self._write_synthetic(root, "long", [150, 70, 85])
            # Reference: short placed together with long (computed envelope).
            self._place(place_main, aligned, [("SHORT", short),
                                              ("LONG", long)],
                        root / "together")
            # Test: short placed alone with the envelope pinned to 150.
            self._place(place_main, aligned, [("SHORT", short)],
                        root / "pinned", envelope=150)
            together = json.loads(
                (root / "together" / "placement_summary.json")
                .read_text(encoding="utf-8"))
            pinned = json.loads(
                (root / "pinned" / "placement_summary.json")
                .read_text(encoding="utf-8"))
            self.assertEqual(
                [p["start_global_index"]
                 for p in together["arms"]["SHORT"]["placements"]],
                [p["start_global_index"]
                 for p in pinned["arms"]["SHORT"]["placements"]])
            self.assertEqual(pinned["envelope_samples"], 150)
            self.assertTrue(pinned["envelope_pinned"])
            self.assertFalse(together["envelope_pinned"])

    def test_pinned_envelope_below_longest_cycle_fails(self) -> None:
        from scripts.place_synthetics_on_background import (
            main as place_main,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            aligned = self._aligned_with_mains(root)
            long = self._write_synthetic(root, "long", [150, 70, 85])
            with self.assertRaises(SystemExit):
                self._place(place_main, aligned, [("LONG", long)],
                            root / "bad", envelope=100)


if __name__ == "__main__":
    unittest.main()
