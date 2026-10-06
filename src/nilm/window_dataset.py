"""Framework-independent sharded Seq2Point window reader."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ARM_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def validate_arm(arm: str) -> str:
    """Validate an arm label: B0/B1/B2 plus D/E route labels (B3T, B4, ...).

    Labels are case-preserved (they must match the dataset manifest keys
    and the ``{arm.lower()}_*`` index files). Anything containing ``test``
    stays forbidden by protocol.
    """
    if "test" in arm.lower():
        raise ValueError("test is not an arm; test stays locked")
    if not ARM_PATTERN.fullmatch(arm):
        raise ValueError(f"unsupported arm {arm!r}")
    return arm


def valid_center_ranges(timestamps: np.ndarray, *, window_length: int,
                        sample_seconds: int, stride: int = 1,
                        excluded_intervals: list[tuple[int, int]] | None = None
                        ) -> list[tuple[int, int, int, int]]:
    """Return (first_center, stop, stride, count) for valid full windows."""
    ts = np.asarray(timestamps, dtype=np.int64)
    if window_length <= 0 or window_length % 2 != 1:
        raise ValueError("window_length must be a positive odd number")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if not len(ts):
        return []
    half = window_length // 2
    run_starts = np.r_[0, np.flatnonzero(np.diff(ts) != sample_seconds) + 1]
    run_ends = np.r_[run_starts[1:], len(ts)]
    exclusions = excluded_intervals or []
    ranges = []
    for run_start, run_end in zip(run_starts, run_ends):
        first = int(run_start + half)
        stop = int(run_end - half)
        if first >= stop:
            continue
        centers = np.arange(first, stop, stride, dtype=np.int64)
        keep = np.ones(len(centers), dtype=bool)
        if exclusions:
            left_ts = ts[centers - half]
            right_ts = ts[centers + half]
            for excluded_start, excluded_end in exclusions:
                keep &= ((right_ts < excluded_start) | (left_ts > excluded_end))
        centers = centers[keep]
        if not len(centers):
            continue
        block_start = 0
        for pos in range(1, len(centers) + 1):
            if pos == len(centers) or centers[pos] - centers[pos - 1] != stride:
                block = centers[block_start:pos]
                ranges.append((int(block[0]), int(block[-1] + stride),
                               int(stride), int(len(block))))
                block_start = pos
    return ranges


class ShardedWindowDataset:
    """Read normalized Seq2Point windows lazily from prepared NPZ shards."""

    def __init__(self, experiment_dir: str | Path, *, arm: str,
                 partition: str):
        self.experiment_dir = Path(experiment_dir)
        self.project_root = Path(__file__).resolve().parents[2]
        with open(self.experiment_dir / "dataset_manifest.json", encoding="utf-8") as stream:
            self.manifest = json.load(stream)
        with open(self.experiment_dir / "normalization.json", encoding="utf-8") as stream:
            self.normalization = json.load(stream)
        self.arm = validate_arm(arm)
        self.partition = partition
        ranges = pd.read_csv(self.experiment_dir / "window_ranges.csv")
        self.ranges = ranges[ranges["partition"] == partition].reset_index(drop=True)
        self.cumulative = np.cumsum(self.ranges["count"].to_numpy(dtype=np.int64))
        self.window_length = int(self.manifest["window_length"])
        self.half = self.window_length // 2
        self._cached_path = None
        self._cached_data = None

    def __len__(self) -> int:
        return int(self.cumulative[-1]) if len(self.cumulative) else 0

    def _source(self, shard_index: int) -> tuple[Path, str, str]:
        # Validation and test are always real-only, independently of arm.
        source_arm = self.arm if self.partition == "train" else "B0"
        entry = self.manifest["sources"][source_arm][self.partition][shard_index]
        path = Path(entry["path"])
        if not path.is_absolute():
            path = self.project_root / path
        return (path, entry["mains_field"], entry["appliance_field"])

    def _load(self, path: Path):
        if self._cached_path != path:
            if self._cached_data is not None:
                self._cached_data.close()
            self._cached_data = np.load(path)
            self._cached_path = path
        return self._cached_data

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.float32]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        range_index = int(np.searchsorted(self.cumulative, index, side="right"))
        previous = int(self.cumulative[range_index - 1]) if range_index else 0
        row = self.ranges.iloc[range_index]
        center = int(row["first_center"] + (index - previous) * row["stride"])
        path, mains_field, appliance_field = self._source(int(row["shard_index"]))
        data = self._load(path)
        mains = data[mains_field][center - self.half:center + self.half + 1]
        target = float(data[appliance_field][center])
        mains_cfg = self.normalization["mains_w"]
        app_cfg = self.normalization["appliance_w"]
        x = ((mains.astype(np.float32) - mains_cfg["mean"])
             / mains_cfg["std"]).astype(np.float32)
        y = np.float32((target - app_cfg["mean"]) / app_cfg["std"])
        return x, y

    def close(self) -> None:
        if self._cached_data is not None:
            self._cached_data.close()
            self._cached_data = None
            self._cached_path = None

    def denormalize_target(self, value):
        cfg = self.normalization["appliance_w"]
        return np.asarray(value) * cfg["std"] + cfg["mean"]
