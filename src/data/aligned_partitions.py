"""Build leakage-free, aligned aggregate/appliance/background partitions."""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


PARTITIONS = ("train", "validation", "test")


def partition_intervals(inventory: pd.DataFrame) -> dict[str, tuple[int, int]]:
    """Return non-overlapping [start, end) intervals containing assigned cycles.

    The first cycle of the next partition is the current partition's exclusive
    boundary.  This keeps all idle/background samples between adjacent cycles
    on only one side of the chronological split.
    """
    required = {"partition", "start_unix", "end_unix"}
    missing = required.difference(inventory.columns)
    if missing:
        raise ValueError(f"inventory missing required columns: {sorted(missing)}")

    assigned = inventory[inventory["partition"].isin(PARTITIONS)].copy()
    starts: dict[str, int] = {}
    ends: dict[str, int] = {}
    for name in PARTITIONS:
        part = assigned[assigned["partition"] == name]
        if part.empty:
            raise ValueError(f"partition {name!r} has no cycles")
        starts[name] = int(part["start_unix"].min())
        ends[name] = int(part["end_unix"].max()) + 1

    intervals = {
        "train": (starts["train"], starts["validation"]),
        "validation": (starts["validation"], starts["test"]),
        "test": (starts["test"], ends["test"]),
    }
    for name, (start, end) in intervals.items():
        if start >= end:
            raise ValueError(f"invalid interval for {name}: [{start}, {end})")
    return intervals


def align_to_grid(value: int, sample_seconds: int, *, direction: str) -> int:
    """Align a Unix timestamp to the fixed epoch-based sampling grid."""
    if sample_seconds <= 0:
        raise ValueError("sample_seconds must be positive")
    if direction == "ceil":
        return int(math.ceil(value / sample_seconds) * sample_seconds)
    if direction == "floor":
        return int(math.floor(value / sample_seconds) * sample_seconds)
    raise ValueError("direction must be 'ceil' or 'floor'")


def fill_short_gaps(series: pd.Series, max_gap_samples: int) -> pd.Series:
    """Time-interpolate only complete missing runs up to the requested size."""
    if max_gap_samples <= 0 or not series.isna().any():
        return series
    missing = series.isna().to_numpy()
    preserve_nan = np.zeros(len(series), dtype=bool)
    pos = 0
    while pos < len(series):
        if not missing[pos]:
            pos += 1
            continue
        stop = pos + 1
        while stop < len(series) and missing[stop]:
            stop += 1
        if pos == 0 or stop == len(series) or stop - pos > max_gap_samples:
            preserve_nan[pos:stop] = True
        pos = stop
    result = series.interpolate(method="time", limit_area="inside")
    result.iloc[preserve_nan] = np.nan
    return result


def aligned_power_frame(branch: pd.Series, mains: pd.Series, *, start_unix: int,
                        end_unix: int, sample_seconds: int = 6,
                        max_interpolate_seconds: int = 150) -> tuple[pd.DataFrame, dict]:
    """Align two power series and calculate signed and clipped background.

    Only rows where both measurements are available after short-gap
    interpolation are returned.  Long gaps are omitted and represented by a
    change in ``segment_id``.
    """
    if end_unix <= start_unix:
        raise ValueError("end_unix must be greater than start_unix")
    grid_start = align_to_grid(start_unix, sample_seconds, direction="ceil")
    grid_end = align_to_grid(end_unix, sample_seconds, direction="ceil")
    timestamps = np.arange(grid_start, grid_end, sample_seconds, dtype=np.int64)
    grid = pd.to_datetime(timestamps, unit="s", utc=True)
    rule = f"{sample_seconds}s"

    def prepare(series: pd.Series) -> pd.Series:
        out = series.astype(np.float64).sort_index()
        if out.index.tz is None:
            out.index = out.index.tz_localize("UTC")
        else:
            out.index = out.index.tz_convert("UTC")
        out = out[~out.index.duplicated(keep="last")]
        # Interpolate while the caller-provided padding is still present, then
        # crop to the target grid.  Cropping first would turn a shard boundary
        # into an artificial edge and can lose one otherwise recoverable row.
        out = out.resample(rule, origin="epoch").mean()
        out = fill_short_gaps(out, max_interpolate_seconds // sample_seconds)
        return out.reindex(grid)

    branch_grid = prepare(branch)
    mains_grid = prepare(mains)
    valid = branch_grid.notna() & mains_grid.notna()
    valid_timestamps = timestamps[valid.to_numpy()]
    appliance = branch_grid[valid].to_numpy(dtype=np.float64)
    aggregate = mains_grid[valid].to_numpy(dtype=np.float64)
    signed = aggregate - appliance
    clipped = np.maximum(signed, 0.0)

    gaps = np.diff(valid_timestamps, prepend=valid_timestamps[:1])
    segment_id = np.cumsum(gaps > sample_seconds, dtype=np.int64)
    frame = pd.DataFrame({
        "timestamp": valid_timestamps,
        "mains_w": aggregate.astype(np.float32),
        "appliance_w": appliance.astype(np.float32),
        "background_signed_w": signed.astype(np.float32),
        "background_clipped_w": clipped.astype(np.float32),
        "segment_id": segment_id,
    })
    negative = signed < 0
    identity_error = np.abs(aggregate - (appliance + signed))
    quality = {
        "grid_start_unix": int(grid_start),
        "grid_end_unix_exclusive": int(grid_end),
        "grid_rows": int(len(timestamps)),
        "valid_rows": int(valid.sum()),
        "missing_rows": int((~valid).sum()),
        "valid_fraction": float(valid.mean()) if len(valid) else 0.0,
        "continuous_segments": int(segment_id[-1] + 1) if len(segment_id) else 0,
        "negative_background_rows": int(negative.sum()),
        "negative_background_fraction": float(negative.mean()) if len(negative) else 0.0,
        "negative_background_energy_wh": float(
            -signed[negative].sum() * sample_seconds / 3600.0),
        "max_signed_identity_error_w": float(identity_error.max()) if len(identity_error) else 0.0,
    }
    return frame, quality


def save_aligned_npz(path: str | Path, frame: pd.DataFrame, *,
                     compressed: bool = True) -> None:
    """Atomically save a shard without coercing timestamps to floating point."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {column: frame[column].to_numpy() for column in frame.columns}
    temporary = path.with_name(path.name + ".tmp.npz")
    if temporary.exists():
        temporary.unlink()
    if compressed:
        np.savez_compressed(temporary, **arrays)
    else:
        np.savez(temporary, **arrays)
    os.replace(temporary, path)


def cycle_grid_counts(cycles: pd.DataFrame, timestamps: np.ndarray, *,
                      sample_seconds: int = 6) -> dict[str, int]:
    """Count aligned valid samples falling inside every cycle."""
    counts: dict[str, int] = {}
    for row in cycles.itertuples(index=False):
        left = int(np.searchsorted(timestamps, int(row.start_unix), side="left"))
        right = int(np.searchsorted(timestamps, int(row.end_unix), side="right"))
        counts[str(row.cycle_id)] = right - left
    return counts
