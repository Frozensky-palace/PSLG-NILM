"""Build a lightweight inventory of appliance work cycles.

The full UK-DALE appliance CSV is hundreds of megabytes.  This module scans it
in chunks and keeps only active points in memory, so creating the inventory is
safe on a normal workstation.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def parse_time(value: str | float | int | None) -> float | None:
    """Convert an ISO date/time or Unix timestamp to UTC seconds."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return float(ts.timestamp())


def iso_utc(seconds: float) -> str:
    return datetime.fromtimestamp(float(seconds), tz=timezone.utc).isoformat()


@dataclass
class _OpenCycle:
    start: float
    last_active: float
    first_row: int
    last_row: int
    active_samples: int
    power_sum: float
    power_max: float


class CycleDetector:
    """Streaming detector matching the repository's threshold grouping rule."""

    def __init__(self, *, threshold_w: float, max_inactive_seconds: float,
                 min_duration_seconds: float, sample_seconds: float,
                 dataset: str, building: int, appliance: str,
                 valid_start: float | None = None,
                 valid_end: float | None = None):
        self.threshold_w = float(threshold_w)
        self.max_inactive_seconds = float(max_inactive_seconds)
        self.min_duration_seconds = float(min_duration_seconds)
        self.sample_seconds = float(sample_seconds)
        self.dataset = str(dataset)
        self.building = int(building)
        self.appliance = str(appliance)
        self.valid_start = valid_start
        self.valid_end = valid_end
        self.current: _OpenCycle | None = None
        self.records: list[dict] = []
        self.invalid_rows = 0
        self._last_timestamp: float | None = None

    def feed(self, timestamps: np.ndarray, powers: np.ndarray,
             row_offset: int = 0) -> None:
        timestamps = np.asarray(timestamps, dtype=np.float64)
        powers = np.asarray(powers, dtype=np.float64)
        if timestamps.shape != powers.shape:
            raise ValueError("timestamps and powers must have the same shape")

        finite = np.isfinite(timestamps) & np.isfinite(powers)
        self.invalid_rows += int((~finite).sum())
        active_local = np.flatnonzero(finite & (powers >= self.threshold_w))
        for local_idx in active_local:
            ts = float(timestamps[local_idx])
            power = float(powers[local_idx])
            source_row = int(row_offset + local_idx)
            if self._last_timestamp is not None and ts < self._last_timestamp:
                raise ValueError("input timestamps must be sorted in ascending order")
            self._last_timestamp = ts

            if self.current is None:
                self.current = _OpenCycle(
                    start=ts, last_active=ts,
                    first_row=source_row, last_row=source_row,
                    active_samples=1, power_sum=power, power_max=power)
                continue

            if ts - self.current.last_active <= self.max_inactive_seconds:
                self.current.last_active = ts
                self.current.last_row = source_row
                self.current.active_samples += 1
                self.current.power_sum += power
                self.current.power_max = max(self.current.power_max, power)
            else:
                self._finish_current()
                self.current = _OpenCycle(
                    start=ts, last_active=ts,
                    first_row=source_row, last_row=source_row,
                    active_samples=1, power_sum=power, power_max=power)

    def finish(self) -> list[dict]:
        self._finish_current()
        return list(self.records)

    def _finish_current(self) -> None:
        c = self.current
        if c is None:
            return
        duration = float(c.last_active - c.start)
        meets_duration = duration >= self.min_duration_seconds
        has_mains_coverage = (
            (self.valid_start is None or c.start >= self.valid_start)
            and (self.valid_end is None or c.last_active <= self.valid_end)
        )
        eligible = bool(meets_duration and has_mains_coverage)
        reasons: list[str] = []
        if not meets_duration:
            reasons.append("shorter_than_min_duration")
        if not has_mains_coverage:
            reasons.append("outside_mains_coverage")

        safe_appliance = "".join(
            ch if ch.isalnum() else "_" for ch in self.appliance.lower()).strip("_")
        cycle_id = (
            f"{self.dataset.lower()}_b{self.building}_{safe_appliance}_"
            f"{int(c.start)}_{int(c.last_active)}"
        )
        self.records.append({
            "cycle_id": cycle_id,
            "dataset": self.dataset,
            "building": self.building,
            "appliance": self.appliance,
            "start_unix": int(c.start),
            "end_unix": int(c.last_active),
            "start_utc": iso_utc(c.start),
            "end_utc": iso_utc(c.last_active),
            "duration_seconds": duration,
            "active_sample_count": int(c.active_samples),
            "mean_active_power_w": float(c.power_sum / c.active_samples),
            "max_power_w": float(c.power_max),
            "active_energy_wh_approx": float(
                c.power_sum * self.sample_seconds / 3600.0),
            "source_row_start": int(c.first_row),
            "source_row_end": int(c.last_row),
            "meets_min_duration": bool(meets_duration),
            "has_mains_coverage": bool(has_mains_coverage),
            "eligible": eligible,
            "exclusion_reason": ";".join(reasons),
        })
        self.current = None


def scan_cycle_csv(path: str | Path, detector: CycleDetector,
                   *, chunk_rows: int = 1_000_000) -> tuple[pd.DataFrame, dict]:
    """Scan ``timestamp,power`` CSV and return inventory plus scan metadata."""
    path = Path(path)
    rows = 0
    for chunk_no, chunk in enumerate(pd.read_csv(
            path, usecols=["timestamp", "power"], chunksize=chunk_rows), start=1):
        detector.feed(chunk["timestamp"].to_numpy(), chunk["power"].to_numpy(), rows)
        rows += len(chunk)
        print(f"[cycle-inventory] chunk={chunk_no} rows={rows:,}", flush=True)
    frame = pd.DataFrame(detector.finish())
    if not frame.empty:
        frame = frame.sort_values(["start_unix", "cycle_id"]).reset_index(drop=True)
    metadata = {
        "source_csv": str(path.resolve()),
        "rows_scanned": int(rows),
        "invalid_rows": int(detector.invalid_rows),
        "candidate_groups": int(len(frame)),
        "eligible_cycles": int(frame["eligible"].sum()) if not frame.empty else 0,
        "threshold_w": detector.threshold_w,
        "max_inactive_seconds": detector.max_inactive_seconds,
        "min_duration_seconds": detector.min_duration_seconds,
        "sample_seconds": detector.sample_seconds,
        "valid_start_unix": detector.valid_start,
        "valid_end_unix": detector.valid_end,
    }
    return frame, metadata


def summarize_inventory(frame: pd.DataFrame) -> dict:
    eligible = frame[frame["eligible"]] if not frame.empty else frame
    result = {
        "all_candidates": int(len(frame)),
        "eligible_cycles": int(len(eligible)),
        "excluded_cycles": int(len(frame) - len(eligible)),
        "exclusion_reasons": (
            frame.loc[~frame["eligible"], "exclusion_reason"]
            .value_counts(dropna=False).to_dict() if not frame.empty else {}),
    }
    if not eligible.empty:
        result["eligible_time_range"] = {
            "start_utc": str(eligible.iloc[0]["start_utc"]),
            "end_utc": str(eligible.iloc[-1]["end_utc"]),
        }
        for column in ("duration_seconds", "active_sample_count",
                       "mean_active_power_w", "max_power_w",
                       "active_energy_wh_approx"):
            series = eligible[column].astype(float)
            result[column] = {
                "min": float(series.min()),
                "median": float(series.median()),
                "mean": float(series.mean()),
                "max": float(series.max()),
            }
    return result

