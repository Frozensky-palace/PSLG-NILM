"""Deterministic placement of paired synthetic events into idle data runs."""
from __future__ import annotations

import numpy as np


def idle_runs(timestamps: np.ndarray, appliance_w: np.ndarray, *,
              sample_seconds: int = 6, idle_threshold_w: float = 20.0
              ) -> list[tuple[int, int]]:
    """Return [start, end) index runs that are continuous and appliance-idle."""
    timestamps = np.asarray(timestamps, dtype=np.int64)
    power = np.asarray(appliance_w, dtype=np.float64)
    if len(timestamps) != len(power):
        raise ValueError("timestamps and appliance power must align")
    if not len(timestamps):
        return []
    eligible = power <= idle_threshold_w
    breaks = np.ones(len(timestamps), dtype=bool)
    breaks[1:] = (
        (~eligible[1:]) | (~eligible[:-1])
        | (np.diff(timestamps) != sample_seconds)
    )
    starts = np.flatnonzero(eligible & breaks)
    stops = []
    for start in starts:
        stop = int(start) + 1
        while (stop < len(timestamps) and eligible[stop]
               and timestamps[stop] - timestamps[stop - 1] == sample_seconds):
            stop += 1
        stops.append(stop)
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)]


def schedule_lengths(runs: list[tuple[int, int]], lengths: np.ndarray, *,
                     total_rows: int, guard_samples: int,
                     seed: int) -> np.ndarray:
    """Place variable-length events without overlap, longest first."""
    event_lengths = np.asarray(lengths, dtype=np.int64)
    if np.any(event_lengths <= 0):
        raise ValueError("event lengths must be positive")
    occupancy = np.zeros(total_rows, dtype=bool)
    starts = np.full(len(event_lengths), -1, dtype=np.int64)
    rng = np.random.default_rng(seed)
    order = np.argsort(-event_lengths, kind="stable")
    for event_index in order:
        length = int(event_lengths[event_index])
        candidates = []
        capacities = []
        for run_start, run_end in runs:
            capacity = run_end - run_start - length - 2 * guard_samples + 1
            if capacity > 0:
                candidates.append((run_start, run_end))
                capacities.append(capacity)
        if not candidates:
            raise RuntimeError(f"no idle run can hold event of {length} samples")
        probabilities = np.asarray(capacities, dtype=float)
        probabilities /= probabilities.sum()
        placed = False
        for _ in range(20000):
            choice = int(rng.choice(len(candidates), p=probabilities))
            run_start, run_end = candidates[choice]
            low = run_start + guard_samples
            high = run_end - guard_samples - length
            position = int(rng.integers(low, high + 1))
            left = position - guard_samples
            right = position + length + guard_samples
            if not occupancy[left:right].any():
                occupancy[left:right] = True
                starts[event_index] = position
                placed = True
                break
        if not placed:
            raise RuntimeError(f"could not place event {event_index} without overlap")
    return starts
