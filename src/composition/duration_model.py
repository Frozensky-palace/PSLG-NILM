"""Empirical per-state duration model for the HSMM (roadmap F1).

Durations are stored as the observed per-state samples from the frozen
train state library and drawn by empirical resampling — no parametric
family is assumed, so the frozen model reproduces exactly what the train
data exhibited (including the wide low-power state).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class StateDurationModel:
    """Empirical duration distributions per state label."""

    def __init__(self, durations_by_state: dict[int, list[int]]):
        cleaned = {}
        for state, durations in durations_by_state.items():
            values = sorted(int(d) for d in durations if int(d) > 0)
            if values:
                cleaned[int(state)] = values
        if not cleaned:
            raise ValueError("duration model needs at least one state "
                             "with positive observed durations")
        self.durations_by_state = cleaned

    @classmethod
    def fit(cls, observations: list[tuple[int, int]]) -> "StateDurationModel":
        """``observations``: (state_label, duration_seconds) pairs."""
        durations_by_state: dict[int, list[int]] = {}
        for state, duration in observations:
            durations_by_state.setdefault(int(state), []).append(
                int(duration))
        return cls(durations_by_state)

    @property
    def states(self) -> list[int]:
        return sorted(self.durations_by_state)

    def sample_duration(self, state: int, rng: np.random.Generator) -> int:
        if state not in self.durations_by_state:
            raise KeyError(f"no observed durations for state {state}")
        values = self.durations_by_state[state]
        return int(values[int(rng.integers(len(values)))])

    def median(self, state: int) -> int:
        values = self.durations_by_state[state]
        return int(values[len(values) // 2])

    def to_dict(self) -> dict:
        return {"durations_by_state": {
            str(k): v for k, v in sorted(self.durations_by_state.items())}}

    @classmethod
    def from_dict(cls, data: dict) -> "StateDurationModel":
        return cls({int(k): list(v) for k, v
                    in data["durations_by_state"].items()})

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1),
                              encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "StateDurationModel":
        return cls.from_dict(json.loads(Path(path).read_text(
            encoding="utf-8")))
