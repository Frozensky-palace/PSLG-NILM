"""Ablatable boundary handling between consecutive primitives (roadmap F2).

The Phase A freeze (docs/state_and_b2_policy_freeze_v1.md) proved generic
cross-fade and endpoint-offset *harmful*, so ``none`` is the frozen
default. Endpoint *selection* — preferring a candidate whose start power
matches the previous segment's end power — remains an allowed ablation
and is implemented here as a pure selection rule (never waveform surgery).
"""
from __future__ import annotations

import numpy as np

BOUNDARY_MODES = ("none", "endpoint_match")


class BoundaryHandler:
    """Selection-only boundary policy; ``none`` is the frozen default."""

    def __init__(self, mode: str = "none"):
        if mode not in BOUNDARY_MODES:
            raise ValueError(f"boundary mode must be one of {BOUNDARY_MODES}")
        self.mode = mode

    def select(self, candidates: list[tuple[int, np.ndarray]],
               previous_end_w: float | None,
               rng: np.random.Generator) -> int:
        """Pick one candidate index given the previous segment's end power.

        ``candidates``: (index, waveform) pairs; waveform[0] is the start
        power. ``endpoint_match`` minimises |start - previous_end|; ties
        and the ``none`` mode fall back to the seeded rng.
        """
        if not candidates:
            raise ValueError("candidate list is empty")
        if self.mode == "none" or previous_end_w is None:
            return candidates[int(rng.integers(len(candidates)))][0]
        best_index = None
        best_gap = None
        for index, wave in candidates:
            gap = abs(float(np.asarray(wave, dtype=np.float64)[0])
                      - float(previous_end_w))
            if best_gap is None or gap < best_gap:
                best_index, best_gap = index, gap
        return int(best_index)
