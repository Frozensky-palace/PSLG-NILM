"""HSMM path sampler: state order (Markov) + per-state durations (F1).

Unlike the B4 basic composer — which samples a path length then borrows
donor durations — this sampler draws each state's duration from its
empirical train distribution, making "how long does this state last" an
explicit modeled constraint. The path probability is recorded so every
composed cycle carries its own provenance.
"""
from __future__ import annotations

import math

import numpy as np

from src.composition.duration_model import StateDurationModel
from src.composition.transition_model import MarkovChain

MAX_PATH_STATES = 12


class HSMMPathSampler:
    """Order + duration joint sampler; the two models stay inspectable."""

    def __init__(self, markov: MarkovChain, durations: StateDurationModel,
                 path_lengths: dict[int, int]):
        self.markov = markov
        self.durations = durations
        self.path_lengths = {int(k): int(v)
                             for k, v in path_lengths.items()}
        if not self.path_lengths:
            raise ValueError("path-length distribution is required")

    def sample_length(self, rng: np.random.Generator) -> int:
        lengths = sorted(self.path_lengths)
        weights = np.array([self.path_lengths[n] for n in lengths],
                           dtype=np.float64)
        n = int(rng.choice(lengths, p=weights / weights.sum()))
        return min(max(n, 1), MAX_PATH_STATES)

    def sample_sequence(self, rng: np.random.Generator,
                        n_states: int | None = None,
                        max_attempts: int = 16) -> list[tuple[int, int]]:
        """Return [(state, duration_seconds), ...] of the target length.

        Individual draws stop early when the chain reaches a state with no
        observed successor (inventing transitions is forbidden). The whole
        path is therefore re-drawn up to ``max_attempts`` times and the
        longest draw wins — no transition outside train is ever created.
        """
        n_states = n_states if n_states is not None \
            else self.sample_length(rng)
        best: list[tuple[int, int]] = []
        for _ in range(max_attempts):
            sequence = self._draw_once(rng, n_states)
            if len(sequence) > len(best):
                best = sequence
            if len(sequence) >= n_states:
                break
        return best

    def _draw_once(self, rng: np.random.Generator,
                   n_states: int) -> list[tuple[int, int]]:
        sequence: list[tuple[int, int]] = []
        state = self.markov.sample_initial(rng)
        while True:
            if state not in self.durations.states:
                break
            duration = self.durations.sample_duration(state, rng)
            sequence.append((state, duration))
            if len(sequence) >= n_states:
                break
            nxt = self.markov.sample_next(state, rng)
            if nxt is None:
                break
            state = nxt
        return sequence

    def log_probability(self, sequence: list[tuple[int, int]]) -> float:
        """Joint log-probability of states (Markov) and durations (empirical)."""
        path = [state for state, _ in sequence]
        total = self.markov.log_probability(path)
        if not math.isfinite(total):
            return -math.inf
        for state, duration in sequence:
            values = self.durations.durations_by_state.get(state)
            if not values or duration not in values:
                return -math.inf
            total += math.log(1.0 / len(values))
        return total
