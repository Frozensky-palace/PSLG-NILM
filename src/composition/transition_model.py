"""First-order Markov state-order model fitted from train sequences (F1).

Sampling only ever follows transitions observed in the train data: states
without outgoing evidence terminate the path instead of inventing moves.
Laplace smoothing is applied *within observed outgoing options only*, and
the chosen alpha is frozen into the model file.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


class MarkovChain:
    """Initial-state distribution plus smoothed transition distribution."""

    def __init__(self, initial_counts: dict[int, int],
                 transition_counts: dict[tuple[int, int], int],
                 smoothing_alpha: float):
        self.initial_counts = {int(k): int(v)
                               for k, v in initial_counts.items()}
        self.transition_counts = {
            (int(a), int(b)): int(v)
            for (a, b), v in transition_counts.items()}
        self.smoothing_alpha = float(smoothing_alpha)

    @classmethod
    def fit(cls, sequences: list[list[int]],
            smoothing_alpha: float = 1.0) -> "MarkovChain":
        if not sequences:
            raise ValueError("no state sequences to fit")
        initial_counts: dict[int, int] = {}
        transition_counts: dict[tuple[int, int], int] = {}
        for sequence in sequences:
            if not sequence:
                continue
            initial_counts[sequence[0]] = \
                initial_counts.get(sequence[0], 0) + 1
            for current, nxt in zip(sequence, sequence[1:]):
                transition_counts[(current, nxt)] = \
                    transition_counts.get((current, nxt), 0) + 1
        return cls(initial_counts, transition_counts, smoothing_alpha)

    @property
    def states(self) -> list[int]:
        states = set(self.initial_counts)
        for (a, b) in self.transition_counts:
            states.update((a, b))
        return sorted(states)

    def initial_distribution(self) -> tuple[list[int], np.ndarray]:
        states = sorted(self.initial_counts)
        weights = np.array([self.initial_counts[s] for s in states],
                           dtype=np.float64) + self.smoothing_alpha
        return states, weights / weights.sum()

    def outgoing(self, state: int) -> dict[int, float]:
        options = {nxt: count for (frm, nxt), count
                   in self.transition_counts.items() if frm == state}
        if not options:
            return {}
        keys = sorted(options)
        weights = np.array([options[k] for k in keys],
                           dtype=np.float64) + self.smoothing_alpha
        return dict(zip(keys, weights / weights.sum()))

    def sample_initial(self, rng: np.random.Generator) -> int:
        states, weights = self.initial_distribution()
        return int(rng.choice(states, p=weights))

    def sample_next(self, state: int, rng: np.random.Generator
                    ) -> int | None:
        distribution = self.outgoing(state)
        if not distribution:
            return None
        keys = sorted(distribution)
        return int(rng.choice(keys, p=[distribution[k] for k in keys]))

    def sample_path(self, rng: np.random.Generator,
                    n_states: int) -> list[int]:
        """Draw exactly ``n_states`` states; stops early when a state has
        no observed outgoing transition (invention is forbidden)."""
        path = [self.sample_initial(rng)]
        while len(path) < n_states:
            nxt = self.sample_next(path[-1], rng)
            if nxt is None:
                break
            path.append(nxt)
        return path

    def log_probability(self, path: list[int]) -> float:
        if not path:
            return -math.inf
        states, weights = self.initial_distribution()
        if path[0] not in states:
            return -math.inf
        total = math.log(dict(zip(states, weights))[path[0]])
        for current, nxt in zip(path, path[1:]):
            distribution = self.outgoing(current)
            if nxt not in distribution:
                return -math.inf
            total += math.log(distribution[nxt])
        return total

    def to_dict(self) -> dict:
        return {
            "initial_counts": {str(k): v for k, v
                               in self.initial_counts.items()},
            "transition_counts": [[f"{a}>{b}", v] for (a, b), v
                                  in sorted(self.transition_counts.items())],
            "smoothing_alpha": self.smoothing_alpha,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MarkovChain":
        transition_counts = {
            tuple(key.split(">")): value
            for key, value in data["transition_counts"]
        }
        transition_counts = {(int(a), int(b)): int(v)
                             for (a, b), v in transition_counts.items()}
        initial = {int(k): int(v)
                   for k, v in data["initial_counts"].items()}
        return cls(initial, transition_counts,
                   data.get("smoothing_alpha", 1.0))

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1),
                              encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "MarkovChain":
        return cls.from_dict(json.loads(Path(path).read_text(
            encoding="utf-8")))
