"""B5 constrained composer: HSMM path + durations + endpoint selection (F3).

The B4 basic composer samples a path length and borrows donor durations;
this composer makes both explicit (HSMM) and adds the ablatable boundary
rule. Ablation ladder from roadmap F3 is driven by construction flags:

- B4-basic equivalent:  use_hsmm_durations=False, boundary="none"
- B4 + Markov:          use_hsmm_durations=False
- B4 + HSMM:            defaults
- B4 + HSMM + endpoint: boundary="endpoint_match"
"""
from __future__ import annotations

import numpy as np

from src.composition.boundary_handler import BoundaryHandler
from src.composition.hsmm_sequence import HSMMPathSampler
from src.generation.base_generator import BaseGenerator
from src.generation.schema import StateSegmentRecord, SyntheticCycleRecord
from src.validation.synthetic_quality import resample_to_length


class ConstrainedComposer(BaseGenerator):
    """Compose cycles from a labelled primitive pool under HSMM control."""

    name = "b5_hsmm_compose"
    version = "1"

    def __init__(self, hsmm: HSMMPathSampler,
                 pool_waves: list[np.ndarray], pool_labels: list[int],
                 *, sample_seconds: int = 6, boundary_mode: str = "none",
                 use_hsmm_durations: bool = True):
        if len(pool_waves) != len(pool_labels):
            raise ValueError("pool waves/labels length mismatch")
        self.hsmm = hsmm
        self.pool_waves = [np.asarray(w, dtype=np.float64)
                           for w in pool_waves]
        self.pool_labels = list(pool_labels)
        self.boundary = BoundaryHandler(boundary_mode)
        self.use_hsmm_durations = bool(use_hsmm_durations)
        self.sample_seconds = int(sample_seconds)
        self._by_state: dict[int, list[tuple[int, np.ndarray]]] = {}
        for index, wave in enumerate(self.pool_waves):
            self._by_state.setdefault(self.pool_labels[index], []).append(
                (index, wave))

    def config(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "boundary_mode": self.boundary.mode,
            "use_hsmm_durations": self.use_hsmm_durations,
        }

    def generate_cycle(self, synthetic_cycle_id: str, seed: int,
                       rng: np.random.Generator
                       ) -> tuple[np.ndarray, SyntheticCycleRecord]:
        sequence = self.hsmm.sample_sequence(rng)
        if not sequence:
            raise RuntimeError(f"{synthetic_cycle_id}: empty HSMM sequence")

        parts: list[np.ndarray] = []
        segments: list[StateSegmentRecord] = []
        boundary_costs: list[float] = []
        previous_end: float | None = None
        for state, target_seconds in sequence:
            candidates = self._by_state.get(state)
            if not candidates:
                raise RuntimeError(
                    f"{synthetic_cycle_id}: empty primitive pool for state "
                    f"{state}")
            index = self.boundary.select(candidates, previous_end, rng)
            donor_wave = next(wave for i, wave in candidates
                              if i == index)
            target_samples = max(2, int(round(
                target_seconds / self.sample_seconds)))
            if self.use_hsmm_durations:
                primitive = resample_to_length(donor_wave, target_samples)
            else:
                primitive = donor_wave.copy()
            if previous_end is not None:
                boundary_costs.append(
                    abs(float(primitive[0]) - float(previous_end)))
            parts.append(primitive)
            segments.append(StateSegmentRecord(
                state_label=state,
                target_duration_seconds=target_seconds,
                actual_duration_seconds=len(primitive) * self.sample_seconds,
                target_samples=target_samples,
                actual_samples=len(primitive),
                mean_power_w=float(primitive.mean()),
                energy_wh=float(primitive.sum() * self.sample_seconds
                                / 3600.0),
            ))
            previous_end = float(primitive[-1])
        power = np.concatenate(parts)
        record = SyntheticCycleRecord(
            synthetic_cycle_id=synthetic_cycle_id,
            route="B5",
            seed=seed,
            generator_name=self.name,
            generator_version=self.version,
            checkpoint_sha256=None,
            conditions={
                "boundary_mode": self.boundary.mode,
                "use_hsmm_durations": self.use_hsmm_durations,
                "path_log_probability": float(
                    self.hsmm.log_probability(sequence)),
                "mean_boundary_cost_w": (
                    float(np.mean(boundary_costs))
                    if boundary_costs else 0.0),
            },
            state_path=[state for state, _ in sequence],
            segments=segments,
            # Waveform-level truth: endpoint_match only *selects* which
            # primitive joins next; it never modifies samples, so the
            # record keeps the schema value "none" and the mode lives in
            # conditions where the ablation is audited.
            boundary_treatment="none",
            sample_seconds=self.sample_seconds,
            created_utc="",
            waveform_sha256="",
        )
        return power, record
