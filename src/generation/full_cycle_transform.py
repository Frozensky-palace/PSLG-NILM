"""B3-T: real-cycle transform generator (roadmap D1).

The cheapest full-cycle baseline: draw a real *train* cycle, apply a
restricted time-axis and power-axis scaling, and enforce the physical
guards. It trains nothing, so it also serves as the reference for asking
whether neural generators add value at all.

Guards (roadmap D1): power never negative; duration, energy and peak must
stay inside the train quantile envelope. Violations trigger a bounded
number of deterministic re-draws, never silent acceptance.
"""
from __future__ import annotations

import numpy as np

from src.generation.base_generator import BaseGenerator
from src.generation.schema import StateSegmentRecord, SyntheticCycleRecord
from src.validation.synthetic_quality import (
    load_real_reference,
    resample_to_length,
)

MAX_ATTEMPTS = 64


class RealCycleTransformGenerator(BaseGenerator):
    """One frozen instance per run; donors come only from the train library."""

    name = "b3_transform"
    version = "1"

    def __init__(self, real_library_dir, *, sample_seconds: int = 6,
                 time_scale_range: tuple[float, float] = (0.85, 1.2),
                 power_scale_range: tuple[float, float] = (0.9, 1.1)):
        self.real_library_dir = str(real_library_dir)
        self.sample_seconds = int(sample_seconds)
        self.time_scale_range = tuple(float(v) for v in time_scale_range)
        self.power_scale_range = tuple(float(v) for v in power_scale_range)
        if not (0 < self.time_scale_range[0] <= self.time_scale_range[1]):
            raise ValueError("invalid time_scale_range")
        if not (0 < self.power_scale_range[0] <= self.power_scale_range[1]):
            raise ValueError("invalid power_scale_range")

        self.donors = load_real_reference(real_library_dir)
        self.donor_ids = [f"cycle_{index:04d}" for index in range(
            len(self.donors))]
        self.donor_durations = np.array(
            [len(p) * self.sample_seconds for p in self.donors])
        self.donor_energies = np.array([
            float(p.sum() * self.sample_seconds / 3600.0)
            for p in self.donors])
        self.donor_peaks = np.array([float(p.max()) for p in self.donors])
        # Guard envelope: the transform must keep outputs inside the train
        # quantile range (roadmap D1). q01/q99 keeps it permissive but real.
        self.duration_bounds = (
            float(np.quantile(self.donor_durations, 0.01)),
            float(np.quantile(self.donor_durations, 0.99)))
        self.energy_bounds = (
            float(np.quantile(self.donor_energies, 0.01)),
            float(np.quantile(self.donor_energies, 0.99)))
        self.peak_limit = float(self.donor_peaks.max())

    def config(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "real_library_dir": self.real_library_dir,
            "time_scale_range": list(self.time_scale_range),
            "power_scale_range": list(self.power_scale_range),
            "n_donors": len(self.donors),
        }

    def generate_cycle(self, synthetic_cycle_id: str, seed: int,
                       rng: np.random.Generator
                       ) -> tuple[np.ndarray, SyntheticCycleRecord]:
        last_error = "no attempt made"
        for _ in range(MAX_ATTEMPTS):
            donor_index = int(rng.integers(len(self.donors)))
            donor = self.donors[donor_index]
            time_scale = float(rng.uniform(*self.time_scale_range))
            power_scale = float(rng.uniform(*self.power_scale_range))
            target_samples = max(2, int(round(len(donor) * time_scale)))
            power = resample_to_length(donor, target_samples) * power_scale
            duration = len(power) * self.sample_seconds
            energy = float(power.sum() * self.sample_seconds / 3600.0)
            peak = float(power.max())
            if duration < self.duration_bounds[0] \
                    or duration > self.duration_bounds[1]:
                last_error = f"duration {duration}s outside guard envelope"
                continue
            if energy < self.energy_bounds[0] or energy > self.energy_bounds[1]:
                last_error = f"energy {energy:.1f}Wh outside guard envelope"
                continue
            if peak > self.peak_limit:
                last_error = f"peak {peak:.1f}W exceeds train maximum"
                continue
            if (power < 0).any():
                last_error = "negative power produced"
                continue
            segment = StateSegmentRecord(
                state_label=0,
                target_duration_seconds=int(len(donor) * self.sample_seconds),
                actual_duration_seconds=duration,
                target_samples=len(donor),
                actual_samples=len(power),
                mean_power_w=float(power.mean()),
                energy_wh=energy,
            )
            record = SyntheticCycleRecord(
                synthetic_cycle_id=synthetic_cycle_id,
                route="B3",
                seed=seed,
                generator_name=self.name,
                generator_version=self.version,
                checkpoint_sha256=None,
                conditions={
                    "donor_cycle_id": self.donor_ids[donor_index],
                    "time_scale": time_scale,
                    "power_scale": power_scale,
                },
                state_path=[0],
                segments=[segment],
                boundary_treatment="none",
                sample_seconds=self.sample_seconds,
                created_utc="",
                waveform_sha256="",
            )
            return power, record
        raise RuntimeError(
            f"{synthetic_cycle_id}: transform guard failed after "
            f"{MAX_ATTEMPTS} attempts; last error: {last_error}")
