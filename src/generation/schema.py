"""Unified record schema for every synthetic cycle (Phase B2, doc §B2).

Every generated cycle must persist the fields required by the roadmap: route
(B3/B4/B5), seed, checkpoint hash, conditions, state sequence, per-state target
and realized durations, boundary treatment, timestamps, waveform hash and
quality-check flags. Nothing is written without passing schema validation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

GENERATION_ROUTES = ("B3", "B4", "B5")
BOUNDARY_TREATMENTS = ("none", "linear_crossfade", "endpoint_offset",
                       "generated_transition")


@dataclass
class StateSegmentRecord:
    """One state inside a generated cycle."""

    state_label: int
    target_duration_seconds: float
    actual_duration_seconds: float
    target_samples: int
    actual_samples: int
    mean_power_w: float
    energy_wh: float
    donor_state_block_id: int | None = None
    donor_cycle_id: str | None = None


@dataclass
class SyntheticCycleRecord:
    """Provenance-complete record for one synthetic cycle."""

    synthetic_cycle_id: str
    route: str
    seed: int
    generator_name: str
    generator_version: str
    checkpoint_sha256: str | None
    conditions: dict[str, Any]
    state_path: list[int]
    segments: list[StateSegmentRecord]
    boundary_treatment: str
    sample_seconds: int
    created_utc: str
    waveform_sha256: str
    template_cycle_id: str | None = None
    passed_physical_checks: bool = False
    passed_memorization_checks: bool = False
    quality_notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "SyntheticCycleRecord":
        data = dict(payload)
        segment_payloads = data.pop("segments", [])
        record = cls(
            segments=[StateSegmentRecord(**segment)
                      for segment in segment_payloads],
            **data)
        return record

    def validate(self) -> list[str]:
        """Return a list of schema violations; empty means valid."""
        errors = []
        if not self.synthetic_cycle_id:
            errors.append("synthetic_cycle_id must be non-empty")
        if self.route not in GENERATION_ROUTES:
            errors.append(f"route must be one of {GENERATION_ROUTES}")
        if self.boundary_treatment not in BOUNDARY_TREATMENTS:
            errors.append(
                f"boundary_treatment must be one of {BOUNDARY_TREATMENTS}")
        if self.sample_seconds <= 0:
            errors.append("sample_seconds must be positive")
        if not self.segments:
            errors.append("at least one state segment is required")
        for position, segment in enumerate(self.segments):
            if segment.actual_samples <= 0:
                errors.append(f"segment {position} has non-positive samples")
            expected = segment.actual_samples * self.sample_seconds
            if abs(expected - segment.actual_duration_seconds) > 1e-6:
                errors.append(
                    f"segment {position} duration disagrees with sample "
                    f"count ({segment.actual_duration_seconds} s vs "
                    f"{expected} s)")
        return errors
