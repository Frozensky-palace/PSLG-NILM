"""Abstract generator interface shared by B3/B4/B5 routes (Phase B2).

A generator produces waveform arrays plus schema-complete records; the shared
``generate_dataset`` helper persists cycles, records and a summary manifest so
that no route can skip provenance fields.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from src.generation.provenance import (
    canonical_config_hash,
    sha256_of_bytes,
    utc_now_string,
)
from src.generation.schema import (
    SyntheticCycleRecord,
    StateSegmentRecord,
)


class BaseGenerator(ABC):
    """One frozen instance per run; subclasses implement sampling."""

    name: str = "base"
    version: str = "0"

    @abstractmethod
    def generate_cycle(self, synthetic_cycle_id: str, seed: int,
                       rng: np.random.Generator
                       ) -> tuple[np.ndarray, SyntheticCycleRecord]:
        """Return (power waveform in W, schema-valid record)."""

    def generate_dataset(self, output_dir: Path, count: int, seed: int,
                         sample_seconds: int = 6) -> list[SyntheticCycleRecord]:
        """Sample ``count`` cycles, persist waveforms plus records."""
        output_dir = Path(output_dir)
        (output_dir / "cycles").mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed)
        records = []
        for index in range(count):
            synthetic_cycle_id = f"synthetic_{index:04d}"
            power, record = self.generate_cycle(synthetic_cycle_id, seed, rng)
            power = np.asarray(power, dtype=np.float32)
            if (power < 0).any():
                raise RuntimeError(
                    f"{synthetic_cycle_id}: negative power is not allowed")
            record.waveform_sha256 = sha256_of_bytes(power.tobytes())
            record.created_utc = utc_now_string()
            record.sample_seconds = sample_seconds
            errors = record.validate()
            if errors:
                raise RuntimeError(
                    f"{synthetic_cycle_id}: schema violations: {errors}")
            np.savez_compressed(
                output_dir / "cycles" / f"{synthetic_cycle_id}.npz",
                appliance_w=power,
                relative_time_s=np.arange(len(power), dtype=np.int64)
                * sample_seconds,
            )
            records.append(record)
        summary = {
            "protocol": "synthetic_dataset_v1",
            "generator": self.name,
            "generator_version": self.version,
            "config_hash": canonical_config_hash(self.config()),
            "count": count,
            "seed": seed,
            "sample_seconds": sample_seconds,
            "records": [record.to_dict() for record in records],
        }
        (output_dir / "generation_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8")
        return records

    def config(self) -> dict:
        """Serializable generator configuration for hashing."""
        return {"name": self.name, "version": self.version}


class ConstantGenerator(BaseGenerator):
    """Minimal deterministic generator used in tests and pipeline smoke runs."""

    name = "constant"
    version = "1"

    def __init__(self, length_samples: int = 100, power_w: float = 500.0):
        if length_samples <= 0:
            raise ValueError("length_samples must be positive")
        self.length_samples = int(length_samples)
        self.power_w = float(power_w)

    def generate_cycle(self, synthetic_cycle_id: str, seed: int,
                       rng: np.random.Generator
                       ) -> tuple[np.ndarray, SyntheticCycleRecord]:
        power = np.full(self.length_samples, self.power_w, dtype=np.float32)
        segment = StateSegmentRecord(
            state_label=0,
            target_duration_seconds=self.length_samples * 6,
            actual_duration_seconds=self.length_samples * 6,
            target_samples=self.length_samples,
            actual_samples=self.length_samples,
            mean_power_w=float(power.mean()),
            energy_wh=float(power.astype(np.float64).sum() * 6 / 3600.0),
        )
        record = SyntheticCycleRecord(
            synthetic_cycle_id=synthetic_cycle_id,
            route="B3",
            seed=seed,
            generator_name=self.name,
            generator_version=self.version,
            checkpoint_sha256=None,
            conditions={"power_w": self.power_w},
            state_path=[0],
            segments=[segment],
            boundary_treatment="none",
            sample_seconds=6,
            created_utc="",
            waveform_sha256="",
        )
        return power, record
