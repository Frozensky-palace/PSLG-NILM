"""Physical, distribution, diversity and boundary checks for synthetic cycles.

Every generation route (B3/B4/B5) must pass this gate before its cycles are
placed on a background: negative power or impossible peaks are hard failures,
distribution drift and zero diversity are warnings recorded for review.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REAL_CYCLE_KEYS = ("appliance_w",)


def load_real_reference(real_library_dir: Path,
                        max_cycles: int | None = None) -> list[np.ndarray]:
    """Load real train cycle waveforms (appliance_w) as float64 arrays."""
    real_library_dir = Path(real_library_dir)
    inventory_path = real_library_dir / "real_cycle_library.csv"
    if not inventory_path.exists():
        raise FileNotFoundError(inventory_path)
    import csv

    rows = list(csv.DictReader(open(inventory_path, encoding="utf-8")))
    arrays: list[np.ndarray] = []
    for row in rows[:max_cycles] if max_cycles else rows:
        path = real_library_dir / row["path"]
        with np.load(path) as data:
            arrays.append(data["appliance_w"].astype(np.float64))
    if not arrays:
        raise ValueError(f"no real cycles loaded from {real_library_dir}")
    return arrays


def load_synthetic_cycles(synthetic_dir: Path) -> tuple[list[np.ndarray], dict]:
    """Load generator output: cycle arrays plus generation_summary.json."""
    synthetic_dir = Path(synthetic_dir)
    summary_path = synthetic_dir / "generation_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    arrays: list[np.ndarray] = []
    for record in summary["records"]:
        path = synthetic_dir / "cycles" / f"{record['synthetic_cycle_id']}.npz"
        with np.load(path) as data:
            arrays.append(data["appliance_w"].astype(np.float64))
    return arrays, summary


def resample_to_length(power: np.ndarray, length: int) -> np.ndarray:
    """Linear-interpolate a waveform to exactly ``length`` samples."""
    power = np.asarray(power, dtype=np.float64)
    if len(power) == 0:
        raise ValueError("cannot resample an empty waveform")
    if len(power) == length:
        return power.copy()
    positions = np.linspace(0.0, len(power) - 1.0, num=length)
    return np.interp(positions, np.arange(len(power)), power)


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "q05": float(np.quantile(values, 0.05)),
        "q50": float(np.quantile(values, 0.50)),
        "q95": float(np.quantile(values, 0.95)),
    }


def physical_checks(power: np.ndarray, sample_seconds: int,
                    real_durations: np.ndarray, real_energies: np.ndarray,
                    real_peaks: np.ndarray) -> dict:
    """Per-cycle hard checks; ``power`` in W, ``real_*`` reference stats."""
    power = np.asarray(power, dtype=np.float64)
    duration = len(power) * sample_seconds
    energy_wh = float(power.sum() * sample_seconds / 3600.0)
    peak = float(power.max()) if len(power) else 0.0
    negative = bool((power < 0).any())
    duration_in_bounds = bool(
        np.quantile(real_durations, 0.05) <= duration
        <= np.quantile(real_durations, 0.95))
    energy_ratio = (energy_wh / float(np.quantile(real_energies, 0.5))
                    if np.quantile(real_energies, 0.5) > 0 else float("inf"))
    return {
        "negative_power": negative,
        "duration_seconds": duration,
        "duration_in_real_q05_q95": duration_in_bounds,
        "energy_wh": energy_wh,
        "energy_ratio_vs_real_median": energy_ratio,
        "peak_w": peak,
        "peak_exceeds_real_max_x1p05": bool(peak > real_peaks.max() * 1.05),
    }


def diversity_index(powers: list[np.ndarray], length: int = 256,
                    max_cycles: int = 200) -> dict:
    """Mean pairwise distance of unit-normalized resampled waveforms."""
    if len(powers) < 2:
        return {"mean_pairwise_distance": 0.0, "identical_pairs": 0,
                "sampled_cycles": len(powers)}
    vectors = np.stack([
        resample_to_length(p, length) for p in powers[:max_cycles]])
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norm, 1e-9)
    n = len(vectors)
    distances = []
    identical = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(vectors[i] - vectors[j]))
            distances.append(d)
            if d < 1e-6:
                identical += 1
    return {
        "mean_pairwise_distance": float(np.mean(distances)),
        "min_pairwise_distance": float(np.min(distances)),
        "identical_pairs": identical,
        "sampled_cycles": n,
    }


def boundary_checks(powers: list[np.ndarray], sample_seconds: int,
                    window_samples: int = 5) -> dict:
    """Start/end power and short-window slope statistics across cycles."""
    starts, ends, slopes = [], [], []
    for power in powers:
        power = np.asarray(power, dtype=np.float64)
        if len(power) < 2 * window_samples:
            continue
        starts.append(float(power[0]))
        ends.append(float(power[-1]))
        first = power[:window_samples]
        last = power[-window_samples:]
        slopes.append(float(np.abs(np.diff(first)).mean()))
        slopes.append(float(np.abs(np.diff(last)).mean()))
    return {
        "median_start_power_w": float(np.median(starts)) if starts else 0.0,
        "median_end_power_w": float(np.median(ends)) if ends else 0.0,
        "median_boundary_abs_slope_w_per_sample": (
            float(np.median(slopes)) if slopes else 0.0),
    }


def evaluate_synthetic_dataset(synthetic_dir: Path, real_library_dir: Path,
                               sample_seconds: int = 6,
                               max_real_cycles: int | None = None
                               ) -> dict:
    """Full quality gate for one generator output directory."""
    synth, summary = load_synthetic_cycles(synthetic_dir)
    real = load_real_reference(real_library_dir, max_real_cycles)
    if not synth:
        raise ValueError("synthetic dataset contains no cycles")

    real_durations = np.array([len(p) * sample_seconds for p in real])
    real_energies = np.array([p.sum() * sample_seconds / 3600.0 for p in real])
    real_peaks = np.array([p.max() for p in real])

    per_cycle = [
        physical_checks(p, sample_seconds, real_durations, real_energies,
                        real_peaks)
        for p in synth
    ]
    negative = sum(1 for c in per_cycle if c["negative_power"])
    impossible_peak = sum(1 for c in per_cycle
                          if c["peak_exceeds_real_max_x1p05"])
    out_of_duration = sum(1 for c in per_cycle
                          if not c["duration_in_real_q05_q95"])

    synth_durations = np.array([c["duration_seconds"] for c in per_cycle])
    synth_energies = np.array([c["energy_wh"] for c in per_cycle])
    synth_peaks = np.array([c["peak_w"] for c in per_cycle])
    distribution = {
        "duration_seconds": {"real": _quantiles(real_durations),
                             "synthetic": _quantiles(synth_durations)},
        "energy_wh": {"real": _quantiles(real_energies),
                      "synthetic": _quantiles(synth_energies)},
        "peak_w": {"real": _quantiles(real_peaks),
                   "synthetic": _quantiles(synth_peaks)},
    }

    diversity = diversity_index(synth)
    boundary = boundary_checks(synth, sample_seconds)

    flags = {
        "negative_power": "PASS" if negative == 0 else "FAIL",
        "impossible_peak": "PASS" if impossible_peak == 0 else "FAIL",
        "nonempty": "PASS" if len(synth) > 0 else "FAIL",
        "duration_distribution": (
            "PASS" if out_of_duration <= len(synth) * 0.2 else "WARN"),
        "energy_scale": (
            "PASS" if 0.5 <= float(np.median(synth_energies)
                                   / max(np.quantile(real_energies, 0.5), 1e-9)
                                   ) <= 2.0 else "WARN"),
        "diversity": ("PASS" if diversity["identical_pairs"] == 0 else "WARN"),
    }
    report = {
        "protocol": "synthetic_quality_v1",
        "synthetic_dir": str(synthetic_dir),
        "generator": summary.get("generator"),
        "config_hash": summary.get("config_hash"),
        "n_real_reference": len(real),
        "n_synthetic": len(synth),
        "physical": {
            "negative_power_cycles": negative,
            "impossible_peak_cycles": impossible_peak,
            "out_of_duration_cycles": out_of_duration,
        },
        "distribution": distribution,
        "diversity": diversity,
        "boundary": boundary,
        "flags": flags,
        "passed": all(v != "FAIL" for v in flags.values()),
    }
    return report
