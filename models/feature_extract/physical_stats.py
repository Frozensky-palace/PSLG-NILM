"""Deterministic physical/statistical features for a fast state pilot."""
from __future__ import annotations

import numpy as np


FEATURE_NAMES = [
    "length_samples", "mean_w", "std_w", "rms_w", "min_w", "max_w",
    "q10_w", "q25_w", "median_w", "q75_w", "q90_w", "range_w",
    "energy_sample_w", "start_w", "end_w", "end_minus_start_w",
    "mean_abs_diff_w", "std_diff_w", "max_abs_diff_w",
    "fraction_above_20w", "fraction_above_500w", "fraction_above_1500w",
] + [f"paa_{idx:02d}_w" for idx in range(16)]


def _paa(values: np.ndarray, bins: int = 16) -> np.ndarray:
    edges = np.linspace(0, len(values), bins + 1).astype(int)
    return np.array([
        values[left:right].mean() if right > left else values[min(left, len(values) - 1)]
        for left, right in zip(edges[:-1], edges[1:])
    ], dtype=np.float64)


def physical_stats(data: np.ndarray, config: dict):
    """Extract interpretable features while ignoring padded tensor positions."""
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] < 1:
        raise ValueError("physical_stats expects (samples, timesteps, channels)")
    raw_lengths = config.get("lengths")
    if raw_lengths is None:
        lengths = np.full(len(values), values.shape[1], dtype=np.int64)
    else:
        lengths = np.asarray(raw_lengths).reshape(-1).astype(np.int64)
    if len(lengths) != len(values):
        raise ValueError("lengths must align with samples")

    rows = []
    for sample, length in zip(values, lengths):
        length = int(max(1, min(length, sample.shape[0])))
        x = sample[:length, 0]
        diff = np.diff(x)
        q10, q25, q50, q75, q90 = np.quantile(x, [.10, .25, .50, .75, .90])
        rows.append([
            length, x.mean(), x.std(), np.sqrt(np.mean(x ** 2)), x.min(), x.max(),
            q10, q25, q50, q75, q90, x.max() - x.min(), x.sum(),
            x[0], x[-1], x[-1] - x[0],
            np.mean(np.abs(diff)) if len(diff) else 0.0,
            diff.std() if len(diff) else 0.0,
            np.max(np.abs(diff)) if len(diff) else 0.0,
            np.mean(x > 20), np.mean(x > 500), np.mean(x > 1500),
            *_paa(x, 16),
        ])
    features = np.asarray(rows, dtype=np.float64)
    history = {
        "model_name": "physical_stats",
        "epochs_trained": 0,
        "feature_names": FEATURE_NAMES,
        "simple_explanation": "可解释的时长、功率、能量、变化和粗略形状特征",
    }
    return features, history
