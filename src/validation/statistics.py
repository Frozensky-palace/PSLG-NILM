"""Result statistics for the B0-B5 comparison (Phase B4).

Provides the frozen analysis primitives: per-cycle error decomposition,
paired bootstrap confidence intervals over identical evaluation windows,
multi-seed aggregation and an index-equality fairness audit so that groups can
only be compared when they scored the very same windows.
"""
from __future__ import annotations

import numpy as np

from src.nilm.metrics import nilm_metrics


def per_cycle_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                      cycle_spans: list[tuple[int, int]],
                      threshold_w: float = 20.0) -> list[dict]:
    """Split one flat prediction series into cycles and score each span.

    ``cycle_spans`` are (start, end_exclusive) positions into the series; a
    span marked as background may be omitted entirely.
    """
    results = []
    for position, (start, end) in enumerate(cycle_spans):
        true = np.asarray(y_true[start:end], dtype=np.float64)
        pred = np.asarray(y_pred[start:end], dtype=np.float64)
        metrics = nilm_metrics(true, pred, threshold_w=threshold_w)
        metrics["cycle_position"] = position
        metrics["samples"] = int(end - start)
        results.append(metrics)
    return results


def paired_bootstrap_ci(metric_a: np.ndarray, metric_b: np.ndarray,
                        statistic=np.mean, n_bootstraps: int = 10_000,
                        alpha: float = 0.05, seed: int = 17
                        ) -> dict[str, float]:
    """CI for statistic(A) - statistic(B) over paired per-cycle values."""
    a = np.asarray(metric_a, dtype=np.float64).reshape(-1)
    b = np.asarray(metric_b, dtype=np.float64).reshape(-1)
    if len(a) != len(b) or not len(a):
        raise ValueError("paired samples must be non-empty and aligned")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be inside (0, 1)")
    rng = np.random.default_rng(seed)
    deltas = statistic(a) - statistic(b)
    boots = np.empty(n_bootstraps, dtype=np.float64)
    for index in range(n_bootstraps):
        pick = rng.integers(0, len(a), size=len(a))
        boots[index] = statistic(a[pick]) - statistic(b[pick])
    lower, upper = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "statistic": float(deltas),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "alpha": float(alpha),
        "n_bootstraps": int(n_bootstraps),
        "n_pairs": int(len(a)),
    }


def multi_seed_summary(values: list[float]) -> dict[str, float]:
    """Mean, sample std and 95% t-interval for one metric across seeds."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(array):
        raise ValueError("at least one seed value is required")
    mean = float(array.mean())
    if len(array) < 2:
        return {"mean": mean, "std": 0.0, "ci95_half_width": 0.0,
                "n_seeds": int(len(array))}
    std = float(array.std(ddof=1))
    half_width = float(1.96 * std / np.sqrt(len(array)))
    return {"mean": mean, "std": std, "ci95_half_width": half_width,
            "n_seeds": int(len(array))}


def audit_identical_indices(reference: np.ndarray, *groups: np.ndarray
                            ) -> dict[str, object]:
    """Check that every group scored the exact same evaluation windows."""
    reference = np.asarray(reference)
    mismatches = []
    for position, group in enumerate(groups):
        group = np.asarray(group)
        if group.shape != reference.shape or not np.array_equal(
                group, reference):
            mismatches.append(position)
    return {
        "identical": not mismatches,
        "n_groups": len(groups),
        "mismatched_group_positions": mismatches,
    }
