"""Utilities for fair B1/B2 real-cycle composition pilots."""
from __future__ import annotations

import numpy as np


def resample_waveform(values: np.ndarray, target_length: int) -> np.ndarray:
    """Linearly resample one state waveform to an exact target length."""
    source = np.asarray(values, dtype=np.float64).reshape(-1)
    if target_length <= 0 or len(source) == 0:
        raise ValueError("source and target length must be positive")
    if len(source) == target_length:
        return source.astype(np.float32, copy=True)
    if len(source) == 1:
        return np.full(target_length, source[0], dtype=np.float32)
    old_grid = np.linspace(0.0, 1.0, len(source))
    new_grid = np.linspace(0.0, 1.0, target_length)
    return np.interp(new_grid, old_grid, source).astype(np.float32)


def boundary_jumps(values: np.ndarray, boundaries: list[int]) -> np.ndarray:
    """Absolute power changes at internal, exclusive segment boundaries."""
    waveform = np.asarray(values, dtype=np.float64).reshape(-1)
    valid = [int(pos) for pos in boundaries if 0 < int(pos) < len(waveform)]
    return np.asarray([
        abs(float(waveform[pos]) - float(waveform[pos - 1])) for pos in valid
    ], dtype=np.float64)


def choose_donor_indices(candidate_indices: np.ndarray, candidate_cycle_ids: np.ndarray,
                         template_cycle_id: str, rng: np.random.Generator) -> int:
    """Choose a same-state donor, preferring another source cycle."""
    indices = np.asarray(candidate_indices, dtype=np.int64)
    cycle_ids = np.asarray(candidate_cycle_ids).astype(str)
    if len(indices) == 0 or len(indices) != len(cycle_ids):
        raise ValueError("donor candidates must be non-empty and aligned")
    cross_cycle = indices[cycle_ids != str(template_cycle_id)]
    pool = cross_cycle if len(cross_cycle) else indices
    return int(rng.choice(pool))


def matched_donor_scores(*, target_samples: int, target_mean_power_w: float,
                         target_start_power_w: float, target_end_power_w: float,
                         candidate_samples: np.ndarray,
                         candidate_mean_power_w: np.ndarray,
                         candidate_start_power_w: np.ndarray,
                         candidate_end_power_w: np.ndarray,
                         power_scale_w: float = 500.0,
                         mean_weight: float = 0.25,
                         endpoint_weight: float = 0.25) -> np.ndarray:
    """Score donor compatibility; lower means less resampling and boundary change."""
    samples = np.asarray(candidate_samples, dtype=np.float64).reshape(-1)
    mean_power = np.asarray(candidate_mean_power_w, dtype=np.float64).reshape(-1)
    start_power = np.asarray(candidate_start_power_w, dtype=np.float64).reshape(-1)
    end_power = np.asarray(candidate_end_power_w, dtype=np.float64).reshape(-1)
    lengths = {len(samples), len(mean_power), len(start_power), len(end_power)}
    if lengths != {len(samples)} or not len(samples):
        raise ValueError("candidate feature arrays must be non-empty and aligned")
    if target_samples <= 0 or np.any(samples <= 0):
        raise ValueError("target and candidate sample counts must be positive")
    if power_scale_w <= 0:
        raise ValueError("power_scale_w must be positive")
    if mean_weight < 0 or endpoint_weight < 0:
        raise ValueError("matching weights must be non-negative")
    duration_cost = np.abs(np.log(samples / float(target_samples)))
    mean_cost = np.abs(mean_power - float(target_mean_power_w)) / power_scale_w
    start_cost = np.abs(start_power - float(target_start_power_w)) / power_scale_w
    end_cost = np.abs(end_power - float(target_end_power_w)) / power_scale_w
    return (duration_cost + mean_weight * mean_cost
            + endpoint_weight * (start_cost + end_cost))


def choose_matched_donor(candidate_indices: np.ndarray,
                         candidate_cycle_ids: np.ndarray, *,
                         template_cycle_id: str,
                         target_samples: int,
                         target_mean_power_w: float,
                         target_start_power_w: float,
                         target_end_power_w: float,
                         candidate_samples: np.ndarray,
                         candidate_mean_power_w: np.ndarray,
                         candidate_start_power_w: np.ndarray,
                         candidate_end_power_w: np.ndarray,
                         rng: np.random.Generator,
                         top_k: int = 5,
                         power_scale_w: float = 500.0,
                         mean_weight: float = 0.25,
                         endpoint_weight: float = 0.25) -> tuple[int, float]:
    """Choose a compatible same-state donor from another training cycle."""
    indices = np.asarray(candidate_indices, dtype=np.int64).reshape(-1)
    cycle_ids = np.asarray(candidate_cycle_ids).astype(str).reshape(-1)
    feature_arrays = [
        np.asarray(candidate_samples).reshape(-1),
        np.asarray(candidate_mean_power_w).reshape(-1),
        np.asarray(candidate_start_power_w).reshape(-1),
        np.asarray(candidate_end_power_w).reshape(-1),
    ]
    if not len(indices) or any(len(values) != len(indices) for values in
                               [cycle_ids, *feature_arrays]):
        raise ValueError("donor candidates and features must be non-empty and aligned")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    cross_cycle = cycle_ids != str(template_cycle_id)
    if not np.any(cross_cycle):
        raise ValueError("matched policy requires a donor from another cycle")
    filtered_indices = indices[cross_cycle]
    filtered_features = [values[cross_cycle] for values in feature_arrays]
    scores = matched_donor_scores(
        target_samples=target_samples,
        target_mean_power_w=target_mean_power_w,
        target_start_power_w=target_start_power_w,
        target_end_power_w=target_end_power_w,
        candidate_samples=filtered_features[0],
        candidate_mean_power_w=filtered_features[1],
        candidate_start_power_w=filtered_features[2],
        candidate_end_power_w=filtered_features[3],
        power_scale_w=power_scale_w,
        mean_weight=mean_weight,
        endpoint_weight=endpoint_weight,
    )
    top_count = min(top_k, len(filtered_indices))
    top_positions = np.argsort(scores, kind="stable")[:top_count]
    chosen_position = int(rng.choice(top_positions))
    return int(filtered_indices[chosen_position]), float(scores[chosen_position])


def filter_candidates_by_length_ratio(
        candidate_indices: np.ndarray, candidate_samples: np.ndarray,
        target_samples: int, ratio_min: float, ratio_max: float
) -> tuple[np.ndarray, np.ndarray]:
    """Keep candidates whose resample ratio target/donor lies in [min, max]."""
    indices = np.asarray(candidate_indices, dtype=np.int64).reshape(-1)
    samples = np.asarray(candidate_samples, dtype=np.float64).reshape(-1)
    if len(indices) == 0 or len(indices) != len(samples):
        raise ValueError("candidate indices and samples must be non-empty and aligned")
    if any(value <= 0 for value in samples) or target_samples <= 0:
        raise ValueError("sample counts must be positive")
    if ratio_min <= 0 or ratio_max < ratio_min:
        raise ValueError("ratio limits must satisfy 0 < ratio_min <= ratio_max")
    ratios = target_samples / samples
    mask = (ratios >= ratio_min) & (ratios <= ratio_max)
    return indices[mask], ratios[mask]


def correct_block_endpoints(resampled_donor: np.ndarray,
                            template_start_power_w: float,
                            template_end_power_w: float,
                            max_power_w: float = 4096.0
                            ) -> tuple[np.ndarray, float]:
    """Shift a resampled donor with a linear ramp so endpoints match the template.

    The correction adds a linear delta from (template_start - donor_start) to
    (template_end - donor_end), clips at zero and never exceeds ``max_power_w``.
    Returns the corrected waveform and the sample-sum change caused by the
    correction (convert to Wh with the sample period downstream).
    """
    donor = np.asarray(resampled_donor, dtype=np.float64).reshape(-1)
    if not len(donor):
        raise ValueError("donor waveform must be non-empty")
    delta_start = float(template_start_power_w) - float(donor[0])
    delta_end = float(template_end_power_w) - float(donor[-1])
    ramp = np.linspace(delta_start, delta_end, len(donor))
    corrected = donor + ramp
    np.clip(corrected, 0.0, float(max_power_w), out=corrected)
    energy_delta = float(corrected.sum() - donor.sum())
    return corrected.astype(np.float32), energy_delta


def apply_linear_crossfade(waveform: np.ndarray, boundaries: list[int],
                           window_samples: int) -> tuple[np.ndarray, float]:
    """Blend every internal boundary with a short linear ramp, in place length.

    The window spans ``window_samples`` centred on each boundary; samples inside
    the window are replaced by a linear ramp between the anchors just outside
    the window, so the total length is unchanged and values stay non-negative.
    Returns the new waveform and the sample-sum change (energy audit).
    """
    values = np.asarray(waveform, dtype=np.float64).reshape(-1)
    if window_samples < 2:
        raise ValueError("crossfade window must contain at least two samples")
    if any(value < 0 for value in values):
        raise ValueError("crossfade expects non-negative power values")
    result = values.copy()
    half = window_samples // 2
    for position in boundaries:
        boundary = int(position)
        if not 0 < boundary < len(result):
            raise ValueError("boundaries must be interior sample positions")
        left = max(0, boundary - half)
        right = min(len(result), boundary + half)
        if right - left < 2:
            continue
        anchor_left = float(result[left - 1]) if left > 0 else float(result[0])
        anchor_right = (float(result[right])
                        if right < len(result) else float(result[-1]))
        result[left:right] = np.linspace(anchor_left, anchor_right,
                                         right - left)
    energy_delta = float(result.sum() - values.sum())
    return result.astype(np.float32), energy_delta
