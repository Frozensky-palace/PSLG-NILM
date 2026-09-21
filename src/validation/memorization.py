"""Nearest-neighbor replication audit for synthetic cycles (guide §5.3).

A synthetic cycle counts as *replicated* when its shape is closer to some
real train cycle than the real-vs-real baseline allows by chance. The audit
never modifies data; it only reports evidence for later review.
"""
from __future__ import annotations

import hashlib

import numpy as np

from src.validation.synthetic_quality import resample_to_length


def unit_vector(power: np.ndarray, length: int = 256) -> np.ndarray:
    """Resample a waveform and L2-normalize it, focusing on shape not scale."""
    vector = resample_to_length(np.asarray(power, dtype=np.float64), length)
    norm = np.linalg.norm(vector)
    if norm < 1e-9:
        return vector
    return vector / norm


def waveform_digest(power: np.ndarray) -> str:
    """SHA-256 of raw bytes; catches bit-exact duplicates of real cycles."""
    return hashlib.sha256(
        np.asarray(power, dtype=np.float32).tobytes()).hexdigest()


def nearest_distances(targets: np.ndarray, references: np.ndarray,
                      chunk_size: int = 64) -> np.ndarray:
    """For each target row, the smallest Euclidean distance to references.

    Targets are processed in chunks so peak memory stays at
    ``chunk_size * len(references) * dim`` instead of the full cross product.
    """
    targets = np.asarray(targets, dtype=np.float64)
    references = np.asarray(references, dtype=np.float64)
    if len(references) == 0 or len(targets) == 0:
        return np.full(len(targets), np.inf)
    result = np.empty(len(targets), dtype=np.float64)
    for start in range(0, len(targets), chunk_size):
        chunk = targets[start:start + chunk_size]
        diff = chunk[:, None, :] - references[None, :, :]
        result[start:start + len(chunk)] = np.sqrt(
            (diff ** 2).sum(axis=2)).min(axis=1)
    return result


def real_baseline(real_vectors: np.ndarray, baseline_count: int,
                  seed: int) -> np.ndarray:
    """Real-vs-real nearest distances (each point excludes itself)."""
    real_vectors = np.asarray(real_vectors)
    if len(real_vectors) < 2:
        return np.array([])
    rng = np.random.default_rng(seed)
    count = min(baseline_count, len(real_vectors))
    picked = rng.choice(len(real_vectors), size=count, replace=False)
    sample = real_vectors[picked]
    diff = sample[:, None, :] - sample[None, :, :]
    distances = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(distances, np.inf)
    return distances.min(axis=1)


def audit_memorization(synthetic_powers: list[np.ndarray],
                       real_powers: list[np.ndarray], *, length: int = 256,
                       baseline_count: int = 200,
                       threshold_percentile: float = 1.0,
                       seed: int = 0) -> dict:
    """Full replication audit: distances, threshold, rate, exact copies."""
    if not synthetic_powers or not real_powers:
        raise ValueError("both synthetic and real waveforms are required")
    real_vectors = np.stack([unit_vector(p, length) for p in real_powers])
    synth_vectors = np.stack([unit_vector(p, length) for p in synthetic_powers])

    baseline = real_baseline(real_vectors, baseline_count, seed)
    if len(baseline) == 0:
        threshold = float("inf")
    else:
        threshold = float(np.quantile(baseline, threshold_percentile / 100.0))

    distances = nearest_distances(synth_vectors, real_vectors)
    replicated = distances < threshold
    real_digests = {waveform_digest(p) for p in real_powers}
    synth_digests = [waveform_digest(p) for p in synthetic_powers]
    exact = [digest for digest in synth_digests if digest in real_digests]

    return {
        "protocol": "memorization_audit_v1",
        "n_synthetic": len(synthetic_powers),
        "n_real_reference": len(real_powers),
        "threshold_percentile": threshold_percentile,
        "threshold_distance": threshold,
        "distance_min": float(distances.min()),
        "distance_median": float(np.median(distances)),
        "distance_max": float(distances.max()),
        "replicated_count": int(replicated.sum()),
        "replication_rate": float(replicated.mean()),
        "exact_duplicate_count": len(exact),
        "passed": len(exact) == 0,
    }
