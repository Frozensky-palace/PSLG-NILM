"""Train-only state exchangeability and waveform example selection (A2).

All computations use a state library built from train cycles only. Nothing here
reads validation or test data.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

FEATURE_COLUMNS = (
    "duration_seconds",
    "mean_power_w",
    "std_power_w",
    "start_power_w",
    "end_power_w",
    "energy_wh",
    "mean_abs_slope_w_per_sample",
)

EXAMPLE_CATEGORIES = (
    "typical",
    "shape_typical",
    "shape_outlier",
    "shortest",
    "longest",
    "highest_power",
    "lowest_power",
)


@dataclass
class StateLibrary:
    """Loaded state library artifacts (inventory plus concatenated waveforms)."""

    tag: str
    directory: Path
    inventory: pd.DataFrame
    power_w: np.ndarray
    offsets: np.ndarray

    def waveform(self, row_index: int) -> np.ndarray:
        start = int(self.offsets[row_index])
        end = int(self.offsets[row_index + 1])
        return self.power_w[start:end]


def load_state_library(tag: str, directory: Path) -> StateLibrary:
    inventory = pd.read_csv(directory / "state_inventory.csv").sort_values(
        "state_block_id").reset_index(drop=True)
    if not (inventory["source_partition"].astype(str) == "train").all():
        raise ValueError(f"state library {tag} must contain train blocks only")
    with np.load(directory / "state_waveforms.npz") as data:
        power = data["power_w"].copy()
        offsets = data["offsets"].copy()
    if len(offsets) != len(inventory) + 1:
        raise ValueError(f"state library {tag} offsets do not match inventory")
    return StateLibrary(tag=tag, directory=directory, inventory=inventory,
                        power_w=power, offsets=offsets)


def shape_signature(power: np.ndarray, length: int = 128) -> np.ndarray:
    """Min-max normalized resampled shape; amplitude information removed."""
    values = np.asarray(power, dtype=np.float64).reshape(-1)
    grid = np.linspace(0.0, 1.0, length)
    resampled = np.interp(grid, np.linspace(0.0, 1.0, len(values)), values)
    spread = float(resampled.max() - resampled.min())
    if spread <= 0:
        return np.zeros(length, dtype=np.float64)
    return (resampled - resampled.min()) / spread


def resample_waveform(power: np.ndarray, target_length: int) -> np.ndarray:
    values = np.asarray(power, dtype=np.float64).reshape(-1)
    if target_length <= 0:
        raise ValueError("target_length must be positive")
    if len(values) == target_length:
        return values.copy()
    grid = np.linspace(0.0, 1.0, target_length)
    return np.interp(grid, np.linspace(0.0, 1.0, len(values)), values)


def _feature_matrix(inventory: pd.DataFrame) -> np.ndarray:
    columns = []
    for name in FEATURE_COLUMNS:
        values = inventory[name].to_numpy(dtype=np.float64)
        if name in ("duration_seconds", "energy_wh"):
            values = np.log(values + 1.0)
        columns.append(values)
    matrix = np.column_stack(columns)
    mean = matrix.mean(axis=0)
    std = np.maximum(matrix.std(axis=0), 1e-9)
    return (matrix - mean) / std


def _percentiles(values: np.ndarray) -> dict[str, float]:
    p5, p50, p95 = np.percentile(values, [5, 50, 95])
    return {"p5": float(p5), "median": float(p50), "p95": float(p95)}


def nearest_neighbor_same_cycle_fraction(features: np.ndarray,
                                         cycle_ids: np.ndarray) -> float:
    """Fraction of blocks whose nearest same-state neighbour shares the cycle."""
    if len(features) < 2:
        return 0.0
    distances = cdist(features, features)
    np.fill_diagonal(distances, np.inf)
    nearest = np.argmin(distances, axis=1)
    return float(np.mean(cycle_ids[nearest] == cycle_ids))


def replacement_boundary_jumps(inventory: pd.DataFrame, library: StateLibrary,
                               state_label: int, rng: np.random.Generator
                               ) -> dict[str, float]:
    """Boundary jumps when same-state donors from other cycles replace blocks.

    For each interior block of the state, a donor waveform from another cycle is
    resampled to the block length and the resulting left/right boundary steps
    are compared with the original ones.
    """
    blocks = inventory[inventory["state_label"] == state_label]
    by_cycle = {
        str(cycle_id): group.sort_values("start_sample")
        for cycle_id, group in inventory.groupby("cycle_id")
    }
    same_state = blocks.index.to_numpy()
    same_state_cycles = blocks["cycle_id"].astype(str).to_numpy()
    original_jumps: list[float] = []
    replaced_jumps: list[float] = []
    for block in blocks.itertuples(index=False):
        cycle_blocks = by_cycle[str(block.cycle_id)]
        position = int(np.flatnonzero(
            cycle_blocks["state_block_id"].to_numpy()
            == block.state_block_id)[0])
        left = cycle_blocks.iloc[position - 1] if position > 0 else None
        right = (cycle_blocks.iloc[position + 1]
                 if position + 1 < len(cycle_blocks) else None)
        if left is None and right is None:
            continue
        cross_cycle = np.flatnonzero(same_state_cycles != str(block.cycle_id))
        if not len(cross_cycle):
            continue
        donor_index = same_state[int(rng.choice(cross_cycle))]
        donor_waveform = library.waveform(donor_index)
        donor = resample_waveform(donor_waveform, int(block.samples))
        if left is not None:
            original_jumps.append(abs(float(block.start_power_w)
                                      - float(left["end_power_w"])))
            replaced_jumps.append(abs(float(donor[0])
                                      - float(left["end_power_w"])))
        if right is not None:
            original_jumps.append(abs(float(right["start_power_w"])
                                      - float(block.end_power_w)))
            replaced_jumps.append(abs(float(right["start_power_w"])
                                      - float(donor[-1])))
    original = np.asarray(original_jumps, dtype=np.float64)
    replaced = np.asarray(replaced_jumps, dtype=np.float64)
    if not len(original):
        return {"blocks_with_neighbors": 0, "original_mean_jump_w": 0.0,
                "replaced_mean_jump_w": 0.0, "replaced_over_original": 0.0}
    ratio = float(replaced.mean() / max(original.mean(), 1e-9))
    return {"blocks_with_neighbors": int(len(original)),
            "original_mean_jump_w": float(original.mean()),
            "replaced_mean_jump_w": float(replaced.mean()),
            "replaced_over_original": ratio}


def multimodality_profile(library: StateLibrary, state_label: int,
                          n_segments: int = 8, n_clusters: int = 3,
                          seed: int = 17) -> dict:
    """Detect internal multimodality by clustering segment-mean profiles."""
    blocks = library.inventory[library.inventory["state_label"] == state_label]
    if len(blocks) < n_clusters:
        return {"blocks": int(len(blocks)), "subcluster_sizes": [],
                "profile_entropy": 0.0, "between_over_within": 0.0}
    profiles = np.empty((len(blocks), n_segments), dtype=np.float64)
    for position, row_index in enumerate(blocks.index.to_numpy()):
        waveform = library.waveform(row_index)
        resampled = resample_waveform(waveform, n_segments * 8)
        profiles[position] = resampled.reshape(n_segments, -1).mean(axis=1)
    clustering = KMeans(n_clusters=n_clusters, n_init=4, random_state=seed)
    labels = clustering.fit_predict(profiles)
    sizes = np.bincount(labels, minlength=n_clusters)
    proportions = sizes[sizes > 0] / len(labels)
    entropy = float(-np.sum(proportions * np.log(proportions))
                    / np.log(len(proportions)))
    centroids = clustering.cluster_centers_
    within = float(np.mean(np.square(
        profiles - centroids[labels]).sum(axis=1)) ** 0.5)
    between = float(np.mean(np.square(
        centroids - profiles.mean(axis=0)).sum(axis=1)) ** 0.5)
    return {"blocks": int(len(blocks)),
            "subcluster_sizes": [int(size) for size in sizes],
            "profile_entropy": entropy,
            "between_over_within": between / max(within, 1e-9),
            "subcluster_profiles": centroids.tolist()}


def state_metrics(library: StateLibrary, rng_seed: int = 17) -> pd.DataFrame:
    """One row per state with distribution, distance and replacement metrics."""
    inventory = library.inventory
    features = _feature_matrix(inventory)
    cycle_ids = inventory["cycle_id"].astype(str).to_numpy()
    labels = inventory["state_label"].to_numpy()
    rng = np.random.default_rng(rng_seed)
    records = []
    for state_label in sorted(inventory["state_label"].unique()):
        member = labels == state_label
        within = float(cdist(features[member], features[member])[np.triu_indices(
            int(member.sum()), k=1)].mean()) if int(member.sum()) > 1 else 0.0
        between_parts = []
        for other_label in sorted(set(labels) - {state_label}):
            other = labels == other_label
            between_parts.append(
                cdist(features[member], features[other]).mean())
        between = float(np.mean(between_parts)) if between_parts else 0.0
        nn_same_cycle = nearest_neighbor_same_cycle_fraction(
            features[member], cycle_ids[member])
        group = inventory[member]
        duration = group["duration_seconds"].to_numpy(dtype=float)
        power = group["mean_power_w"].to_numpy(dtype=float)
        energy = group["energy_wh"].to_numpy(dtype=float)
        start = group["start_power_w"].to_numpy(dtype=float)
        end = group["end_power_w"].to_numpy(dtype=float)
        slope = group["mean_abs_slope_w_per_sample"].to_numpy(dtype=float)
        record = {
            "tag": library.tag,
            "state_label": int(state_label),
            "blocks": int(member.sum()),
            "cycles": int(group["cycle_id"].nunique()),
            "median_duration_s": float(np.median(duration)),
            "duration_p5_s": float(np.percentile(duration, 5)),
            "duration_p95_s": float(np.percentile(duration, 95)),
            "median_power_w": float(np.median(power)),
            "power_p5_w": float(np.percentile(power, 5)),
            "power_p95_w": float(np.percentile(power, 95)),
            "median_energy_wh": float(np.median(energy)),
            "median_start_w": float(np.median(start)),
            "median_end_w": float(np.median(end)),
            "median_abs_slope_w": float(np.median(slope)),
            "within_state_dist": within,
            "between_state_dist": between,
            "within_over_between": within / max(between, 1e-9),
            "nn_same_cycle_fraction": nn_same_cycle,
        }
        record.update(replacement_boundary_jumps(
            inventory, library, int(state_label), rng))
        multimodal = multimodality_profile(library, int(state_label))
        record.update({
            "multimodal_entropy": multimodal["profile_entropy"],
            "multimodal_between_over_within": multimodal["between_over_within"],
            "multimodal_subcluster_sizes": ",".join(
                str(size) for size in multimodal["subcluster_sizes"]),
        })
        records.append(record)
    return pd.DataFrame(records)


def select_example_blocks(library: StateLibrary) -> dict[int, dict[str, int]]:
    """Choose typical, edge and outlier example blocks for every state."""
    inventory = library.inventory
    features = _feature_matrix(inventory)
    selections: dict[int, dict[str, int]] = {}
    for state_label in sorted(inventory["state_label"].unique()):
        positions = np.flatnonzero(
            inventory["state_label"].to_numpy() == state_label)
        state_features = features[positions]
        group = inventory.iloc[positions]
        median_feature = np.median(state_features, axis=0)
        scale = np.maximum(state_features.std(axis=0), 1e-9)
        typical_score = np.abs(state_features - median_feature) / scale
        shapes = np.stack([
            shape_signature(library.waveform(index)) for index in positions])
        median_shape = np.median(shapes, axis=0)
        shape_distance = np.square(shapes - median_shape).mean(axis=1)
        mean_power = group["mean_power_w"].to_numpy(dtype=float)
        durations = group["duration_seconds"].to_numpy(dtype=float)
        selections[int(state_label)] = {
            "typical": int(positions[int(np.argmin(typical_score.sum(axis=1)))]),
            "shape_typical": int(positions[int(np.argmin(shape_distance))]),
            "shape_outlier": int(positions[int(np.argmax(shape_distance))]),
            "shortest": int(positions[int(np.argmin(durations))]),
            "longest": int(positions[int(np.argmax(durations))]),
            "highest_power": int(positions[int(np.argmax(mean_power))]),
            "lowest_power": int(positions[int(np.argmin(mean_power))]),
        }
    return selections


def transition_context_stats(inventory: pd.DataFrame) -> pd.DataFrame:
    """Start power conditioned on predecessor state, per state."""
    records = []
    grouped = inventory.groupby("state_label")
    for state_label, group in grouped:
        for predecessor, sub in group.groupby("previous_state_label"):
            records.append({
                "state_label": int(state_label),
                "previous_state_label": (
                    int(predecessor) if np.isfinite(predecessor) else -1),
                "blocks": int(len(sub)),
                "median_start_power_w": float(sub["start_power_w"].median()),
                "median_duration_s": float(sub["duration_seconds"].median()),
            })
    return pd.DataFrame(records)


def exchangeability_verdict(row: pd.Series, long_state_seconds: float = 1800.0
                            ) -> str:
    """Rule-based human-readable verdict, reviewed and refined in the report."""
    reasons = []
    if row["median_duration_s"] >= long_state_seconds:
        reasons.append("长状态：内部可能混合多个物理阶段")
    if row["multimodal_entropy"] >= 0.85:
        reasons.append("形状子类分布接近均匀，多模态明显")
    if row["nn_same_cycle_fraction"] >= 0.30:
        reasons.append("最近邻常来自同一 cycle，块间差异大")
    if row["replaced_over_original"] >= 1.5:
        reasons.append("跨 cycle 替换后边界跳变显著增大")
    if reasons:
        return "需要条件化或继续细分：" + "；".join(reasons)
    return "可直接交换候选：分布紧凑、替换后边界变化可控"
