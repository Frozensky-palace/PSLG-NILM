"""B4: shared conditional primitive CVAE plus basic composition (roadmap E1/E2).

Segments come from the frozen state library v1. The same CVAE backbone as
B3-V is conditioned on the state label and the segment's normalized stats;
composition follows the empirical train path model (first-state counts,
transition counts, per-cycle state counts) — deliberately *basic*, so that
B5's constraints have something measurable to improve upon.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.generation.base_generator import BaseGenerator
from src.generation.full_cycle_cvae import (
    BUCKET_MULTIPLE,
    LengthBucketizer,
    condition_vector,
)
from src.generation.schema import StateSegmentRecord, SyntheticCycleRecord

MAX_PATH_STATES = 12


def load_state_segments(state_library_dir: Path
                        ) -> tuple[list[np.ndarray], list[int], dict]:
    """Return (segment waveforms, state labels, inventory) from a library."""
    state_library_dir = Path(state_library_dir)
    rows = list(csv.DictReader(
        open(state_library_dir / "state_inventory.csv", encoding="utf-8")))
    with np.load(state_library_dir / "state_waveforms.npz") as data:
        power = data["power_w"]
        offsets = data["offsets"]
    waves: list[np.ndarray] = []
    labels: list[int] = []
    for row in rows:
        start = int(row["waveform_offset_start"])
        end = int(row["waveform_offset_end"])
        waves.append(power[start:end].astype(np.float64))
        labels.append(int(row["state_label"]))
    return waves, labels, {"rows": len(rows)}


def fit_path_model(inventory_rows: list[dict]) -> dict:
    """Empirical first-state counts, transitions and path lengths (train)."""
    first_state: dict[int, int] = {}
    transitions: dict[tuple[int, int], int] = {}
    path_lengths: dict[int, int] = {}
    per_cycle: dict[str, list[tuple[int, str]]] = {}
    for row in inventory_rows:
        cycle_id = row["cycle_id"]
        per_cycle.setdefault(cycle_id, []).append(
            (int(row["start_sample"]), row["state_block_id"]))
    labels_by_block = {
        row["state_block_id"]: int(row["state_label"])
        for row in inventory_rows
    }
    for cycle_id, blocks in per_cycle.items():
        blocks.sort()
        labels = [labels_by_block[block_id] for _, block_id in blocks]
        if not labels:
            continue
        first_state[labels[0]] = first_state.get(labels[0], 0) + 1
        path_lengths[len(labels)] = path_lengths.get(len(labels), 0) + 1
        for current, nxt in zip(labels, labels[1:]):
            key = (current, nxt)
            transitions[key] = transitions.get(key, 0) + 1
    return {"first_state": first_state, "transitions": transitions,
            "path_lengths": path_lengths}


def sample_path(path_model: dict, n_states: int, rng: np.random.Generator
                ) -> list[int]:
    """Draw one state path: first state from counts, then transitions."""
    first_state = path_model["first_state"]
    transitions = path_model["transitions"]
    states = sorted(first_state)
    weights = np.array([first_state[s] for s in states], dtype=np.float64)
    path = [int(rng.choice(states, p=weights / weights.sum()))]
    while len(path) < n_states:
        current = path[-1]
        options = sorted(next_state for (frm, next_state) in transitions
                         if frm == current)
        if not options:
            break
        weights = np.array([transitions[(current, o)] for o in options],
                           dtype=np.float64)
        path.append(int(rng.choice(options, p=weights / weights.sum())))
    return path


def build_segment_conditions(waves: list[np.ndarray], labels: list[int],
                             n_states: int, sample_seconds: int = 6
                             ) -> tuple[np.ndarray, LengthBucketizer, dict]:
    """Condition matrix = generic stats + one-hot state; fit the bucketizer."""
    lengths = np.array([len(w) for w in waves])
    bucketizer = LengthBucketizer.fit(lengths, n_buckets=6)
    length_scale = max(int(bucketizer.bucket_length(
        bucketizer.n_buckets - 1)), 1)
    mean_scale = max(float(np.quantile(
        [w.mean() for w in waves], 0.95)), 1e-9)
    rows = []
    for wave in waves:
        base = condition_vector(len(wave), wave, sample_seconds, mean_scale,
                                length_scale)
        onehot = np.zeros(n_states, dtype=np.float32)
        rows.append(np.concatenate([base, onehot]))
    conditions = np.stack(rows)
    normalizer = {"length_scale": length_scale, "mean_power_scale":
                  mean_scale}
    return conditions, bucketizer, normalizer


def segment_condition(length_samples: int, power: np.ndarray,
                      state_label: int, n_states: int,
                      normalizer: dict, sample_seconds: int = 6
                     ) -> np.ndarray:
    base = condition_vector(length_samples, power, sample_seconds,
                            normalizer["mean_power_scale"],
                            normalizer["length_scale"])
    onehot = np.zeros(n_states, dtype=np.float32)
    onehot[state_label] = 1.0
    return np.concatenate([base, onehot]).astype(np.float32)


class PrimitiveComposer(BaseGenerator):
    """B4 basic composition: sample a path, decode one primitive per state."""

    name = "b4_primitive_compose"
    version = "1"

    def __init__(self, model, bucketizer: LengthBucketizer,
                 donor_waves: list[np.ndarray], donor_labels: list[int],
                 path_model: dict, n_states: int, normalizer: dict,
                 sample_seconds: int = 6, device: str = "cpu",
                 power_scale: float = 1.0):
        import torch

        self.model = model
        self.bucketizer = bucketizer
        self.donor_waves = donor_waves
        self.donor_labels = donor_labels
        self.path_model = path_model
        self.n_states = n_states
        self.normalizer = normalizer
        self.sample_seconds = int(sample_seconds)
        self.device = device
        self.power_scale = float(power_scale)
        self._torch = torch
        model.to(device)
        model.eval()
        self._donors_by_state: dict[int, list[int]] = {}
        for index, label in enumerate(donor_labels):
            self._donors_by_state.setdefault(label, []).append(index)

    def config(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "n_states": self.n_states,
            "buckets": self.bucketizer.to_dict(),
        }

    def _decode_primitive(self, state_label: int, donor: np.ndarray,
                          rng_seed: int) -> np.ndarray:
        torch = self._torch
        torch.manual_seed(rng_seed)
        condition = torch.from_numpy(segment_condition(
            len(donor), donor, state_label, self.n_states,
            self.normalizer, self.sample_seconds)).unsqueeze(0).to(
                self.device)
        bucket_length = self.bucketizer.bucket_length(
            self.bucketizer.encode(len(donor)))
        latent = torch.randn(1, self.model.latent_dim, device=self.device)
        with torch.no_grad():
            decoded = self.model.decode(latent, condition).cpu().numpy()
        trimmed = np.asarray(decoded[0, 0, :len(donor)],
                             dtype=np.float64).clip(min=0.0) * self.power_scale
        del bucket_length
        return trimmed

    def generate_cycle(self, synthetic_cycle_id: str, seed: int,
                       rng: np.random.Generator
                       ) -> tuple[np.ndarray, SyntheticCycleRecord]:
        n_states_pool = sorted(self.path_model["path_lengths"])
        weights = np.array([
            self.path_model["path_lengths"][n] for n in n_states_pool],
            dtype=np.float64)
        n_states = int(rng.choice(
            n_states_pool, p=weights / weights.sum())) if n_states_pool else 2
        n_states = min(n_states, MAX_PATH_STATES)
        path = sample_path(self.path_model, n_states, rng)
        if not path:
            raise RuntimeError(f"{synthetic_cycle_id}: empty path sampled")

        waveform_parts: list[np.ndarray] = []
        segments: list[StateSegmentRecord] = []
        for order, state_label in enumerate(path):
            donors = self._donors_by_state.get(state_label)
            if not donors:
                raise RuntimeError(
                    f"{synthetic_cycle_id}: no donor for state {state_label}")
            donor_index = int(rng.choice(donors))
            donor = self.donor_waves[donor_index]
            primitive = self._decode_primitive(
                state_label, donor,
                rng_seed=seed * 1_000_003 + hash(synthetic_cycle_id) % 1_000
                + order)
            energy = float(primitive.sum() * self.sample_seconds / 3600.0)
            segments.append(StateSegmentRecord(
                state_label=state_label,
                target_duration_seconds=len(donor) * self.sample_seconds,
                actual_duration_seconds=len(primitive)
                * self.sample_seconds,
                target_samples=len(donor),
                actual_samples=len(primitive),
                mean_power_w=float(primitive.mean()),
                energy_wh=energy,
            ))
            waveform_parts.append(primitive)
        power = np.concatenate(waveform_parts)
        record = SyntheticCycleRecord(
            synthetic_cycle_id=synthetic_cycle_id,
            route="B4",
            seed=seed,
            generator_name=self.name,
            generator_version=self.version,
            checkpoint_sha256=None,
            conditions={"n_states": n_states,
                        "bucket_boundaries": self.bucketizer.to_dict()},
            state_path=path,
            segments=segments,
            boundary_treatment="none",
            sample_seconds=self.sample_seconds,
            created_utc="",
            waveform_sha256="",
        )
        return power, record
