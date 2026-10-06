"""Compose full cycles from an existing pool of generated primitives (B4).

Slices each primitive cycle of ``--primitives-dir`` into its labelled
segments, groups them by state and re-assembles full cycles following the
empirical train path model from the frozen state library. Composition is
the explicitly *basic* B4 rule; B5 constraints operate downstream of this.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.base_generator import BaseGenerator  # noqa: E402
from src.generation.primitive_cvae import fit_path_model  # noqa: E402
from src.generation.schema import (  # noqa: E402
    StateSegmentRecord,
    SyntheticCycleRecord,
)


def load_primitive_pool(primitives_dir: Path
                        ) -> tuple[list[np.ndarray], list[list[dict]], dict]:
    """Return (cycle waveforms, per-cycle segment records, summary)."""
    primitives_dir = Path(primitives_dir)
    summary = json.loads((primitives_dir / "generation_summary.json")
                         .read_text(encoding="utf-8"))
    waves: list[np.ndarray] = []
    segments_per_cycle: list[list[dict]] = []
    for record in summary["records"]:
        with np.load(primitives_dir / "cycles"
                     / f"{record['synthetic_cycle_id']}.npz") as data:
            waves.append(data["appliance_w"].astype(np.float64))
        segments_per_cycle.append(record["segments"])
    return waves, segments_per_cycle, summary


class PrimitivePoolComposer(BaseGenerator):
    """Assemble cycles from a labelled primitive pool + empirical paths."""

    name = "b4_composed"
    version = "1"

    def __init__(self, pool: list[np.ndarray],
                 segments_per_cycle: list[list[dict]], path_model: dict,
                 sample_seconds: int = 6):
        self.pool = pool
        self.segments_per_cycle = segments_per_cycle
        self.path_model = path_model
        self.sample_seconds = int(sample_seconds)
        self._by_state: dict[int, list[tuple[int, dict]]] = {}
        for cycle_index, segments in enumerate(segments_per_cycle):
            cursor = 0
            for segment in segments:
                length = int(segment["actual_samples"])
                self._by_state.setdefault(int(segment["state_label"]),
                                          []).append(
                    (cycle_index, {**segment, "_offset": cursor,
                                   "_length": length}))
                cursor += length

    def config(self) -> dict:
        return {"name": self.name, "version": self.version,
                "n_states": sorted(self._by_state)}

    def generate_cycle(self, synthetic_cycle_id: str, seed: int,
                       rng: np.random.Generator
                       ) -> tuple[np.ndarray, SyntheticCycleRecord]:
        pool_lengths = sorted(self.path_model["path_lengths"])
        weights = np.array([self.path_model["path_lengths"][n]
                            for n in pool_lengths], dtype=np.float64)
        n_states = int(rng.choice(pool_lengths,
                                  p=weights / weights.sum()))
        path = []
        first_state = self.path_model["first_state"]
        states = sorted(first_state)
        state_weights = np.array([first_state[s] for s in states])
        path.append(int(rng.choice(states, p=state_weights
                                   / state_weights.sum())))
        transitions = self.path_model["transitions"]
        while len(path) < n_states:
            options = sorted(next_state
                             for (frm, next_state) in transitions
                             if frm == path[-1])
            if not options:
                break
            w = np.array([transitions[(path[-1], o)] for o in options],
                         dtype=np.float64)
            path.append(int(rng.choice(options, p=w / w.sum())))

        parts: list[np.ndarray] = []
        segments: list[StateSegmentRecord] = []
        for order, state_label in enumerate(path):
            candidates = self._by_state.get(state_label)
            if not candidates:
                raise RuntimeError(
                    f"{synthetic_cycle_id}: empty primitive pool for state "
                    f"{state_label}")
            cycle_index, segment = candidates[
                int(rng.integers(len(candidates)))]
            offset = int(segment["_offset"])
            length = int(segment["_length"])
            parts.append(self.pool[cycle_index][offset:offset + length])
            segments.append(StateSegmentRecord(
                state_label=state_label,
                target_duration_seconds=length * self.sample_seconds,
                actual_duration_seconds=length * self.sample_seconds,
                target_samples=length,
                actual_samples=length,
                mean_power_w=float(parts[-1].mean()),
                energy_wh=float(
                    parts[-1].sum() * self.sample_seconds / 3600.0),
            ))
        power = np.concatenate(parts)
        record = SyntheticCycleRecord(
            synthetic_cycle_id=synthetic_cycle_id,
            route="B4",
            seed=seed,
            generator_name=self.name,
            generator_version=self.version,
            checkpoint_sha256=None,
            conditions={"n_states": n_states},
            state_path=path,
            segments=segments,
            boundary_treatment="none",
            sample_seconds=self.sample_seconds,
            created_utc="",
            waveform_sha256="",
        )
        return power, record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--primitives-dir", required=True,
                    help="generation output with labelled segments")
    ap.add_argument("--state-library-dir", required=True,
                    help="frozen state library (path model source)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--sample-seconds", type=int, default=6)
    args = ap.parse_args()

    pool, segments_per_cycle, _ = load_primitive_pool(
        Path(args.primitives_dir))
    rows = list(csv.DictReader(open(
        Path(args.state_library_dir) / "state_inventory.csv",
        encoding="utf-8")))
    path_model = fit_path_model(rows)
    composer = PrimitivePoolComposer(pool, segments_per_cycle, path_model,
                                     sample_seconds=args.sample_seconds)
    records = composer.generate_dataset(
        Path(args.output_dir), count=args.count, seed=args.seed,
        sample_seconds=args.sample_seconds)
    print(f"[compose] composed {len(records)} cycles -> {args.output_dir}")


if __name__ == "__main__":
    main()
