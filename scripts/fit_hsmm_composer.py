"""Fit the frozen HSMM composition model from a state library (Phase F1).

Reads the frozen state library v1 inventory, reconstructs per-cycle state
sequences (ordered by start sample), and fits:
- Markov initial/transition counts (Laplace alpha frozen here),
- per-state empirical duration distributions,
- per-cycle state-count distribution.
The output JSON is the frozen composition model consumed by
compose_b5_cycles.py. Only train data is ever read.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.composition.duration_model import StateDurationModel  # noqa: E402
from src.composition.transition_model import MarkovChain  # noqa: E402


def build_sequences(inventory_rows: list[dict]) -> list[list[int]]:
    per_cycle: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in inventory_rows:
        per_cycle[row["cycle_id"]].append(
            (int(row["start_sample"]), int(row["state_label"])))
    sequences = []
    for cycle_id in sorted(per_cycle):
        blocks = sorted(per_cycle[cycle_id])
        sequences.append([label for _, label in blocks])
    return sequences


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state-library-dir", required=True,
                    help="frozen state library v1 (train-only)")
    ap.add_argument("--smoothing-alpha", type=float, default=1.0,
                    help="Laplace alpha within observed outgoing options; "
                         "frozen into the model file")
    ap.add_argument("--output", required=True, help="hsmm_model.json path")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(
        Path(args.state_library_dir) / "state_inventory.csv",
        encoding="utf-8")))
    partitions = {row.get("partition", "train") for row in rows}
    if partitions - {"train"}:
        raise SystemExit(f"non-train partitions found: {partitions - {'train'}}")

    sequences = build_sequences(rows)
    markov = MarkovChain.fit(sequences, smoothing_alpha=args.smoothing_alpha)

    observations = [
        (int(row["state_label"]), int(row["duration_seconds"]))
        for row in rows
    ]
    durations = StateDurationModel.fit(observations)

    path_lengths: dict[int, int] = {}
    for sequence in sequences:
        path_lengths[len(sequence)] = path_lengths.get(len(sequence), 0) + 1

    payload = {
        "protocol": "hsmm_model_v1",
        "n_sequences": len(sequences),
        "n_states": len(durations.states),
        "markov": markov.to_dict(),
        "durations": durations.to_dict(),
        "path_lengths": {str(k): v for k, v in sorted(path_lengths.items())},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=1, ensure_ascii=False),
                      encoding="utf-8")
    print(f"[hsmm] sequences={len(sequences)} states={len(durations.states)}"
          f" transitions={len(markov.transition_counts)} -> {output}")


if __name__ == "__main__":
    main()
