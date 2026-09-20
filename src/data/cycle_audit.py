"""Sampling helpers for visual quality review of detected work cycles."""
from __future__ import annotations

import numpy as np
import pandas as pd


def select_audit_cycles(frame: pd.DataFrame, *, per_partition: int = 6,
                        seed: int = 42) -> pd.DataFrame:
    """Select representative and extreme cycles from each partition.

    The default six rows are shortest, median-duration, longest, lowest-energy,
    highest-energy and one reproducible random cycle. Duplicate selections are
    replaced by additional random cycles.
    """
    required = {
        "cycle_id", "partition", "duration_seconds",
        "active_energy_wh_approx", "source_row_start", "source_row_end",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"audit inventory missing columns: {sorted(missing)}")
    if per_partition < 1:
        raise ValueError("per_partition must be positive")

    rng = np.random.default_rng(seed)
    selections: list[pd.Series] = []
    for partition in ("train", "validation", "test"):
        part = frame[frame["partition"] == partition].copy()
        if part.empty:
            raise ValueError(f"partition {partition!r} has no cycles")
        part = part.sort_values(["start_unix", "cycle_id"]).reset_index(drop=True)
        ordered_duration = part.sort_values("duration_seconds")
        ordered_energy = part.sort_values("active_energy_wh_approx")
        representative_candidates: list[tuple[str, pd.Series]] = [
            ("shortest", ordered_duration.iloc[0]),
            ("median_duration", ordered_duration.iloc[len(ordered_duration) // 2]),
            ("longest", ordered_duration.iloc[-1]),
            ("lowest_energy", ordered_energy.iloc[0]),
            ("highest_energy", ordered_energy.iloc[-1]),
        ]
        chosen: dict[str, pd.Series] = {}
        reasons: dict[str, list[str]] = {}
        for reason, row in representative_candidates:
            cycle_id = str(row["cycle_id"])
            chosen.setdefault(cycle_id, row.copy())
            reasons.setdefault(cycle_id, []).append(reason)

        random_order = rng.permutation(len(part))
        for idx in random_order:
            if len(chosen) >= min(per_partition, len(part)):
                break
            row = part.iloc[int(idx)]
            cycle_id = str(row["cycle_id"])
            if cycle_id not in chosen:
                chosen[cycle_id] = row.copy()
                reasons[cycle_id] = ["random"]

        for cycle_id, row in chosen.items():
            selected = row.copy()
            selected["audit_reason"] = "+".join(reasons[cycle_id])
            selections.append(selected)

    result = pd.DataFrame(selections)
    return result.reset_index(drop=True)
