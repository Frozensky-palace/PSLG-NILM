"""Chronological, cycle-level train/validation/test split with leak checks."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


PARTITIONS = ("train", "validation", "test")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def apply_protocol_filters(frame: pd.DataFrame, *,
                           device_instance: str | int | None = None) -> pd.DataFrame:
    """Limit otherwise valid cycles to the device instance used by a protocol.

    The original eligibility is retained in ``inventory_eligible``.  Cycles
    from other physical devices remain visible for audit, but are not assigned
    to train/validation/test.
    """
    out = frame.copy()
    out["inventory_eligible"] = out["eligible"].astype(bool)
    if device_instance is None:
        return out
    if "device_instance" not in out.columns:
        raise ValueError("device_instance filter requested but inventory has no such column")
    wanted = str(device_instance)
    selected = out["device_instance"].astype(str) == wanted
    rejected = out["inventory_eligible"] & ~selected
    out.loc[rejected, "eligible"] = False
    if "exclusion_reason" not in out.columns:
        out["exclusion_reason"] = ""
    out["exclusion_reason"] = out["exclusion_reason"].fillna("").astype(str)
    out.loc[rejected, "exclusion_reason"] = (
        out.loc[rejected, "exclusion_reason"]
        .apply(lambda value: ";".join(
            part for part in (value, "different_device_instance") if part)))
    out["selected_for_protocol"] = out["eligible"].astype(bool)
    return out


def chronological_cycle_split(frame: pd.DataFrame, *, train_ratio: float = 0.6,
                              validation_ratio: float = 0.2,
                              test_ratio: float = 0.2) -> pd.DataFrame:
    """Assign whole eligible cycles in chronological order.

    Ratios operate on cycle count, not individual samples.  Ineligible rows are
    retained in the result with partition ``excluded`` for auditability.
    """
    ratios = [float(train_ratio), float(validation_ratio), float(test_ratio)]
    if any(value <= 0 for value in ratios):
        raise ValueError("all split ratios must be positive")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("train/validation/test ratios must sum to 1")
    required = {"cycle_id", "start_unix", "end_unix", "eligible"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"inventory missing required columns: {sorted(missing)}")

    out = frame.copy()
    out["partition"] = "excluded"
    eligible = out[out["eligible"].astype(bool)].sort_values(
        ["start_unix", "cycle_id"])
    n = len(eligible)
    if n < 3:
        raise ValueError("at least 3 eligible cycles are required for a 3-way split")

    n_train = max(1, int(n * train_ratio))
    n_validation = max(1, int(n * validation_ratio))
    if n_train + n_validation >= n:
        n_validation = 1
        n_train = n - 2
    n_test = n - n_train - n_validation
    if min(n_train, n_validation, n_test) < 1:
        raise ValueError("split produced an empty partition")

    train_ids = eligible.iloc[:n_train]["cycle_id"]
    validation_ids = eligible.iloc[n_train:n_train + n_validation]["cycle_id"]
    test_ids = eligible.iloc[n_train + n_validation:]["cycle_id"]
    out.loc[out["cycle_id"].isin(train_ids), "partition"] = "train"
    out.loc[out["cycle_id"].isin(validation_ids), "partition"] = "validation"
    out.loc[out["cycle_id"].isin(test_ids), "partition"] = "test"
    return out.sort_values(["start_unix", "cycle_id"]).reset_index(drop=True)


def leakage_report(frame: pd.DataFrame) -> dict:
    """Return machine-readable checks for the cycle-level split."""
    assigned = frame[frame["partition"].isin(PARTITIONS)].copy()
    duplicate_ids = assigned[assigned["cycle_id"].duplicated(False)]["cycle_id"].unique()
    problems: list[dict] = []

    ordered = assigned.sort_values(["start_unix", "end_unix"])
    rows = list(ordered[["cycle_id", "partition", "start_unix", "end_unix"]]
                .itertuples(index=False, name=None))
    for left, right in zip(rows, rows[1:]):
        if float(right[2]) <= float(left[3]) and right[1] != left[1]:
            problems.append({
                "type": "cross_partition_time_overlap",
                "left_cycle": left[0],
                "left_partition": left[1],
                "right_cycle": right[0],
                "right_partition": right[1],
            })

    partition_order = {name: idx for idx, name in enumerate(PARTITIONS)}
    seen = [partition_order[p] for p in ordered["partition"]]
    chronological_order_ok = seen == sorted(seen)
    if not chronological_order_ok:
        problems.append({"type": "partition_order_is_not_chronological"})

    eligible_unassigned = int(
        (frame["eligible"].astype(bool) & ~frame["partition"].isin(PARTITIONS)).sum())
    if eligible_unassigned:
        problems.append({
            "type": "eligible_cycle_unassigned",
            "count": eligible_unassigned,
        })
    if len(duplicate_ids):
        problems.append({
            "type": "duplicate_cycle_id",
            "cycle_ids": duplicate_ids.tolist(),
        })

    counts = frame["partition"].value_counts().to_dict()
    ranges = {}
    for partition in PARTITIONS:
        part = assigned[assigned["partition"] == partition]
        ranges[partition] = {
            "cycles": int(len(part)),
            "start_unix": int(part["start_unix"].min()) if len(part) else None,
            "end_unix": int(part["end_unix"].max()) if len(part) else None,
        }
    return {
        "passed": not problems,
        "problems": problems,
        "duplicate_cycle_ids": int(len(duplicate_ids)),
        "eligible_unassigned": eligible_unassigned,
        "chronological_order_ok": chronological_order_ok,
        "partition_counts": {str(k): int(v) for k, v in counts.items()},
        "partition_ranges": ranges,
    }
