"""Auditable pair averaging of aligned power data, never averaging labels."""
from __future__ import annotations

import numpy as np


POWER_FIELDS = ("mains_w", "appliance_w", "background_signed_w")
REQUIRED = {"timestamp", "segment_id", *POWER_FIELDS, "background_clipped_w"}


def expected_count(start, end, source_seconds, factor):
    """Epoch bins whose complete source support lies in inclusive [start,end]."""
    step = source_seconds * factor
    first = -(-int(start) // step) * step
    last = ((int(end) - (factor - 1) * source_seconds) // step) * step
    return max(0, (last - first) // step + 1)


def cycle_owners(timestamps, intervals):
    owners = np.full(len(timestamps), -1, dtype=np.int64)
    previous_end = None
    for index, (start, end) in enumerate(sorted(intervals)):
        if end < start or (previous_end is not None and start <= previous_end):
            raise ValueError("cycle intervals must be ordered, non-overlapping and nonempty")
        left = int(np.searchsorted(timestamps, start, side="left"))
        right = int(np.searchsorted(timestamps, end, side="right"))
        owners[int(left):right] = index
        previous_end = end
    return owners


def coarsen_arrays(arrays, *, source_seconds=6, factor=2, intervals=()):
    """Keep complete epoch-anchored bins within a shard/continuous segment.

    factor=1 produces the fresh native-rate control. factor=2 averages two
    consecutive powers and timestamps the output at the left edge. Incomplete
    bins are dropped and accounted for, not interpolated. On/off transitions
    remain continuous; cycle ownership is recorded, never averaged as a label.
    """
    if factor not in (1, 2) or source_seconds <= 0:
        raise ValueError("factor must be 1 or 2; source_seconds must be positive")
    if not REQUIRED.issubset(arrays):
        raise ValueError(f"missing fields: {sorted(REQUIRED.difference(arrays))}")
    if set(arrays).difference(REQUIRED):
        raise ValueError("unexpected source fields; define their aggregation explicitly: "
                         + str(sorted(set(arrays).difference(REQUIRED))))
    ts = np.asarray(arrays["timestamp"])
    if ts.ndim != 1 or not np.issubdtype(ts.dtype, np.integer):
        raise ValueError("timestamp must be a 1-D integer Unix-second vector")
    if np.any(np.diff(ts) <= 0) or np.any(ts % source_seconds):
        raise ValueError("timestamps must increase strictly on the source epoch grid")
    n = len(ts)
    for name in REQUIRED:
        value = np.asarray(arrays[name])
        if value.ndim != 1 or len(value) != n or not np.isfinite(value).all():
            raise ValueError(f"{name}: expected aligned finite 1-D values")
    if not np.issubdtype(arrays["segment_id"].dtype, np.integer):
        raise ValueError("segment_id must be integer; labels must not be averaged")
    owners = cycle_owners(ts, intervals)
    step = factor * source_seconds
    if factor == 1:
        left = right = np.arange(n, dtype=np.int64)
    else:
        left = np.flatnonzero(ts % step == 0)
        left = left[left + 1 < n]
        ok = ((ts[left + 1] - ts[left] == source_seconds)
              & (arrays["segment_id"][left + 1] == arrays["segment_id"][left]))
        left = left[ok]
        right = left + 1
    kept = np.zeros(n, dtype=bool)
    kept[left] = True
    kept[right] = True
    out = {"timestamp": ts[left].copy(),
           "source_left_row": left, "source_right_row": right}
    energy = {}
    for field in POWER_FIELDS:
        values = np.asarray(arrays[field], dtype=np.float64)
        out[field] = (values[left] + values[right]) / 2 if factor == 2 else values.copy()
        original = float(values.sum() * source_seconds / 3600)
        retained = float(values[kept].sum() * source_seconds / 3600)
        output = float(out[field].sum() * step / 3600)
        energy[field] = {
            "input_wh": original, "retained_input_wh": retained,
            "discarded_input_wh": float(values[~kept].sum() * source_seconds / 3600),
            "output_wh": output, "retained_energy_error_wh": output - retained,
        }
        if not np.isclose(output, retained, rtol=1e-10, atol=1e-7):
            raise ValueError(f"energy conservation failed for {field}")
    # Keep signed background linear; clipping then averaging is NOT equivalent.
    out["background_clipped_w"] = np.maximum(out["background_signed_w"], 0)
    if factor == 1:
        out["background_clipped_w"] = np.asarray(arrays["background_clipped_w"]).copy()
    breaks = np.ones(len(left), dtype=bool)
    if len(left) > 1:
        breaks[1:] = ((np.diff(out["timestamp"]) != step)
                      | (np.diff(arrays["segment_id"][left]) != 0))
    out["segment_id"] = np.cumsum(breaks, dtype=np.int64) - 1
    out["source_cycle_owner"] = np.where(owners[left] == owners[right], owners[left], -2)
    summary = {
        "input_rows": n, "output_rows": len(left),
        "retained_input_rows": int(kept.sum()), "discarded_input_rows": int((~kept).sum()),
        "output_sample_seconds": step,
        "discard_policy": "incomplete epoch bin, segment boundary or shard boundary",
        "bins_crossing_cycle_boundary": int(np.sum(owners[left] != owners[right])),
        "energy": energy,
        "max_background_identity_error_w": float(np.max(np.abs(
            out["mains_w"] - out["appliance_w"] - out["background_signed_w"]))) if len(left) else 0.,
    }
    # The row-index arrays provide an exact mapping; discarded rows can be
    # recovered as the complement without storing the source waveforms again.
    return out, summary
