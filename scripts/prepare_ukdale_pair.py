"""Export an aligned UK-DALE appliance/mains pair for an end-to-end run.

Both outputs are ``(n, 2)`` NumPy arrays with identical Unix timestamps and
``[timestamp, power]`` columns.  Data are resampled onto one fixed grid.  Short
appliance gaps are interpolated; long appliance gaps become zero (unobserved
means inactive for event extraction).  Rows where mains remain unavailable are
removed from both arrays.

Example:
    python scripts/prepare_ukdale_pair.py --building 1 \
        --appliance "washing machine" --start 2013-03-18 --end 2013-03-25 \
        --out-prefix input/ukdale_b1_wm_week
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_dataset_series import _first_power_col, find_meter


def _fill_short_gaps(series: pd.Series, max_gap_samples: int) -> pd.Series:
    """Interpolate only complete NaN runs no longer than max_gap_samples."""
    if max_gap_samples <= 0 or not series.isna().any():
        return series
    missing = series.isna().to_numpy()
    keep_nan = np.zeros(len(series), dtype=bool)
    i = 0
    while i < len(series):
        if not missing[i]:
            i += 1
            continue
        j = i + 1
        while j < len(series) and missing[j]:
            j += 1
        if i == 0 or j == len(series) or (j - i) > max_gap_samples:
            keep_nan[i:j] = True
        i = j
    out = series.interpolate(method="time", limit_area="inside")
    out.iloc[keep_nan] = np.nan
    return out


def _select_series(store: pd.HDFStore, key: str, start: str, end: str) -> pd.Series:
    frame = store.select(
        key,
        where=f"index>=Timestamp('{start}') & index<Timestamp('{end}')",
    )
    if frame.empty:
        raise ValueError(f"no samples in {key} for [{start}, {end})")
    return frame[_first_power_col(frame)].astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5", default="datasets/ukdale/ukdale.h5")
    ap.add_argument("--building", type=int, default=1)
    ap.add_argument("--appliance", default="washing machine")
    ap.add_argument("--start", required=True, help="inclusive timestamp/date")
    ap.add_argument("--end", required=True, help="exclusive timestamp/date")
    ap.add_argument("--sample-seconds", type=int, default=6)
    ap.add_argument("--max-interpolate-seconds", type=int, default=150)
    ap.add_argument("--mains-meter", type=int, default=None,
                    help="override NILMTK's site meter number")
    ap.add_argument("--allow-washer-dryer", action="store_true")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--branch-csv", default=None,
                    help="optionally also write the aligned branch as timestamp,power CSV")
    args = ap.parse_args()

    if args.sample_seconds <= 0:
        raise SystemExit("--sample-seconds must be positive")
    if not os.path.exists(args.h5):
        raise SystemExit(f"h5 not found: {args.h5}")

    from nilmtk import DataSet

    with pd.HDFStore(args.h5, mode="r") as store:
        branch_key, branch_meta = find_meter(
            store, args.building, args.appliance, args.allow_washer_dryer)
        if branch_key is None:
            raise SystemExit(
                f"no appliance matching '{args.appliance}' in building {args.building}")

        if args.mains_meter is None:
            ds = DataSet(args.h5)
            try:
                mains_key = ds.buildings[args.building].elec.mains().key
            finally:
                ds.close()
        else:
            mains_key = f"/building{args.building}/elec/meter{args.mains_meter}"

        print(f"[prepare-pair] branch={branch_key} mains={mains_key}")
        branch = _select_series(store, branch_key, args.start, args.end)
        mains = _select_series(store, mains_key, args.start, args.end)

    rule = f"{args.sample_seconds}s"
    branch = branch.resample(rule, origin="epoch").mean()
    mains = mains.resample(rule, origin="epoch").mean()
    grid = branch.index.union(mains.index).sort_values()
    branch = branch.reindex(grid)
    mains = mains.reindex(grid)

    max_gap = args.max_interpolate_seconds // args.sample_seconds
    branch = _fill_short_gaps(branch, max_gap)
    mains = _fill_short_gaps(mains, max_gap)

    # A branch outage must not erase valid aggregate data.  Treat long branch
    # gaps as inactive, while dropping rows for which the aggregate is unknown.
    branch = branch.fillna(0.0)
    valid = mains.notna().to_numpy()
    timestamps = grid.view(np.int64) // 10**9
    branch_array = np.column_stack((timestamps[valid], branch.to_numpy()[valid]))
    mains_array = np.column_stack((timestamps[valid], mains.to_numpy()[valid]))

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    branch_path = str(prefix) + "_branch.npy"
    mains_path = str(prefix) + "_mains.npy"
    meta_path = str(prefix) + "_metadata.json"
    np.save(branch_path, branch_array)
    np.save(mains_path, mains_array)
    if args.branch_csv:
        csv_path = Path(args.branch_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(branch_array, columns=["timestamp", "power"]).to_csv(
            csv_path, index=False)

    metadata = {
        "h5": args.h5,
        "building": args.building,
        "appliance": args.appliance,
        "branch_key": branch_key,
        "mains_key": mains_key,
        "branch_metadata": branch_meta,
        "requested_start": args.start,
        "requested_end": args.end,
        "sample_seconds": args.sample_seconds,
        "rows": int(len(branch_array)),
        "actual_start_unix": float(branch_array[0, 0]),
        "actual_end_unix": float(branch_array[-1, 0]),
        "dropped_missing_mains_rows": int((~valid).sum()),
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2, default=str)

    print(f"[prepare-pair] rows={len(branch_array):,}, timestamps_aligned=True")
    print(f"[prepare-pair] branch -> {branch_path}")
    print(f"[prepare-pair] mains  -> {mains_path}")
    if args.branch_csv:
        print(f"[prepare-pair] branch CSV -> {args.branch_csv}")
    print(f"[prepare-pair] meta   -> {meta_path}")


if __name__ == "__main__":
    main()
