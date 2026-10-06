"""Create waveform plots and a checklist for manual cycle-quality review."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cycle_audit import select_audit_cycles


def capture_ranges(csv_path: Path, selected: pd.DataFrame, *,
                   context_rows: int, context_seconds: int,
                   chunk_rows: int) -> dict[str, pd.DataFrame]:
    ranges = {}
    for row in selected.itertuples(index=False):
        ranges[str(row.cycle_id)] = {
            "capture_start": max(0, int(row.source_row_start) - context_rows),
            "capture_end": int(row.source_row_end) + context_rows,
            "time_start": float(row.start_unix) - context_seconds,
            "time_end": float(row.end_unix) + context_seconds,
            "pieces": [],
        }

    offset = 0
    for chunk_no, chunk in enumerate(pd.read_csv(
            csv_path, usecols=["timestamp", "power"], chunksize=chunk_rows), start=1):
        chunk_start = offset
        chunk_end = offset + len(chunk) - 1
        for item in ranges.values():
            left = max(item["capture_start"], chunk_start)
            right = min(item["capture_end"], chunk_end)
            if left <= right:
                local_left = left - chunk_start
                local_right = right - chunk_start + 1
                piece = chunk.iloc[local_left:local_right].copy()
                piece["source_row"] = np.arange(left, right + 1)
                item["pieces"].append(piece)
        offset += len(chunk)
        print(f"[cycle-audit] chunk={chunk_no} rows={offset:,}", flush=True)

    captured = {}
    for cycle_id, item in ranges.items():
        pieces = item["pieces"]
        waveform = (
            pd.concat(pieces, ignore_index=True) if pieces
            else pd.DataFrame(columns=["timestamp", "power", "source_row"])
        )
        if not waveform.empty:
            waveform = waveform[
                (waveform["timestamp"] >= item["time_start"])
                & (waveform["timestamp"] <= item["time_end"])
            ].reset_index(drop=True)
        captured[cycle_id] = waveform
    return captured


def waveform_stats(waveform: pd.DataFrame) -> dict:
    if waveform.empty:
        return {
            "captured_rows": 0, "nan_power_count": 0,
            "negative_power_count": 0, "max_timestamp_gap_seconds": None,
        }
    timestamps = waveform["timestamp"].to_numpy(dtype=float)
    powers = waveform["power"].to_numpy(dtype=float)
    gaps = np.diff(timestamps)
    return {
        "captured_rows": int(len(waveform)),
        "nan_power_count": int(np.isnan(powers).sum()),
        "negative_power_count": int(np.sum(np.isfinite(powers) & (powers < 0))),
        "max_timestamp_gap_seconds": float(np.nanmax(gaps)) if len(gaps) else 0.0,
    }


def draw_cycle(ax, waveform: pd.DataFrame, row, threshold_w: float,
               *, compact: bool = False) -> None:
    start = float(row.start_unix)
    end = float(row.end_unix)
    if waveform.empty:
        ax.text(0.5, 0.5, "missing waveform", ha="center", va="center")
        return
    minutes = (waveform["timestamp"].to_numpy(dtype=float) - start) / 60.0
    power = waveform["power"].to_numpy(dtype=float)
    timestamps = waveform["timestamp"].to_numpy(dtype=float)
    # Do not draw a diagonal line across a recording outage.
    if len(power) > 1:
        power = power.copy()
        power[1:][np.diff(timestamps) > 60.0] = np.nan
    ax.plot(minutes, power, linewidth=0.8)
    ax.axhline(threshold_w, color="tab:orange", linewidth=0.7, linestyle="--")
    ax.axvline(0, color="tab:green", linewidth=0.7)
    ax.axvline((end - start) / 60.0, color="tab:red", linewidth=0.7)
    ax.set_xlabel("Minutes from detected start")
    ax.set_ylabel("Power (W)")
    title = (f"{row.partition} | {row.audit_reason} | "
             f"{float(row.duration_seconds) / 60:.1f} min")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    if compact:
        ax.tick_params(labelsize=7)
        ax.xaxis.label.set_size(7)
        ax.yaxis.label.set_size(7)
        ax.title.set_size(8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--source-csv", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--per-partition", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--context-seconds", type=int, default=300)
    ap.add_argument("--sample-seconds", type=int, default=6)
    ap.add_argument("--threshold-w", type=float, default=20.0)
    ap.add_argument("--chunk-rows", type=int, default=1_000_000)
    args = ap.parse_args()

    inventory_path = Path(args.inventory)
    source_path = Path(args.source_csv)
    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    inventory = pd.read_csv(inventory_path)
    selected = select_audit_cycles(
        inventory, per_partition=args.per_partition, seed=args.seed)
    context_rows = int(round(args.context_seconds / args.sample_seconds))
    captured = capture_ranges(
        source_path, selected,
        context_rows=context_rows, context_seconds=args.context_seconds,
        chunk_rows=args.chunk_rows)

    checklist_rows = []
    overview_fig, overview_axes = plt.subplots(
        args.per_partition, 3,
        figsize=(15, max(3 * args.per_partition, 6)),
        constrained_layout=True,
    )
    if args.per_partition == 1:
        overview_axes = np.asarray([overview_axes])
    partition_col = {"train": 0, "validation": 1, "test": 2}
    partition_row_count = {"train": 0, "validation": 0, "test": 0}

    for row in selected.itertuples(index=False):
        cycle_id = str(row.cycle_id)
        waveform = captured[cycle_id]
        stats = waveform_stats(waveform)
        plot_name = f"{row.partition}_{row.audit_reason}_{cycle_id}.png"
        plot_path = plots_dir / plot_name

        fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
        draw_cycle(ax, waveform, row, args.threshold_w)
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

        r = partition_row_count[row.partition]
        c = partition_col[row.partition]
        draw_cycle(overview_axes[r, c], waveform, row, args.threshold_w, compact=True)
        partition_row_count[row.partition] += 1

        checklist_rows.append({
            **row._asdict(),
            **stats,
            "plot_path": str(plot_path.resolve()),
            "manual_complete_cycle": "",
            "manual_single_cycle": "",
            "manual_has_large_gap": "",
            "manual_has_abnormal_spike": "",
            "manual_keep": "",
            "manual_notes": "",
        })

    overview_path = output_dir / "audit_overview.png"
    overview_fig.suptitle(
        "UK-DALE House 1 washing machine cycle audit: green=start, red=end",
        fontsize=12)
    overview_fig.savefig(overview_path, dpi=150)
    plt.close(overview_fig)

    checklist = pd.DataFrame(checklist_rows)
    checklist_path = output_dir / "audit_checklist.csv"
    checklist.to_csv(checklist_path, index=False, encoding="utf-8-sig")
    selection_path = output_dir / "audit_selection.csv"
    selected.to_csv(selection_path, index=False, encoding="utf-8-sig")
    manifest = {
        "source_inventory": str(inventory_path.resolve()),
        "source_csv": str(source_path.resolve()),
        "per_partition": args.per_partition,
        "seed": args.seed,
        "context_seconds": args.context_seconds,
        "selected_cycles": int(len(selected)),
        "overview": str(overview_path.resolve()),
        "checklist": str(checklist_path.resolve()),
    }
    with open(output_dir / "audit_manifest.json", "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    print(f"[cycle-audit] selected={len(selected)}")
    print(f"[cycle-audit] overview -> {overview_path}")
    print(f"[cycle-audit] checklist -> {checklist_path}")


if __name__ == "__main__":
    main()
