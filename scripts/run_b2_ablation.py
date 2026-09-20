"""Run the controlled B2 policy and boundary ablation grid (Phase A3/A4).

One factor changes at a time relative to the frozen matched baseline
(full features, top-k=5, no boundary treatment). Every variant rebuilds the
paired cycles, places them on the identical background, rebuilds the NILM
inputs and runs the same validation-only CPU smoke. Test data are only used to
define the same frozen window ranges as every other arm; no test evaluation
happens here.

The baseline arms (full features, top-k=5, no boundary treatment) already exist
in frozen directories and are not rebuilt; their vectorized-code smoke numbers
live in ``cpu_nilm_smoke_b2matched_k*_vec2``.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VARIANTS: dict[str, list[str]] = {
    "feat_duration": ["--match-features", "duration"],
    "feat_duration_mean": ["--match-features", "duration_mean"],
    "feat_duration_endpoints": ["--match-features", "duration_endpoints"],
    "topk1": ["--matched-top-k", "1"],
    "topk10": ["--matched-top-k", "10"],
    "ratio_0p67_1p5": ["--length-ratio-min", "0.67",
                       "--length-ratio-max", "1.5"],
    "cf10": ["--boundary-policy", "linear_crossfade",
             "--crossfade-samples", "10"],
    "epoff": ["--boundary-policy", "endpoint_offset"],
    # Single factor on top of the best A3 finding (duration-only matching):
    "feat_duration_ratio_0p67_1p5": [
        "--match-features", "duration",
        "--length-ratio-min", "0.67", "--length-ratio-max", "1.5"],
}

SCRIPTS = PROJECT_ROOT / "scripts"


def run_step(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(command)}\n")
        log.flush()
        completed = subprocess.run(
            [sys.executable, *command], stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise RuntimeError(f"step failed ({completed.returncode}): "
                           f"{command[0]} ... see {log_path}")


def chain(tag: str, state_library_dir: Path, variant_args: list[str],
          aligned_dir: Path, cycle_library_dir: Path, output_root: Path,
          seed: int, ratio: float) -> Path:
    variant_dir = output_root / tag
    paired = variant_dir / "paired"
    placed = variant_dir / "placed"
    inputs = variant_dir / "inputs"
    smoke = variant_dir / "smoke"
    log_path = variant_dir / "chain_log.txt"
    started = time.time()
    if (smoke / "cpu_smoke_results.json").exists():
        print(f"[ablation] {tag}: smoke exists, skip", flush=True)
        return smoke / "cpu_smoke_results.json"
    variant_dir.mkdir(parents=True, exist_ok=True)
    run_step([
        "scripts/build_b1_b2_pilot_cycles.py",
        "--cycle-library-dir", str(cycle_library_dir),
        "--state-library-dir", str(state_library_dir),
        "--output-dir", str(paired),
        "--seed", str(seed), "--synthetic-ratio", str(ratio),
        "--sample-seconds", "6",
        "--b2-policy", "matched", "--matched-top-k", "5",
        *variant_args,
    ], log_path)
    run_step([
        "scripts/place_b1_b2_on_train_background.py",
        "--aligned-dir", str(aligned_dir),
        "--paired-cycle-dir", str(paired),
        "--output-dir", str(placed),
        "--seed", str(seed), "--sample-seconds", "6",
        "--idle-threshold-w", "20", "--guard-seconds", "300",
    ], log_path)
    run_step([
        "scripts/prepare_nilm_b0_b2_inputs.py",
        "--aligned-dir", str(aligned_dir),
        "--placed-dir", str(placed),
        "--output-dir", str(inputs),
        "--window-length", "599", "--train-stride", "6",
        "--eval-stride", "1", "--sample-seconds", "6",
    ], log_path)
    run_step([
        "scripts/run_cpu_nilm_smoke.py",
        "--experiment-dir", str(inputs),
        "--output-dir", str(smoke),
        "--seed", str(seed), "--train-per-class", "25000",
        "--validation-count", "50000",
    ], log_path)
    print(f"[ablation] {tag}: chain done in {time.time() - started:.0f}s",
          flush=True)
    return smoke / "cpu_smoke_results.json"


def collect_row(tag: str, variant_dir: Path) -> dict:
    summary = json.loads(
        (variant_dir / "paired" / "b1_b2_pilot_summary.json").read_text(
            encoding="utf-8"))
    smoke = json.loads(
        (variant_dir / "smoke" / "cpu_smoke_results.json").read_text(
            encoding="utf-8"))["results"]
    row = {
        "tag": tag,
        "b2_mean_boundary_jump_w": summary["b2_mean_boundary_jump_w"],
        "b1_total_energy_wh": summary["b1_total_energy_wh"],
        "b2_total_energy_wh": summary["b2_total_energy_wh"],
        "b2_over_b1_energy": (summary["b2_total_energy_wh"]
                              / summary["b1_total_energy_wh"]),
        "ratio_limited_fallback_donors":
            summary.get("ratio_limited_fallback_donors"),
        "boundary_treatment_energy_delta_wh":
            summary.get("boundary_treatment_total_energy_delta_wh"),
    }
    for arm in ("B0", "B1", "B2"):
        row[f"{arm}_mae_w"] = smoke[arm]["mae_w"]
        row[f"{arm}_f1"] = smoke[arm]["f1"]
        row[f"{arm}_fp"] = smoke[arm]["fp"]
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aligned-dir", required=True)
    ap.add_argument("--cycle-library-dir", required=True)
    ap.add_argument("--state-library", action="append", required=True,
                    help="tag=directory, e.g. k4=reports/.../state_library_pilot_k4_v1")
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--ratio", type=float, default=0.5)
    ap.add_argument("--variants", default="all",
                    help="comma-separated variant names, or 'all'")
    args = ap.parse_args()

    variants = (sorted(VARIANTS) if args.variants == "all"
                else args.variants.split(","))
    unknown = [name for name in variants if name not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for item in args.state_library:
        tag_prefix, _, directory = item.partition("=")
        if not directory:
            raise SystemExit(f"--state-library must be tag=directory: {item}")
        for variant in variants:
            tag = f"{tag_prefix}_{variant}"
            try:
                chain(tag, Path(directory), VARIANTS[variant],
                      Path(args.aligned_dir), Path(args.cycle_library_dir),
                      output_root, args.seed, args.ratio)
            except RuntimeError as error:
                print(f"[ablation] {tag}: FAILED ({error})", flush=True)
                continue
            row = collect_row(tag, output_root / tag)
            row["status"] = "ok"
            append_row(output_root, row)

    print(f"[ablation] results csv -> {output_root / 'ablation_results.csv'}")


def append_row(output_root: Path, row: dict) -> None:
    path = output_root / "ablation_results.csv"
    existing = []
    if path.exists():
        with open(path, encoding="utf-8") as stream:
            existing = [item for item in csv.DictReader(stream)
                        if item["tag"] != row["tag"]]
    fieldnames = list(row)
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for item in existing:
            writer.writerow(item)
        writer.writerow(row)
    print(f"[ablation] appended {row['tag']}: "
          f"B2 MAE={row['B2_mae_w']:.3f} F1={row['B2_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
