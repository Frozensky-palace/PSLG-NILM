"""Build paired B1 full-cycle and B2 real-state-composition pilot cycles."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pilot_composition import (
    apply_linear_crossfade,
    boundary_jumps,
    choose_donor_indices,
    choose_matched_donor,
    correct_block_endpoints,
    filter_candidates_by_length_ratio,
    resample_waveform,
)

# Matched-feature subsets for the A3 ablation: (mean_weight, endpoint_weight).
MATCH_FEATURE_WEIGHTS = {
    "full": (0.25, 0.25),
    "duration": (0.0, 0.0),
    "duration_mean": (0.25, 0.0),
    "duration_endpoints": (0.0, 0.25),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_cycle(path: Path, power: np.ndarray, sample_seconds: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        appliance_w=np.asarray(power, dtype=np.float32),
        relative_time_s=np.arange(len(power), dtype=np.int64) * sample_seconds,
    )
    temporary.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cycle-library-dir", required=True)
    ap.add_argument("--state-library-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--synthetic-ratio", type=float, default=0.5)
    ap.add_argument("--sample-seconds", type=int, default=6)
    ap.add_argument("--b2-policy", choices=("random", "matched"), default="random")
    ap.add_argument("--matched-top-k", type=int, default=5)
    ap.add_argument("--match-features", choices=sorted(MATCH_FEATURE_WEIGHTS),
                    default="full",
                    help="matched-policy feature subset (A3 ablation)")
    ap.add_argument("--length-ratio-min", type=float, default=None,
                    help="exclude matched donors below target/donor ratio")
    ap.add_argument("--length-ratio-max", type=float, default=None,
                    help="exclude matched donors above target/donor ratio")
    ap.add_argument("--boundary-policy",
                    choices=("none", "linear_crossfade", "endpoint_offset"),
                    default="none", help="A4 boundary treatment")
    ap.add_argument("--crossfade-samples", type=int, default=10,
                    help="total samples blended around each boundary")
    args = ap.parse_args()
    if args.boundary_policy == "linear_crossfade" and args.crossfade_samples < 2:
        raise SystemExit("--crossfade-samples must be at least 2")
    if (args.length_ratio_min is None) != (args.length_ratio_max is None):
        raise SystemExit("--length-ratio-min and --length-ratio-max are joint")
    if args.length_ratio_min is not None and (
            args.length_ratio_min <= 0 or args.length_ratio_max
            < args.length_ratio_min):
        raise SystemExit("length ratio limits must satisfy 0 < min <= max")

    cycle_dir = Path(args.cycle_library_dir)
    state_dir = Path(args.state_library_dir)
    output_dir = Path(args.output_dir)
    cycles = pd.read_csv(cycle_dir / "real_cycle_library.csv")
    states = pd.read_csv(state_dir / "state_inventory.csv")
    if set(cycles["partition"].astype(str)) != {"train"}:
        raise SystemExit("B1/B2 source cycle library must contain train only")
    if set(states["source_partition"].astype(str)) != {"train"}:
        raise SystemExit("B2 state library must contain train only")
    if args.synthetic_ratio <= 0:
        raise SystemExit("--synthetic-ratio must be positive")

    with np.load(state_dir / "state_waveforms.npz") as data:
        state_power = data["power_w"].copy()
        offsets = data["offsets"].copy()
    state_by_id = states.set_index("state_block_id", drop=False)
    pools = {
        int(label): group["state_block_id"].to_numpy(dtype=np.int64)
        for label, group in states.groupby("state_label")
    }
    pool_cycle_ids = {
        label: state_by_id.loc[indices, "cycle_id"].astype(str).to_numpy()
        for label, indices in pools.items()
    }
    pool_samples = {
        label: state_by_id.loc[indices, "samples"].to_numpy(dtype=float)
        for label, indices in pools.items()
    }
    mean_weight, endpoint_weight = MATCH_FEATURE_WEIGHTS[args.match_features]

    count = int(round(len(cycles) * args.synthetic_ratio))
    rng = np.random.default_rng(args.seed)
    replace = count > len(cycles)
    template_rows = rng.choice(len(cycles), size=count, replace=replace)
    plan_records, donor_records, metric_records = [], [], []
    ratio_fallback_count = 0
    boundary_energy_delta_wh = []

    for synthetic_index, template_row in enumerate(template_rows):
        source = cycles.iloc[int(template_row)]
        template_cycle_id = str(source["cycle_id"])
        with np.load(cycle_dir / source["path"]) as data:
            b1_power = data["appliance_w"].astype(np.float32).copy()
        template_states = states[states["cycle_id"].astype(str) == template_cycle_id]
        template_states = template_states.sort_values("start_sample")
        if template_states.empty:
            raise RuntimeError(f"no state blocks for template {template_cycle_id}")
        if int(template_states.iloc[0]["start_sample"]) != 0 or \
                int(template_states.iloc[-1]["end_sample_exclusive"]) != len(b1_power):
            raise RuntimeError(f"state coverage mismatch for {template_cycle_id}")

        b2_parts = []
        boundaries = []
        cycle_treatment_delta_wh = 0.0
        cursor = 0
        for block_position, block in enumerate(template_states.itertuples(index=False)):
            label = int(block.state_label)
            donor_score = None
            ratio_limited = False
            if args.b2_policy == "random":
                donor_id = choose_donor_indices(
                    pools[label], pool_cycle_ids[label], template_cycle_id, rng)
            else:
                candidate_ids = pools[label]
                if args.length_ratio_min is not None:
                    candidate_ids, _ = filter_candidates_by_length_ratio(
                        candidate_ids, pool_samples[label], int(block.samples),
                        args.length_ratio_min, args.length_ratio_max)
                    cross = state_by_id.loc[
                        candidate_ids, "cycle_id"].astype(str).to_numpy()
                    if len(candidate_ids) and (cross != template_cycle_id).any():
                        ratio_limited = True
                    else:
                        ratio_fallback_count += 1
                        candidate_ids = pools[label]
                candidates = state_by_id.loc[candidate_ids]
                donor_id, donor_score = choose_matched_donor(
                    candidate_ids,
                    state_by_id.loc[candidate_ids, "cycle_id"].astype(str).to_numpy(),
                    template_cycle_id=template_cycle_id,
                    target_samples=int(block.samples),
                    target_mean_power_w=float(block.mean_power_w),
                    target_start_power_w=float(block.start_power_w),
                    target_end_power_w=float(block.end_power_w),
                    candidate_samples=candidates["samples"].to_numpy(dtype=float),
                    candidate_mean_power_w=(
                        candidates["mean_power_w"].to_numpy(dtype=float)),
                    candidate_start_power_w=(
                        candidates["start_power_w"].to_numpy(dtype=float)),
                    candidate_end_power_w=(
                        candidates["end_power_w"].to_numpy(dtype=float)),
                    rng=rng,
                    top_k=args.matched_top_k,
                    mean_weight=mean_weight,
                    endpoint_weight=endpoint_weight,
                )
            donor = state_by_id.loc[donor_id]
            donor_waveform = state_power[offsets[donor_id]:offsets[donor_id + 1]]
            target_length = int(block.samples)
            replacement = resample_waveform(donor_waveform, target_length)
            if args.boundary_policy == "endpoint_offset":
                replacement, sum_delta = correct_block_endpoints(
                    replacement, float(block.start_power_w),
                    float(block.end_power_w))
                cycle_treatment_delta_wh += (
                    sum_delta * args.sample_seconds / 3600.0)
            b2_parts.append(replacement)
            cursor += target_length
            if block_position + 1 < len(template_states):
                boundaries.append(cursor)
            donor_records.append({
                "synthetic_index": synthetic_index,
                "template_cycle_id": template_cycle_id,
                "block_position": block_position,
                "state_label": label,
                "target_samples": target_length,
                "donor_state_block_id": donor_id,
                "donor_cycle_id": str(donor["cycle_id"]),
                "donor_samples": int(donor["samples"]),
                "cross_cycle_donor": str(donor["cycle_id"]) != template_cycle_id,
                "donor_policy": args.b2_policy,
                "match_features": (args.match_features
                                   if args.b2_policy == "matched" else None),
                "ratio_limit_applied": ratio_limited,
                "match_score": donor_score,
                "target_to_donor_length_ratio": (
                    target_length / max(int(donor["samples"]), 1)),
            })
        b2_power = np.concatenate(b2_parts).astype(np.float32)
        if args.boundary_policy == "linear_crossfade":
            b2_power, crossfade_sum_delta = apply_linear_crossfade(
                b2_power, boundaries, args.crossfade_samples)
            cycle_treatment_delta_wh = (
                crossfade_sum_delta * args.sample_seconds / 3600.0)
        boundary_energy_delta_wh.append(cycle_treatment_delta_wh)
        if len(b1_power) != len(b2_power):
            raise RuntimeError("paired B1/B2 lengths differ")
        if (b2_power < 0).any():
            raise RuntimeError(f"negative power in synthetic cycle {synthetic_index}")

        b1_rel = Path("B1") / "cycles" / f"synthetic_{synthetic_index:04d}.npz"
        b2_rel = Path("B2") / "cycles" / f"synthetic_{synthetic_index:04d}.npz"
        _save_cycle(output_dir / b1_rel, b1_power, args.sample_seconds)
        _save_cycle(output_dir / b2_rel, b2_power, args.sample_seconds)
        b1_jumps = boundary_jumps(b1_power, boundaries)
        b2_jumps = boundary_jumps(b2_power, boundaries)
        plan_records.append({
            "synthetic_index": synthetic_index,
            "seed": args.seed,
            "template_cycle_id": template_cycle_id,
            "samples": len(b1_power),
            "duration_seconds": len(b1_power) * args.sample_seconds,
            "state_path": "-".join(template_states["state_label"].astype(str)),
            "state_blocks": len(template_states),
            "b1_path": b1_rel.as_posix(),
            "b2_path": b2_rel.as_posix(),
            "b1_sha256": _sha256(output_dir / b1_rel),
            "b2_sha256": _sha256(output_dir / b2_rel),
        })
        metric_records.append({
            "synthetic_index": synthetic_index,
            "template_cycle_id": template_cycle_id,
            "samples": len(b1_power),
            "state_boundaries": len(boundaries),
            "b1_energy_wh": float(b1_power.astype(np.float64).sum()
                                  * args.sample_seconds / 3600.0),
            "b2_energy_wh": float(b2_power.astype(np.float64).sum()
                                  * args.sample_seconds / 3600.0),
            "b2_boundary_treatment_energy_delta_wh": (
                float(cycle_treatment_delta_wh)),
            "b1_mean_boundary_jump_w": float(b1_jumps.mean()) if len(b1_jumps) else 0.0,
            "b2_mean_boundary_jump_w": float(b2_jumps.mean()) if len(b2_jumps) else 0.0,
            "b1_max_boundary_jump_w": float(b1_jumps.max()) if len(b1_jumps) else 0.0,
            "b2_max_boundary_jump_w": float(b2_jumps.max()) if len(b2_jumps) else 0.0,
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    plan = pd.DataFrame(plan_records)
    donors = pd.DataFrame(donor_records)
    metrics = pd.DataFrame(metric_records)
    plan_path = output_dir / "paired_cycle_plan.csv"
    donors_path = output_dir / "b2_donor_provenance.csv"
    metrics_path = output_dir / "paired_cycle_metrics.csv"
    plan.to_csv(plan_path, index=False)
    donors.to_csv(donors_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    summary = {
        "protocol": "paired_b1_b2_real_composition_v1",
        "status": "pilot",
        "b2_donor_policy": args.b2_policy,
        "matched_top_k": args.matched_top_k if args.b2_policy == "matched" else None,
        "match_features": (args.match_features
                           if args.b2_policy == "matched" else None),
        "length_ratio_limits": ([args.length_ratio_min, args.length_ratio_max]
                                if args.length_ratio_min is not None else None),
        "ratio_limited_fallback_donors": int(ratio_fallback_count),
        "ratio_limited_donors": int(donors["ratio_limit_applied"].sum()),
        "boundary_policy": args.boundary_policy,
        "crossfade_samples": (args.crossfade_samples
                              if args.boundary_policy == "linear_crossfade"
                              else None),
        "boundary_treatment_total_energy_delta_wh": float(
            metrics["b2_boundary_treatment_energy_delta_wh"].sum()),
        "seed": args.seed,
        "synthetic_ratio": args.synthetic_ratio,
        "paired_cycles": int(len(plan)),
        "same_template_cycles": True,
        "same_cycle_count": True,
        "same_duration_per_pair": True,
        "all_sources_train": True,
        "all_b2_donors_cross_cycle": bool(donors["cross_cycle_donor"].all()),
        "total_duration_seconds_each_arm": int(plan["duration_seconds"].sum()),
        "b1_total_energy_wh": float(metrics["b1_energy_wh"].sum()),
        "b2_total_energy_wh": float(metrics["b2_energy_wh"].sum()),
        "b1_mean_boundary_jump_w": float(
            metrics["b1_mean_boundary_jump_w"].mean()),
        "b2_mean_boundary_jump_w": float(
            metrics["b2_mean_boundary_jump_w"].mean()),
        "cycle_library_sha256": _sha256(cycle_dir / "real_cycle_library.csv"),
        "state_inventory_sha256": _sha256(state_dir / "state_inventory.csv"),
        "state_waveforms_sha256": _sha256(state_dir / "state_waveforms.npz"),
    }
    summary_path = output_dir / "b1_b2_pilot_summary.json"
    with open(summary_path, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(f"[B1/B2] paired cycles={len(plan)}, seed={args.seed}, ratio={args.synthetic_ratio}")
    print(f"[B1/B2] same total duration={summary['total_duration_seconds_each_arm']:,}s")
    print(f"[B1/B2] all B2 donors cross-cycle={summary['all_b2_donors_cross_cycle']}")
    print(f"[B1/B2] summary -> {summary_path}")


if __name__ == "__main__":
    main()
