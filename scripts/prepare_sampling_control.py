"""Verify frozen sources and prepare paired 6s/12s train/validation artifacts.

No model training, test-array reads, raw DAT reconstruction or resplitting.
Every attempt uses a new output directory; failures retain diagnostics.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.downsample_aligned_pairs import safe_source, sha256, write_json
from scripts.inspect_server_setup import git_value, inspect_setup
from scripts.run_recorded_command import run_recorded
from src.generation.provenance import utc_now_string

PARTITIONS = ("train", "validation")


def check_frozen_code(expected_commit):
    if expected_commit:
        commit = git_value(ROOT, "rev-parse", "HEAD")
        changes = git_value(ROOT, "status", "--short", "--untracked-files=no")
        if commit.get("value") != expected_commit or changes.get("value") != "":
            raise ValueError("checkout changed since submission; use a clean frozen commit")


def input_paths(config):
    spec = config["preparation"]
    aligned = Path(spec["source_aligned_dir"]).resolve()
    return {
        "aligned_manifest": aligned / "aligned_partition_manifest.json",
        "coverage": aligned / "cycle_alignment_coverage.csv",
        "inventory": Path(spec["source_inventory_path"]).resolve(),
        "anchor_dataset_manifest": Path(spec["anchor_dataset_manifest"]).resolve(),
    }


def verify_inputs(config, report_path=None):
    spec = config["preparation"]
    if tuple(spec["partitions"]) != PARTITIONS:
        raise ValueError("this preparation job is restricted to train/validation")
    paths = input_paths(config)
    report = {"passed": False, "metadata": {}, "shards": [], "test_arrays_read": False}
    try:
        for name, path in paths.items():
            actual = sha256(path)
            expected = spec["expected_hashes"][name]
            report["metadata"][name] = {"path": str(path), "sha256": actual, "expected_sha256": expected}
            if not expected or actual != expected:
                raise ValueError(f"frozen metadata hash mismatch: {name}")
        manifest = json.loads(paths["aligned_manifest"].read_text())
        anchor = json.loads(paths["anchor_dataset_manifest"].read_text())
        if manifest["sample_seconds"] != 6 or anchor["sample_seconds"] != 6:
            raise ValueError("both source and historical anchor must use 6 seconds")
        if anchor["source_hashes"]["aligned_manifest"] != report["metadata"]["aligned_manifest"]["sha256"]:
            raise ValueError("historical experiment used a different aligned manifest")
        if manifest["source_inventory_sha256"] != report["metadata"]["inventory"]["sha256"]:
            raise ValueError("cycle inventory differs from the original aligned data")
        coverage = pd.read_csv(paths["coverage"])
        inventory = pd.read_csv(paths["inventory"])
        if coverage.cycle_id.duplicated().any() or inventory.cycle_id.duplicated().any():
            raise ValueError("duplicate cycle IDs in frozen metadata")
        # The Windows provenance string remains untouched; the verified SHA
        # identifies its relocated CSV on the server without modifying it.
        for part in PARTITIONS:
            rows = coverage[coverage.partition == part].sort_values("cycle_id")
            original = inventory[inventory.partition == part].sort_values("cycle_id")
            columns = ["cycle_id", "partition", "start_unix", "end_unix"]
            if not rows[columns].reset_index(drop=True).equals(original[columns].reset_index(drop=True)):
                raise ValueError(f"cycle boundaries/partition assignments differ in {part}")
            eligible = rows.analysis_eligible.map(lambda v: str(v).lower() == "true")
            if int(eligible.sum()) != spec["expected_eligible_cycles"][part]:
                raise ValueError(f"unexpected source eligible cycle count in {part}")
            shards = manifest["partitions"][part]["shards"]
            if not shards:
                raise ValueError(f"no source shards in {part}")
            for shard in shards:
                path = safe_source(paths["aligned_manifest"].parent, shard["path"])
                actual = sha256(path)
                entry = {"partition": part, "path": str(path), "bytes": path.stat().st_size,
                         "sha256": actual, "expected_sha256": shard.get("sha256")}
                report["shards"].append(entry)
                if not shard.get("sha256") or actual != shard["sha256"]:
                    raise ValueError(f"source shard hash mismatch: {path}")
        report.update(passed=True, verified_shards=len(report["shards"]),
                      source_eligible_cycles=spec["expected_eligible_cycles"])
    except BaseException as error:
        report["error"] = str(error)
        raise
    finally:
        if report_path:
            write_json(report_path, report)
    return report


def build_summary(output):
    output = Path(output)
    rates, cycles = {}, []
    for label in ("6s", "12s"):
        base = output / label
        audit = json.loads((base / "aligned/downsampling_audit.json").read_text())
        coverage = pd.read_csv(base / "aligned/cycle_alignment_coverage.csv")
        totals = {}
        for part in PARTITIONS:
            shards = [s for s in audit["shards"] if s["partition"] == part]
            rows = coverage[coverage.partition == part]
            energy = {}
            for field in ("mains_w", "appliance_w", "background_signed_w"):
                energy[field] = {key: sum(s["energy"][field][key] for s in shards)
                                 for key in shards[0]["energy"][field]}
            totals[part] = {
                key: sum(s[key] for s in shards)
                for key in ("input_rows", "output_rows", "discarded_input_rows", "bins_crossing_cycle_boundary")}
            totals[part].update(eligible_cycles=int(rows.analysis_eligible.sum()), energy=energy)
            totals[part]["cross_shard_pair_count"] = sum(s.get("cross_shard_pair_count", 0) for s in shards)
            library = pd.read_csv(base / f"{part}_library/real_cycle_library.csv")
            selected = library[["cycle_id", "samples", "duration_grid_seconds", "appliance_energy_wh",
                                "max_appliance_power_w"]]
            paired = rows.merge(selected, on="cycle_id", how="left", validate="one_to_one")
            paired["rate"] = label
            cycles.append(paired)
            totals[part]["minimum_library_samples"] = int(library.samples.min())
        rates[label] = totals
    all_cycles = pd.concat(cycles, ignore_index=True)
    columns = ["cycle_id", "partition", "start_unix", "end_unix", "analysis_eligible",
               "analysis_exclusion_reason", "samples", "duration_grid_seconds",
               "appliance_energy_wh", "max_appliance_power_w"]
    comparison = all_cycles[all_cycles.rate == "6s"][columns].merge(
        all_cycles[all_cycles.rate == "12s"][columns], on=["cycle_id", "partition"],
        how="outer", suffixes=("_6s", "_12s"), validate="one_to_one", indicator=True)
    comparison["energy_delta_wh"] = comparison.appliance_energy_wh_12s - comparison.appliance_energy_wh_6s
    comparison["left_edge_shift_seconds"] = comparison.start_unix_12s - comparison.start_unix_6s
    comparison["right_edge_shift_seconds"] = comparison.end_unix_12s - comparison.end_unix_6s
    comparison.to_csv(output / "cycle_comparison.csv", index=False)
    changes = {}
    for part in PARTITIONS:
        rows = comparison[comparison.partition == part]
        changes[part] = {
            "lost_eligible_cycle_ids": rows.loc[rows.analysis_eligible_6s & ~rows.analysis_eligible_12s, "cycle_id"].tolist(),
            "gained_eligible_cycle_ids": rows.loc[~rows.analysis_eligible_6s & rows.analysis_eligible_12s, "cycle_id"].tolist(),
        }
    summary = {"status": "data_prepared_for_review_not_model_results", "rates": rates,
               "eligibility_changes": changes, "test_arrays_read": False,
               "scope": "Continuous retained-bin energy and per-cycle edge/eligibility changes are distinct; no model was trained."}
    write_json(output / "preparation_summary.json", summary)
    return summary


def package_diagnostics(output):
    output = Path(output)
    candidates = [output / name for name in ("run_manifest.json", "config_snapshot.json",
                  "source_verification.json", "source_verification_after.json", "preparation_summary.json",
                  "cycle_comparison.csv")]
    candidates += list((output / "source_metadata").glob("*"))
    candidates += list((output / "records").glob("*/run_manifest.json"))
    candidates += list((output / "records").glob("*/execution.log"))
    for rate in ("6s", "12s"):
        candidates += [output / rate / "aligned" / name for name in (
            "aligned_partition_manifest.json", "cycle_alignment_coverage.csv", "downsampling_audit.json")]
        for part in PARTITIONS:
            candidates += [output / rate / f"{part}_library" / name for name in (
                "real_cycle_library.csv", "real_cycle_library_manifest.json")]
        candidates += [output / rate / "train_segments" / name for name in (
            "segment_source_map.csv", "segment_export_manifest.json")]
    archive = output / "diagnostics.tar.gz"
    with tarfile.open(archive, "x:gz") as bundle:
        for path in sorted(set(candidates)):
            if path.is_file():
                bundle.add(path, arcname=path.relative_to(output).as_posix(), recursive=False)
    return archive


def prepare(config, output_dir, *, require_slurm=True, expected_commit=None):
    if require_slurm and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("submit this preparation job with Slurm; do not run on the login node")
    output = Path(output_dir).resolve()
    for path in [Path(config["preparation"]["source_aligned_dir"]).resolve(), *input_paths(config).values()]:
        if output == path or output.is_relative_to(path) or path.is_relative_to(output):
            raise ValueError("output must not overlap any source path")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    run = {"status": "running", "started_utc": utc_now_string(), "phase": "sampling_data_preparation",
           "test_arrays_read": False, "model_training_performed": False, "steps_completed": []}
    write_json(output / "run_manifest.json", run)
    write_json(output / "config_snapshot.json", config)
    try:
        run["environment"] = inspect_setup(ROOT, [])
        check_frozen_code(expected_commit)
        verified = verify_inputs(config, output / "source_verification.json")
        snapshot = output / "source_metadata"
        snapshot.mkdir()
        for name, path in input_paths(config).items():
            shutil.copyfile(path, snapshot / f"{name}{path.suffix}")
        source = config["preparation"]["source_aligned_dir"]
        source_inputs = [*input_paths(config).values(), *(Path(s["path"]) for s in verified["shards"])]

        def step(name, script, arguments, inputs, result):
            check_frozen_code(expected_commit)
            code = run_recorded([sys.executable, str(ROOT / "scripts" / script), *map(str, arguments)],
                                output / "records" / name, inputs=inputs, outputs=[result])
            check_frozen_code(expected_commit)
            if code != 0:
                raise RuntimeError(f"step {name} failed ({code}); see its execution.log")
            run["steps_completed"].append(name)
            write_json(output / "run_manifest.json", run)

        for factor, label in ((1, "6s"), (2, "12s")):
            base = output / label
            aligned = base / "aligned"
            arguments = ["--aligned-dir", source, "--output-dir", aligned, "--factor", factor]
            if factor == 2 and config["preparation"].get("pair_across_continuous_storage_shards", False):
                arguments.append("--pair-across-shards")
            step(f"{label}_aligned", "downsample_aligned_pairs.py",
                 arguments, source_inputs, aligned)
            for part in PARTITIONS:
                library = base / f"{part}_library"
                step(f"{label}_{part}_library", "build_real_cycle_library.py",
                     ["--aligned-dir", aligned, "--partition", part, "--output-dir", library], [aligned], library)
            segments = base / "train_segments"
            step(f"{label}_train_segments", "export_cycle_library_segments.py",
                 ["--library-dir", base / "train_library", "--output-dir", segments], [base / "train_library"], segments)
        build_summary(output)
        verify_inputs(config, output / "source_verification_after.json")
        check_frozen_code(expected_commit)
        run["status"] = "completed"
    except BaseException as error:
        run.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        run.update(ended_utc=utc_now_string(), wall_seconds=time.monotonic() - started)
        write_json(output / "run_manifest.json", run)
        try:
            archive = package_diagnostics(output)
        except Exception as error:
            had_primary_error = run["status"] == "failed"
            run.update(status="failed", diagnostics_error=str(error))
            write_json(output / "run_manifest.json", run)
            if not had_primary_error:
                raise
            print(f"[sampling-prepare] diagnostic packaging also failed: {error}", flush=True)
        else:
            print(f"[sampling-prepare] status={run['status']} diagnostics={archive}", flush=True)
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/server/zzz_sampling_control.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    prepare(config, args.output_dir, expected_commit=args.expected_commit)


if __name__ == "__main__":
    main()
