"""Paired train-only state discovery smoke from an accepted preparation run.

Verify frozen metadata and all exported train files, check both GPU environments,
then reuse the inherited state-discovery pipeline on the same first N cycles.
Not a formal state library, generator experiment or NILM accuracy comparison.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import sys
import tarfile
import time

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.downsample_aligned_pairs import safe_source, sha256, write_json
from scripts.prepare_sampling_control import check_frozen_code
from scripts.run_recorded_command import run_recorded
from src.generation.provenance import utc_now_string

RATES = {"6s": 6, "12s": 12}
METADATA_NAMES = ("run_manifest.json", "preparation_summary.json", "source_verification.json",
                  "source_verification_after.json", *(f"records/{label}_{kind}/run_manifest.json"
                  for label in RATES for kind in ("train_library", "train_segments")))


def read_json(path):
    return json.loads(Path(path).read_text())


def verify_prepared(config, report_path):
    source = Path(config["prepared_dir"]).resolve()
    report = {"passed": False, "metadata": [], "train_files": [],
              "validation_arrays_read": False, "test_arrays_read": False}
    try:
        if set(config["expected_metadata_hashes"]) != set(METADATA_NAMES):
            raise ValueError("all eight accepted preparation metadata hashes are required")
        for relative, expected in config["expected_metadata_hashes"].items():
            path = safe_source(source, relative)
            actual = sha256(path)
            report["metadata"].append({"path": str(path), "sha256": actual})
            if not expected or actual != expected:
                raise ValueError(f"accepted preparation metadata changed: {relative}")
        run = read_json(source / "run_manifest.json")
        summary = read_json(source / "preparation_summary.json")
        if (run["status"] != "completed" or run["test_arrays_read"] or run["model_training_performed"]
                or run["environment"]["repo"]["commit"]["value"] != config["preparation_commit"]):
            raise ValueError("preparation identity/status does not match accepted data run")
        before = read_json(source / "source_verification.json")
        after = read_json(source / "source_verification_after.json")
        if not before["passed"] or before != after:
            raise ValueError("preparation sources were not verified unchanged")
        if any(ids for change in summary["eligibility_changes"].values() for ids in change.values()):
            raise ValueError("6s/12s eligible cycle sets changed")
        ordered_ids = []
        for label, seconds in RATES.items():
            for kind in ("train_library", "train_segments"):
                record = read_json(source / f"records/{label}_{kind}/run_manifest.json")
                directory = (source / label / kind).resolve()
                if record["status"] != "completed":
                    raise ValueError(f"incomplete preparation step: {label}/{kind}")
                seen = set()
                for entry in record["outputs"]:
                    path = safe_source(directory, entry["path"])
                    if path in seen or sha256(path) != entry["sha256"]:
                        raise ValueError(f"train file changed or duplicated: {path}")
                    seen.add(path)
                    report["train_files"].append({"path": str(path), "sha256": entry["sha256"]})
                if not seen or seen != {p.resolve() for p in directory.rglob("*") if p.is_file()}:
                    raise ValueError(f"unexpected or unrecorded train files: {directory}")
            base = source / label
            mapping = pd.read_csv(base / "train_segments/segment_source_map.csv").sort_values("csv_idx")
            inventory = pd.read_csv(base / "train_library/real_cycle_library.csv")
            library_meta = read_json(base / "train_library/real_cycle_library_manifest.json")
            n = config["expected_train_cycles"]
            if (len(mapping) != n or len(inventory) != n or not mapping.cycle_id.is_unique
                    or not inventory.cycle_id.is_unique or list(mapping.csv_idx) != list(range(n))
                    or set(mapping.partition) != {"train"} or set(inventory.partition) != {"train"}
                    or set(mapping.cycle_id) != set(inventory.cycle_id)
                    or library_meta["sample_seconds"] != seconds):
                raise ValueError(f"invalid train-only cohort or sampling interval: {label}")
            if list(mapping.filename) != sorted(mapping.filename) or not mapping.filename.is_unique:
                raise ValueError("filename order must match csv_idx")
            inv = inventory.set_index("cycle_id")
            for row in mapping.itertuples(index=False):
                item = inv.loc[row.cycle_id]
                if str(row.source_npz) != str(item.path) or int(row.samples) != int(item.samples):
                    raise ValueError("segment map differs from cycle library")
                safe_source((base / "train_library").resolve(), row.source_npz)
                safe_source((base / "train_segments").resolve(), row.filename)
                if not (base / "train_library" / row.source_npz).is_file() or not (base / "train_segments" / row.filename).is_file():
                    raise ValueError("segment map points to an absent train file")
            ordered_ids.append(list(mapping.cycle_id))
        if ordered_ids[0] != ordered_ids[1]:
            raise ValueError("paired rates must have identical ordered cycle IDs")
        n_smoke = int(config["smoke_cycles"])
        if not 5 <= n_smoke <= len(ordered_ids[0]) or int(config["smoke_epochs"]) < 1:
            raise ValueError("smoke needs at least 5 available cycles and positive epochs")
        report.update(passed=True, selected_cycle_ids=ordered_ids[0][:n_smoke],
                      verified_train_files=len(report["train_files"]))
    except BaseException as error:
        report["error"] = str(error)
        raise
    finally:
        write_json(report_path, report)
    return report


def rate_config(base, source, label, output):
    """Only paths, true sampling units and fresh-cache policy differ by rate."""
    config = copy.deepcopy(base)
    config["paths"]["segments_dir"] = str(source / label / "train_segments")
    config["paths"]["cache_dir"] = str(output / f"unused_cache_{label}")
    config["state_discovery"]["cycle_library_dir"] = str(source / label / "train_library")
    config["state_discovery"]["sample_seconds"] = RATES[label]
    config["temporal_state_merge"]["fs"] = 1.0 / RATES[label]
    config["feature_extract"]["cache"] = False
    return config


def validate_state_outputs(directory, cycle_ids, seconds):
    """Fail on dropped/truncated/overlapping blocks, including all missing cycles."""
    discovery = read_json(directory / "discovery_summary.json")
    if discovery["status"] != "smoke" or not discovery["train_only"] or discovery["test_accessed"]:
        raise ValueError("state result must remain train-only smoke")
    mapping = pd.read_csv(directory / "segments_smoke_subset/segment_source_map.csv")
    if list(mapping.sort_values("csv_idx").cycle_id) != cycle_ids:
        raise ValueError("state smoke used different cycles")
    checks = {}
    for k in (3, 4, 5):
        library = directory / f"state_library_k{k}"
        summary = read_json(library / "state_library_summary.json")
        inventory = pd.read_csv(library / "state_inventory.csv")
        if (not summary["cycle_coverage_exact"] or not summary["all_records_train"]
                or set(inventory.cycle_id) != set(cycle_ids)
                or set(inventory.source_partition) != {"train"}):
            raise ValueError(f"incomplete or non-train state coverage: k={k}")
        if not np.isfinite(inventory[["duration_seconds", "energy_wh", "mean_power_w"]].to_numpy()).all():
            raise ValueError("nonfinite state statistics")
        for row in mapping.itertuples(index=False):
            blocks = inventory[inventory.cycle_id == row.cycle_id].sort_values("start_sample")
            starts = blocks.start_sample.to_numpy()
            ends = blocks.end_sample_exclusive.to_numpy()
            if (starts[0] != 0 or ends[-1] != row.samples or np.any(ends <= starts)
                    or not np.array_equal(starts[1:], ends[:-1])
                    or not np.array_equal(ends - starts, blocks.samples.to_numpy())
                    or not np.array_equal(blocks.duration_seconds, blocks.samples * seconds)):
                raise ValueError(f"state blocks do not reconstruct cycle: {row.cycle_id}")
        checks[str(k)] = {"state_blocks": len(inventory), "states": len(summary["states"]),
                          "cycles": len(cycle_ids), "exact_coverage_verified": True}
    return checks


def package_diagnostics(output, workflow_dirs):
    """Metadata, statistics and logs only; tensors and waveforms stay on server."""
    archive = output / "diagnostics.tar.gz"
    candidates = [(p, p.relative_to(output).as_posix()) for p in output.rglob("*")
                  if p.is_file() and p.suffix in (".json", ".yaml", ".log", ".txt", ".csv")
                  and ("segments_smoke_subset" not in p.parts or p.name == "segment_source_map.csv")]
    for label, directory in workflow_dirs.items():
        if directory.exists():
            candidates += [(p, f"workflows/{label}/{p.relative_to(directory).as_posix()}")
                           for p in directory.rglob("*") if p.is_file() and p.suffix in (".json", ".log")]
    with tarfile.open(archive, "x:gz") as bundle:
        for path, name in sorted(candidates):
            bundle.add(path, arcname=name, recursive=False)
    return archive


def run(config, output_dir, *, expected_commit=None, require_slurm=True):
    if require_slurm and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("submit via Slurm; no training on the login node")
    if set(config["python"]) != {"tensorflow", "torch"}:
        raise ValueError("both TensorFlow and PyTorch environment checks are required")
    source, output = Path(config["prepared_dir"]).resolve(), Path(output_dir).resolve()
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("output must not overlap accepted preparation")
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", job_id):
        raise ValueError("unsafe job identifier")
    run_ids = {label: f"sampling_state_smoke_{job_id}_{label}" for label in RATES}
    workflow_dirs = {label: ROOT / "log" / name for label, name in run_ids.items()}
    for name in run_ids.values():
        if (ROOT / "log" / name).exists() or (ROOT / "output" / name).exists():
            raise FileExistsError(f"refusing to reuse workflow: {name}")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "running", "phase": "paired_state_smoke_not_formal_experiment",
              "started_utc": utc_now_string(), "steps_completed": [], "test_accessed": False,
              "validation_arrays_accessed": False, "generator_training_performed": False,
              "nilm_training_performed": False, "workflow_dirs": {k: str(v) for k, v in workflow_dirs.items()}}
    write_json(output / "config_snapshot.json", config)
    try:
        check_frozen_code(expected_commit)
        verified = verify_prepared(config, output / "input_verification.json")
        write_json(output / "selected_cycles.json", verified["selected_cycle_ids"])
        base = yaml.safe_load((ROOT / config["base_config"]).read_text())

        def step(name, python, script, arguments, inputs, outputs):
            check_frozen_code(expected_commit)
            code = run_recorded([python, str(ROOT / "scripts" / script), *map(str, arguments)],
                                output / "records" / name, inputs=inputs, outputs=outputs)
            check_frozen_code(expected_commit)
            if code != 0:
                raise RuntimeError(f"{name} failed ({code}); see records/{name}/execution.log")
            report["steps_completed"].append(name)
            write_json(output / "run_manifest.json", report)

        for framework, python in config["python"].items():
            environment = output / f"environment_{framework}.json"
            step(f"environment_{framework}", python, "capture_server_environment.py",
                 ["--repo-root", ROOT, "--output", environment], [], [environment])
            gpu = output / f"gpu_{framework}.json"
            step(f"gpu_{framework}", python, "gpu_framework_smoke.py",
                 ["--framework", framework, "--require-gpu", "true", "--output", gpu], [], [gpu])
            if not read_json(gpu)["all_passed"]:
                raise ValueError(f"GPU forward/backward did not pass: {framework}")
        results = {}
        for label, seconds in RATES.items():
            cfg_path = output / f"config_{label}.yaml"
            cfg_path.write_text(yaml.safe_dump(rate_config(base, source, label, output), sort_keys=False))
            result = output / label
            step(f"state_{label}", config["python"]["tensorflow"], "train_state_discovery.py",
                 ["--config", cfg_path, "--output-root", result, "--run-id", run_ids[label],
                  "--k", "3,4,5", "--seed", base["state_discovery"]["seed"],
                  "--runtime-seed", config["runtime_seed"], "--deterministic-tf",
                  "--smoke-cycles", config["smoke_cycles"], "--epochs-override", config["smoke_epochs"]],
                 [cfg_path, source / label / "train_library", source / label / "train_segments"],
                 [result, workflow_dirs[label], ROOT / "output" / run_ids[label]])
            execution_log = (output / "records" / f"state_{label}" / "execution.log").read_text()
            if "prim-glr error:" in execution_log or "TF determinism unavailable:" in execution_log:
                raise ValueError(f"{label}: inherited pipeline reported a suppressed backend failure")
            results[label] = validate_state_outputs(result, verified["selected_cycle_ids"], seconds)
        after = verify_prepared(config, output / "input_verification_after.json")
        if after != verified:
            raise ValueError("accepted train inputs changed during smoke")
        check_frozen_code(expected_commit)
        write_json(output / "state_smoke_summary.json", {
            "status": "paired_smoke_passed_not_formal_result", "rates": results,
            "selected_cycle_ids": verified["selected_cycle_ids"], "epochs_per_rate": config["smoke_epochs"],
            "runtime_seed": config["runtime_seed"], "kmeans_seed": base["state_discovery"]["seed"],
            "scope": "Interface/GPU/coverage checks only; no generator or NILM accuracy claim. State labels are rate-local."})
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report.update(ended_utc=utc_now_string(), wall_seconds=time.monotonic() - started)
        write_json(output / "run_manifest.json", report)
        try:
            archive = package_diagnostics(output, workflow_dirs)
        except Exception as error:
            primary_failure = report["status"] == "failed"
            report.update(status="failed", diagnostics_error=str(error))
            write_json(output / "run_manifest.json", report)
            if not primary_failure:
                raise
            print(f"[sampling-state] packaging also failed: {error}", flush=True)
        else:
            print(f"[sampling-state] status={report['status']} diagnostics={archive}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/server/zzz_sampling_state_smoke.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    run(read_json(args.config), args.output_dir, expected_commit=args.expected_commit)
