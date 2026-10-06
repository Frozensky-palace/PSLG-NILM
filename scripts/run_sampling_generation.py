"""Paired 6s/12s generation: inherited models, budgets and evaluation scripts.

states: full train-only DETSEC-PC libraries, with historical k=4 fixed in advance.
generator: one rate/arm, final checkpoint -> generation -> both original audits.
No placement, NILM or test evaluation is performed here. A quality rejection is
a completed negative experiment, not an execution error or permission to retune.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.downsample_aligned_pairs import safe_source, sha256, write_json
from scripts.prepare_sampling_control import check_frozen_code
from scripts.run_recorded_command import fingerprints, run_recorded
from scripts.run_sampling_state_smoke import (
    RATES, activated_command, package_diagnostics, rate_config, read_json,
    validate_state_outputs, verify_prepared,
)
from src.generation.provenance import canonical_config_hash, utc_now_string
from src.generation.schema import SyntheticCycleRecord

ARM_ROUTES = {"B3T": "transform", "B3CVAE": "cvae", "B3WGAN": "wgan",
              "B3DIFF": "diffusion", "B4": "primitive",
              "B4WGAN": "primitive-wgan", "B4DIFF": "primitive-diffusion"}


def load_config(path):
    config = read_json(path)
    inputs = read_json(ROOT / config["input_config"])
    if ({a["arm"]: a["route"] for a in config["arms"]} != ARM_ROUTES
            or len(config["arms"]) != len(ARM_ROUTES)):
        raise ValueError("the seven inherited generation arms must be present exactly once")
    if config["state_k"] != 4 or config["state_epochs"] != 50:
        raise ValueError("paired v1 freezes historical k=4 and state epoch budget=50")
    if config["generation_count"] != inputs["expected_train_cycles"] // 2:
        raise ValueError("generation budget must remain floor(0.5 * train cycles)")
    return config, inputs


def matrix(config):
    return [{"index": i, "rate": rate, **arm}
            for i, (rate, arm) in enumerate(
                (rate, arm) for rate in RATES for arm in config["arms"])]


def command_specs(config, cell, real, state, output):
    """Build argv only: never alter neural implementations or metric formulas."""
    dt, route = RATES[cell["rate"]], cell["route"]
    primitive = route.startswith("primitive")
    checkpoint, cycles = output / "checkpoint", output / "cycles"
    common = ["--seed", config["generation_seed"], "--sample-seconds", dt]
    specs = []
    if route != "transform":
        args = ["--route", route, "--output-dir", checkpoint,
                "--state-library-dir" if primitive else "--real-library-dir",
                state if primitive else real, "--epochs", cell["epochs"],
                "--batch-size", config["batch_size"], "--learning-rate", cell["learning_rate"],
                "--latent-dim", config["latent_dim"], "--width", config["width"],
                "--n-buckets", config["n_buckets"], "--n-critic", config["n_critic"],
                "--diffusion-steps", config["diffusion_steps"], "--device", "cuda", *common]
        specs.append(("train", "train_full_cycle_generator.py", args,
                      [state if primitive else real], [checkpoint]))
    args = ["--output-dir", cycles, "--count", config["generation_count"], *common]
    if primitive:
        source = {"primitive": "cvae", "primitive-wgan": "wgan",
                  "primitive-diffusion": "diffusion"}[route]
        args += ["--primitive-source", source, "--state-library-dir", state,
                 "--checkpoint-dir", checkpoint, "--device", "cuda"]
        script, sources = "generate_primitive_cycles.py", [state, checkpoint]
    else:
        args += ["--route", route, "--real-library-dir", real]
        script, sources = "generate_full_cycles.py", [real]
        if route != "transform":
            args += ["--checkpoint-dir", checkpoint, "--device", "cuda"]
            sources.append(checkpoint)
    specs.append(("generate", script, args, sources, [cycles]))
    # Preserve the original per-arm protocol AND provide the same all-cycle
    # diversity scope for every arm. Do not overwrite the legacy report.
    legacy_count = config["generation_count"] if cell["arm"] in {"B4WGAN", "B4DIFF"} else 200
    for name, diversity_count in (("quality", legacy_count),
                                   ("quality_all_cycles", config["generation_count"])):
        dest = output / f"{name}_report.json"
        specs.append((name, "evaluate_synthetic_quality.py",
                      ["--synthetic-dir", cycles, "--real-library-dir", real,
                       "--sample-seconds", dt, "--diversity-max-cycles", diversity_count,
                       "--output", dest], [cycles, real], [dest]))
    dest = output / "memorization_report.json"
    specs.append(("memorization", "audit_generation_memorization.py",
                  ["--synthetic-dir", cycles, "--real-library-dir", real,
                   "--threshold-percentile", 1.0, "--max-replication-rate",
                   1.0 if route == "transform" else 0.01, "--output", dest],
                  [cycles, real], [dest]))
    return specs


def finite_numbers(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite numeric result")
    if isinstance(value, dict):
        for item in value.values():
            finite_numbers(item)
    elif isinstance(value, list):
        for item in value:
            finite_numbers(item)


def validate_generated(directory, count, seed, seconds):
    summary = read_json(directory / "generation_summary.json")
    if (summary["count"] != count or len(summary["records"]) != count
            or summary["seed"] != seed or summary["sample_seconds"] != seconds):
        raise ValueError("generated count, seed or sampling interval differs from protocol")
    seen = set()
    for payload in summary["records"]:
        record = SyntheticCycleRecord.from_dict(payload)
        finite_numbers(payload)
        if record.validate() or record.sample_seconds != seconds or record.seed != seed:
            raise ValueError("invalid generated record or time units")
        cid = record.synthetic_cycle_id
        if not re.fullmatch(r"synthetic_[0-9]+", cid) or cid in seen:
            raise ValueError("invalid or duplicate synthetic ID")
        seen.add(cid)
        with np.load(directory / "cycles" / f"{cid}.npz", allow_pickle=False) as data:
            power, timestamps = data["appliance_w"], data["relative_time_s"]
            if (power.ndim != 1 or not len(power) or not np.isfinite(power).all()
                    or (power < 0).any()
                    or not np.array_equal(timestamps, np.arange(len(power)) * seconds)
                    or sum(s.actual_samples for s in record.segments) != len(power)
                    or hashlib.sha256(power.astype(np.float32).tobytes()).hexdigest()
                    != record.waveform_sha256):
                raise ValueError(f"invalid generated waveform: {cid}")
    if {p.stem for p in (directory / "cycles").glob("*.npz")} != seen:
        raise ValueError("unexpected waveform files")
    return {"passed": True, "cycles_checked": count, "sample_seconds": seconds}


def summarize_gates(cell, quality, full_quality, memory, returncodes):
    for report in (quality, full_quality, memory):
        finite_numbers(report)
    cap = 1.0 if cell["arm"] == "B3T" else 0.01
    # The inherited memorization JSON's passed flag covers only exact copies;
    # also enforce the CLI cap, even if passed=true was written before exit 1.
    memory_pass = (memory["passed"] and memory["exact_duplicate_count"] == 0
                   and memory["replication_rate"] <= cap)
    warnings = sorted(k for k, v in quality["flags"].items() if v == "WARN")
    extra = True
    if cell["arm"] in {"B4WGAN", "B4DIFF"}:
        extra = (set(warnings) == {"duration_distribution"}
                 and full_quality["diversity"]["identical_pairs"] == 0
                 and full_quality["diversity"]["min_pairwise_distance"] >= 1e-3)
    passed = (quality["passed"] and full_quality["passed"] and memory_pass
              and extra and all(code == 0 for code in returncodes.values()))
    return {"quality_passed": quality["passed"], "warnings": warnings,
            "memorization_passed_with_route_cap": memory_pass,
            "replication_rate_cap": cap, "inherited_extra_gate_passed": bool(extra),
            "all_gates_passed": bool(passed),
            "result": "gates_passed_pending_review" if passed else "quality_rejected",
            "nilm_ready": False,
            "scope": "Generation evidence only; no NILM comparison or automatic retuning."}


def verify_handoff(state_root, config, inputs, expected_commit):
    state_root = Path(state_root).resolve()
    ready = read_json(state_root / "state_ready.json")
    if (ready["status"] != "ready_for_generation" or ready["git_commit"] != expected_commit
            or ready["config_hash"] != canonical_config_hash(config)
            or ready["input_config_hash"] != canonical_config_hash(inputs)
            or set(ready["states"]) != set(RATES)
            or set(ready["state_files"]) != set(RATES)
            or ready["selected_k"] != config["state_k"]):
        raise ValueError("state handoff differs from frozen code/configuration")
    for rate, entries in ready["state_files"].items():
        directory = state_root / rate
        seen = set()
        for entry in entries:
            path = safe_source(directory, entry["path"])
            if path in seen or sha256(path) != entry["sha256"]:
                raise ValueError("state handoff file changed or duplicated")
            seen.add(path)
        if not seen or seen != {p.resolve() for p in directory.rglob("*") if p.is_file()}:
            raise ValueError("state handoff file inventory incomplete")
    return ready


def mirror_diagnostics(archive, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(archive, "rb") as source, open(destination, "xb") as target:
        shutil.copyfileobj(source, target)


def run(config, inputs, output, *, stage, expected_commit, state_root=None,
        task_index=None, require_slurm=True, mirror=True):
    if require_slurm and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("submit via Slurm; no generation/training on login nodes")
    if stage not in {"states", "generator"} or not expected_commit:
        raise ValueError("explicit stage and frozen commit required")
    cell = None
    if stage == "generator":
        cells = matrix(config)
        if task_index is None or not 0 <= task_index < len(cells) or state_root is None:
            raise ValueError("generator requires valid task index and state root")
        cell = cells[task_index]
    source, output = Path(inputs["prepared_dir"]).resolve(), Path(output).resolve()
    for other in [source, *([Path(state_root).resolve()] if state_root else [])]:
        if output == other or output.is_relative_to(other) or other.is_relative_to(output):
            raise ValueError("output must not overlap source or state handoff")
    job = os.environ.get("SLURM_JOB_ID", "local")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", job):
        raise ValueError("unsafe job identifier")
    workflow_dirs = {}
    if stage == "states":
        workflow_dirs = {r: ROOT / "log" / f"sampling_states_{job}_{r}" for r in RATES}
        for path in workflow_dirs.values():
            if path.exists() or (ROOT / "output" / path.name).exists():
                raise FileExistsError(f"refusing to reuse workflow {path.name}")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "running", "stage": stage, "job_id": job, "cell": cell,
              "git_commit": expected_commit, "started_utc": utc_now_string(),
              "steps": {}, "test_accessed": False, "validation_arrays_accessed": False,
              "nilm_training_performed": False, "state_root": str(state_root) if state_root else None}
    write_json(output / "config_snapshot.json", {"experiment": config, "inputs": inputs})
    write_json(output / "run_manifest.json", report)

    def step(name, framework, script, args, in_paths=(), out_paths=(), gate=False):
        check_frozen_code(expected_commit)
        code = run_recorded(activated_command(inputs, inputs["python"][framework], script, args),
                            output / "records" / name, inputs=in_paths, outputs=out_paths)
        check_frozen_code(expected_commit)
        report["steps"][name] = {"returncode": code}
        write_json(output / "run_manifest.json", report)
        if code and not gate:
            raise RuntimeError(f"{name} failed ({code}); see records/{name}/execution.log")
        if any(not Path(p).exists() for p in out_paths):
            raise RuntimeError(f"{name} did not produce its reports")
        return code

    try:
        check_frozen_code(expected_commit)
        verified = verify_prepared(inputs, output / "input_verification.json",
                                   selection_cycles=inputs["expected_train_cycles"])
        ids = verified["selected_cycle_ids"]
        write_json(output / "selected_cycles.json", ids)
        if stage == "generator":
            handoff = verify_handoff(state_root, config, inputs, expected_commit)
            if handoff["selected_cycle_ids"] != ids:
                raise ValueError("state handoff cohort differs from accepted train cycles")
        framework = "tensorflow" if stage == "states" else "torch"
        env_path, gpu_path = output / "environment.json", output / "gpu.json"
        step("environment", framework, "capture_server_environment.py",
             ["--repo-root", ROOT, "--output", env_path], out_paths=[env_path])
        step("gpu", framework, "gpu_framework_smoke.py",
             ["--framework", framework, "--require-gpu", "true", "--output", gpu_path],
             out_paths=[gpu_path])
        if not read_json(gpu_path)["all_passed"]:
            raise ValueError("GPU check did not pass")
        ready = None
        if stage == "states":
            base = yaml.safe_load((ROOT / inputs["base_config"]).read_text())
            base["feature_extract"]["epochs"] = config["state_epochs"]
            checks = {}
            for rate, seconds in RATES.items():
                cfg = output / f"config_{rate}.yaml"
                cfg.write_text(yaml.safe_dump(rate_config(base, source, rate, output), sort_keys=False))
                result = output / rate
                step(f"state_{rate}", framework, "train_state_discovery.py",
                     ["--config", cfg, "--output-root", result,
                      "--run-id", workflow_dirs[rate].name, "--k", "3,4,5",
                      "--seed", 42, "--runtime-seed", inputs["runtime_seed"], "--deterministic-tf"],
                     [cfg, source / rate / "train_segments", source / rate / "train_library"],
                     [result, workflow_dirs[rate], ROOT / "output" / workflow_dirs[rate].name])
                log = (output / f"records/state_{rate}/execution.log").read_text()
                if "prim-glr error:" in log or "TF determinism unavailable:" in log:
                    raise ValueError("inherited segmentation/determinism error")
                checks[rate] = validate_state_outputs(
                    result, ids, seconds, expected_status="formal_candidate",
                    mapping_path=source / rate / "train_segments/segment_source_map.csv")
                history = read_json(workflow_dirs[rate] / "FeatureExtract_detsec_pc_on_prim-glr/training_history.json")
                finite_numbers(history)
                if not 1 <= history["epochs_trained"] <= config["state_epochs"]:
                    raise ValueError("invalid actual state training epoch count")
            ready = {"status": "ready_for_generation", "git_commit": expected_commit,
                     "config_hash": canonical_config_hash(config),
                     "input_config_hash": canonical_config_hash(inputs), "states": checks,
                     "selected_k": config["state_k"], "k_policy": "historical_k4_fixed_not_reselected",
                     "selected_cycle_ids": ids,
                     "state_files": {r: fingerprints([output / r]) for r in RATES}}
            report["result"] = "full_state_libraries_completed_for_fixed_k4_generation"
        else:
            rate = cell["rate"]
            real = source / rate / "train_library"
            state = Path(state_root) / rate / f"state_library_k{config['state_k']}"
            gate_codes = {}
            for name, script, args, in_paths, out_paths in command_specs(config, cell, real, state, output):
                is_gate = name in {"quality", "quality_all_cycles", "memorization"}
                code = step(name, "torch", script, args, in_paths, out_paths, gate=is_gate)
                if is_gate:
                    gate_codes[name] = code
                elif name == "train":
                    history = read_json(output / "checkpoint/history.json")
                    finite_numbers(history)
                    if len(history) != cell["epochs"]:
                        raise ValueError("generator did not finish the declared epoch budget")
                elif name == "generate":
                    write_json(output / "generation_integrity.json", validate_generated(
                        output / "cycles", config["generation_count"], config["generation_seed"], RATES[rate]))
            quality = read_json(output / "quality_report.json")
            full_quality = read_json(output / "quality_all_cycles_report.json")
            memory = read_json(output / "memorization_report.json")
            if any(r["n_synthetic"] != config["generation_count"] or
                   r["n_real_reference"] != inputs["expected_train_cycles"]
                   for r in (quality, full_quality, memory)):
                raise ValueError("metric sample counts differ from frozen budget")
            if full_quality["diversity"]["sampled_cycles"] != config["generation_count"]:
                raise ValueError("common diversity report did not cover every synthetic cycle")
            if cell["arm"] == "B4DIFF":
                finite_numbers(read_json(output / "cycles/peak_cap_report.json"))
            gates = summarize_gates(cell, quality, full_quality, memory, gate_codes)
            report.update(result=gates["result"], gates=gates)
            write_json(output / "generation_result.json", {
                "cell": cell, "seed": config["generation_seed"], "count": config["generation_count"],
                "gates": gates, "quality": quality, "quality_all_cycles": full_quality,
                "memorization": memory, "sample_seconds": RATES[rate],
                "boundary_slope_unit": "W/sample (inherited; not equal physical windows across rates)",
                "historical_results_are_reference_only": True})
            verify_handoff(state_root, config, inputs, expected_commit)
        after = verify_prepared(inputs, output / "input_verification_after.json",
                                selection_cycles=inputs["expected_train_cycles"])
        if verified != after:
            raise ValueError("prepared train data changed during run")
        check_frozen_code(expected_commit)
        if ready:
            write_json(output / "state_ready.json", ready)
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report.update(ended_utc=utc_now_string(), wall_seconds=time.monotonic() - started)
        write_json(output / "run_manifest.json", report)
        archive = package_diagnostics(output, workflow_dirs)
        destination = ROOT / "log" / f"{output.name}_diagnostics.tar.gz"
        if mirror:
            mirror_diagnostics(archive, destination)
        print(f"[sampling-generation] status={report['status']} result={report.get('result')} "
              f"diagnostics={archive} project_copy={destination if mirror else 'disabled'}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/server/zzz_sampling_generation.json")
    parser.add_argument("--stage", choices=("states", "generator"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--state-root")
    parser.add_argument("--task-index", type=int)
    args = parser.parse_args()
    config, inputs = load_config(args.config)
    run(config, inputs, args.output_dir, stage=args.stage, expected_commit=args.expected_commit,
        state_root=args.state_root, task_index=args.task_index)


if __name__ == "__main__":
    main()
