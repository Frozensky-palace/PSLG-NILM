"""Collect every expected sampling-generation cell, including failed/missing ones.

Runs without training or GPU use. Packages small per-job diagnostics (not model
weights/waveforms) into ONE archive directly under project log for downloading.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import re
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.downsample_aligned_pairs import sha256, write_json
from scripts.run_sampling_generation import load_config, matrix


def compare_primitive_plans(summaries):
    comparisons = {}
    for rate in ("6s", "12s"):
        keys = [(rate, arm) for arm in ("B4", "B4WGAN", "B4DIFF")]
        if not all(key in summaries for key in keys):
            comparisons[rate] = {"status": "incomplete"}
            continue
        def signature(summary):
            return [(r["synthetic_cycle_id"], r["state_path"],
                     [s["target_samples"] for s in r["segments"]],
                     r["conditions"].get("rng_seeds")) for r in summary["records"]]
        signatures = [signature(summaries[key]) for key in keys]
        equal = all(sig == signatures[0] for sig in signatures[1:])
        comparisons[rate] = {"status": "passed" if equal else "mismatch",
                             "cycles_checked": len(signatures[0]),
                             "scope": "Same-rate B4 model paths, target lengths and decode seeds; not cross-rate state-label matching."}
    return comparisons


def collect(config, inputs, artifact_root, project_log, state_job, array_job):
    if not all(re.fullmatch(r"[0-9]+", s) for s in (state_job, array_job)):
        raise ValueError("numeric job IDs required")
    archive = project_log / f"sampling_generation_{array_job}_diagnostics.tar.gz"
    summary_path = project_log / f"sampling_generation_{array_job}_summary.json"
    if archive.exists() or summary_path.exists():
        raise FileExistsError("collection already exists; preserve it, use a new project-log directory for a later snapshot")
    report = {"state_job": state_job, "array_job": array_job, "expected_cells": 14,
              "scope": "Generation comparison only; no NILM results", "cells": [],
              "source_archives": [], "matrix_complete": False}
    bundles, primitives, commits = [], {}, set()
    state_root = artifact_root / f"sampling_states_{state_job}"
    if (state_root / "diagnostics.tar.gz").is_file():
        bundles.append(("states", state_root / "diagnostics.tar.gz"))
    for cell in matrix(config):
        folder = artifact_root / f"sampling_generation_{array_job}_{cell['index']}"
        entry = {"cell": cell, "status": "missing", "directory": str(folder)}
        manifest = folder / "run_manifest.json"
        if manifest.is_file():
            run = json.loads(manifest.read_text())
            if run["cell"] != cell or run.get("state_root") != str(state_root):
                raise ValueError("wrong matrix cell or state source")
            snapshot = json.loads((folder / "config_snapshot.json").read_text())
            if snapshot != {"experiment": config, "inputs": inputs}:
                raise ValueError("configuration drift across matrix")
            commits.add(run["git_commit"])
            entry.update(status=run["status"], result=run.get("result"),
                         error=run.get("error"), wall_seconds=run.get("wall_seconds"))
            if run["status"] == "completed":
                entry["evaluation"] = json.loads((folder / "generation_result.json").read_text())
                if cell["arm"] in {"B4", "B4WGAN", "B4DIFF"}:
                    primitives[(cell["rate"], cell["arm"])] = json.loads(
                        (folder / "cycles/generation_summary.json").read_text())
        bundle = folder / "diagnostics.tar.gz"
        if bundle.is_file():
            bundles.append((f"task_{cell['index']}", bundle))
        report["cells"].append(entry)
    if len(commits) > 1:
        raise ValueError("mixed code commits in comparison")
    report["git_commits"] = sorted(commits)
    report["primitive_plan_pairing"] = compare_primitive_plans(primitives)
    report["matrix_complete"] = all(r["status"] == "completed" for r in report["cells"])
    report["all_gates_passed"] = (report["matrix_complete"] and all(
        r["evaluation"]["gates"]["all_gates_passed"] for r in report["cells"])
        and all(r["status"] == "passed" for r in report["primitive_plan_pairing"].values()))
    report["source_archives"] = [{"label": label, "path": str(path), "sha256": sha256(path)}
                                 for label, path in bundles]
    project_log.mkdir(parents=True, exist_ok=True)
    write_json(summary_path, report)
    with tarfile.open(archive, "x:gz") as destination:
        destination.add(summary_path, arcname="matrix_summary.json", recursive=False)
        for label, path in bundles:
            with tarfile.open(path) as source:
                for member in source:
                    if not member.isfile():
                        continue
                    relative = Path(member.name)
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError("unsafe archive member")
                    payload = source.extractfile(member).read()
                    info = tarfile.TarInfo(f"{label}/{relative.as_posix()}")
                    info.size = len(payload)
                    destination.addfile(info, io.BytesIO(payload))
    return report, archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-job", required=True)
    parser.add_argument("--array-job", required=True)
    parser.add_argument("--artifact-root", type=Path, default=Path("/home/scnu2024024563/pslg_artifacts"))
    parser.add_argument("--project-log", type=Path, default=ROOT / "log")
    args = parser.parse_args()
    config, inputs = load_config(ROOT / "config/server/zzz_sampling_generation.json")
    report, archive = collect(config, inputs, args.artifact_root, args.project_log, args.state_job, args.array_job)
    print(f"matrix_complete={report['matrix_complete']} all_gates_passed={report['all_gates_passed']}")
    print(f"Return: {archive}")
