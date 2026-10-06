"""Inspect supplied server paths before selecting raw UK-DALE channels.

Standard library only; reads labels/metadata and at most three short lines of
candidate data files. Does not scan full waveforms, unzip, install or train.
Historical aligned inputs are located by bounded filename search, not opened.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.inspect_server_setup import inspect_setup, path_info, save_report


def bounded_text(path, limit=32768):
    try:
        with open(path, encoding="utf-8", errors="replace") as stream:
            text = stream.read(limit + 1)
        return {"path": str(path), "text": text[:limit], "truncated": len(text) > limit}
    except OSError as error:
        return {"path": str(path), "error_type": type(error).__name__}


def file_head(path):
    result = path_info(path)
    if result.get("kind") == "file":
        try:
            with open(path, encoding="utf-8", errors="replace") as stream:
                result["first_lines"] = [stream.readline(1024).rstrip() for _ in range(3)]
        except OSError as error:
            result["read_error"] = type(error).__name__
    return result


def locate_manifests(roots, max_directories=3000):
    targets = {"aligned_partition_manifest.json", "cycle_inventory_with_split.csv",
               "dataset_manifest.json", "real_cycle_library_manifest.json", "ukdale.h5"}
    found, visited = [], 0
    errors = []
    for root in roots:
        path = Path(root)
        if not path.is_dir():
            continue
        for current, directories, files in os.walk(path, followlinks=False,
                                                  onerror=lambda e: errors.append(type(e).__name__)):
            visited += 1
            if visited > max_directories:
                return {"files": found, "truncated": True, "visited_directories": visited - 1,
                        "errors": errors}
            depth = len(Path(current).relative_to(path).parts)
            directories[:] = sorted(d for d in directories if d not in
                                     {".git", ".cache", "node_modules", "__pycache__", ".venv"}) if depth < 8 else []
            found.extend(str(Path(current) / name) for name in sorted(files) if name in targets)
    return {"files": found, "truncated": False, "visited_directories": visited, "errors": errors}


def inspect_raw(config):
    data = Path(config["data_root"])
    house = data / f"house_{int(config['building'])}"
    labels = bounded_text(house / "labels.dat")
    candidates = {"channel_5.dat", "channel_54.dat", "mains.dat"}
    matches = []
    for line in labels.get("text", "").splitlines():
        match = re.fullmatch(r"\s*(\d+)\s+(.+)", line)
        if match and any(term in match[2].lower().replace("_", " ")
                         for term in ("washing", "washer", "mains", "aggregate", "whole house")):
            candidates.add(f"channel_{match[1]}.dat")
            matches.append({"channel": int(match[1]), "label": match[2]})
    try:
        names = sorted(item.name for item in house.iterdir())
    except OSError:
        names = []
    return {
        "data_root": path_info(data), "house": path_info(house), "labels": labels,
        "house_filenames": names[:200], "filenames_truncated": len(names) > 200,
        "label_matches": matches, "channel_selection": "not_automatically_selected",
        "candidate_heads": [file_head(house / name) for name in sorted(candidates)],
        "metadata": [bounded_text(data / "metadata" / name) for name in
                     (f"building{int(config['building'])}.yaml", "meter_devices.yaml")],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/server/zzz_sampling_control.json")
    parser.add_argument("--output-dir", default="log/sampling_control_setup")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    report = {"status": "channel_and_environment_inventory_only", "config": config,
              "raw_data": inspect_raw(config),
              "historical_inputs": locate_manifests(config["discovery_roots"]),
              "environment": inspect_setup(config["project_root"], [config["data_root"]])}
    try:
        result = subprocess.run(["sinfo", "-h", "-o", "%P %a %G %l"],
                                capture_output=True, text=True, timeout=15)
        report["slurm_partitions"] = {"returncode": result.returncode,
                                      "stdout": result.stdout[:12000], "stderr": result.stderr[:1000]}
    except (OSError, subprocess.TimeoutExpired) as error:
        report["slurm_partitions"] = {"error_type": type(error).__name__}
    output = save_report(report, args.output_dir)
    print(f"[sampling-control] saved: {output}")
    print(f"[sampling-control] labels: {report['raw_data']['label_matches']}")
    print(f"[sampling-control] historical inputs found: {len(report['historical_inputs']['files'])}")
    print("[sampling-control] No channel has been selected; no dataset or environment was modified.")


if __name__ == "__main__":
    main()
