"""Trace historical B0 inputs without opening or hashing waveform arrays.

Collect bounded metadata and source-file existence under configured roots.
Never remap manifests, select a dataset, split data, train or install packages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PureWindowsPath
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.inspect_sampling_control import locate_manifests
from scripts.inspect_server_setup import inspect_setup, path_info, save_report


MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_TOTAL_BYTES = 12 * 1024 * 1024
METADATA_NAMES = {
    "dataset_manifest.json", "normalization.json", "window_ranges.csv",
    "aligned_partition_manifest.json", "cycle_alignment_coverage.csv",
    "cycle_inventory_with_split.csv", "real_cycle_library_manifest.json",
}


class MetadataReader:
    def __init__(self, roots):
        self.roots = [Path(root).resolve() for root in roots]
        self.records = {}
        self.total_bytes = 0

    def permitted(self, path):
        path = Path(path).resolve()
        return any(path.is_relative_to(root) for root in self.roots)

    def read(self, path):
        path = Path(path).resolve()
        key = str(path)
        if key in self.records:
            return self.records[key]
        record = {"path": key}
        self.records[key] = record
        if not self.permitted(path) or path.name not in METADATA_NAMES:
            record["status"] = "outside_metadata_scope"
            return record
        record.update(path_info(path))
        if record.get("kind") != "file":
            return record
        size = record["bytes"]
        if size > MAX_FILE_BYTES or self.total_bytes + size > MAX_TOTAL_BYTES:
            record["status"] = "metadata_size_limit"
            return record
        try:
            with open(path, "rb") as stream:
                raw = stream.read(MAX_FILE_BYTES + 1)
            if len(raw) > MAX_FILE_BYTES or self.total_bytes + len(raw) > MAX_TOTAL_BYTES:
                record["status"] = "metadata_size_limit"
                return record
            self.total_bytes += len(raw)
            record["sha256"] = hashlib.sha256(raw).hexdigest()
            text = raw.decode("utf-8-sig")
            if path.suffix == ".json":
                record["content"] = json.loads(text)
            else:
                record["text"] = text
            record["status"] = "read"
        except (OSError, UnicodeError, ValueError) as error:
            record.update(status="read_error", error=str(error))
        return record


def candidate_repo_roots(config):
    roots = {Path(config["project_root"]).resolve()}
    for item in config["discovery_roots"]:
        path = Path(item).resolve()
        roots.add(path)
        # The junior's guide uses ~/projects/<repo>. Do not walk the whole home.
        if path.name == "projects" and path.is_dir():
            for child in sorted(path.iterdir())[:200]:
                if child.is_dir() and ((child / ".git").exists() or (child / "reports").is_dir()):
                    roots.add(child.resolve())
    return sorted(roots)


def resolve_reference(declared, roots, reader):
    raw = str(declared)
    if PureWindowsPath(raw).drive or "\\" in raw:
        return {"declared_path": raw, "status": "non_posix_reference", "matches": []}
    path = Path(raw)
    if ".." in path.parts:
        return {"declared_path": raw, "status": "parent_reference_rejected", "matches": []}
    candidates = [path] if path.is_absolute() else [root / path for root in roots]
    matches = sorted({str(p.resolve()) for p in candidates
                      if reader.permitted(p) and p.is_file()})
    return {"declared_path": raw,
            "status": "unique" if len(matches) == 1 else "ambiguous" if matches else "unresolved",
            "matches": [path_info(p) for p in matches]}


def trace_sources(config):
    allowed = [*config["discovery_roots"], config["project_root"]]
    reader = MetadataReader(allowed)
    discovery = locate_manifests(config["discovery_roots"])
    roots = candidate_repo_roots(config)
    for found in discovery["files"][:200]:
        if Path(found).name in {"cycle_inventory_with_split.csv", "real_cycle_library_manifest.json"}:
            reader.read(found)
    dataset_paths = sorted({Path(p).resolve() for p in discovery["files"] if Path(p).name == "dataset_manifest.json"})
    aligned_paths = {Path(p).resolve() for p in discovery["files"] if Path(p).name == "aligned_partition_manifest.json"}
    datasets = []
    for path in dataset_paths[:100]:
        record = reader.read(path)
        data = record.get("content")
        if not isinstance(data, dict):
            continue
        for name in ("normalization.json", "window_ranges.csv"):
            reader.read(path.parent / name)
        item = {"manifest": str(path), "sample_seconds": data.get("sample_seconds"),
                "expected_aligned_manifest_sha256": data.get("source_hashes", {}).get("aligned_manifest"),
                "b0_references": []}
        for partition, shards in data.get("sources", {}).get("B0", {}).items():
            for shard in shards:
                ref = resolve_reference(shard["path"], roots, reader)
                ref["partition"] = partition
                item["b0_references"].append(ref)
                for match in ref["matches"]:
                    aligned = Path(match["path"]).parent.parent / "aligned_partition_manifest.json"
                    if aligned.is_file() and reader.permitted(aligned):
                        aligned_paths.add(aligned)
        datasets.append(item)
    aligned = []
    for path in sorted(aligned_paths)[:100]:
        record = reader.read(path)
        data = record.get("content")
        if not isinstance(data, dict):
            continue
        coverage = reader.read(path.parent / "cycle_alignment_coverage.csv")
        inventory_reference = None
        if data.get("source_inventory"):
            inventory_reference = resolve_reference(data["source_inventory"], roots, reader)
            for match in inventory_reference["matches"]:
                inventory = reader.read(match["path"])
                expected = data.get("source_inventory_sha256")
                match["inventory_hash_matches"] = (inventory.get("sha256") == expected
                                                    if expected and inventory.get("sha256") else None)
        aligned.append({"manifest": str(path), "sample_seconds": data.get("sample_seconds"),
                        "sha256": record.get("sha256"), "coverage_status": coverage["status"],
                        "inventory_reference": inventory_reference})
    for item in datasets:
        expected = item["expected_aligned_manifest_sha256"]
        item["hash_matching_aligned_manifests"] = [
            a["manifest"] for a in aligned if expected and a["sha256"] == expected]
    return {
        "status": "source_metadata_only_not_training_readiness",
        "config": config, "discovery": discovery, "reference_roots": list(map(str, roots)),
        "datasets": datasets, "aligned_candidates": aligned,
        "collection_truncated": (len(dataset_paths) > 100 or len(aligned_paths) > 100
                                 or len(discovery["files"]) > 200 or discovery["truncated"]),
        "metadata": list(reader.records.values()), "metadata_bytes_read": reader.total_bytes,
        "limits": ["No NPZ, HDF5, checkpoint or full DAT content opened or hashed.",
                   "Matching manifest hashes do not verify the waveform files themselves.",
                   "Relative paths with multiple matches are not automatically selected.",
                   "Test references are inventoried as paths only; test arrays are not opened."],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/server/zzz_sampling_control.json")
    parser.add_argument("--output-dir", default="log/sampling_source_trace")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    report = trace_sources(config)
    report["environment"] = inspect_setup(config["project_root"], [])
    output = save_report(report, args.output_dir)
    print(f"[source-trace] datasets={len(report['datasets'])} aligned={len(report['aligned_candidates'])}")
    for item in report["datasets"]:
        unresolved = sum(ref["status"] != "unique" for ref in item["b0_references"])
        print(f"[source-trace] {item['manifest']}: unresolved/ambiguous={unresolved}; "
              f"hash-matched manifests={len(item['hash_matching_aligned_manifests'])}")
    print(f"[source-trace] metadata only; report={output}")


if __name__ == "__main__":
    main()
