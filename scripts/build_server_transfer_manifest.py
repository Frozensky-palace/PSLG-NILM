"""Create checksums for the minimal B0/B1/B2 server training data bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

NILM_BUNDLE_FILES = (
    "dataset_manifest.json", "normalization.json", "window_ranges.csv",
    "b0_train_active_indices.npy", "b0_train_inactive_indices.npy",
    "b1_train_active_indices.npy", "b1_train_inactive_indices.npy",
    "b2_train_active_indices.npy", "b2_train_inactive_indices.npy",
    "validation_monitor_indices.npy",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_nilm_bundle_paths(experiment_dir: Path, root: Path,
                              exclude_test: bool = True
                              ) -> tuple[set[Path], int]:
    """Return project-relative bundle paths plus the skipped shard count."""
    with open(experiment_dir / "dataset_manifest.json", encoding="utf-8") as stream:
        dataset = json.load(stream)
    paths: set[Path] = set()
    skipped_test_shards = 0
    for arm in dataset["sources"].values():
        for partition, shards in arm.items():
            if exclude_test and partition == "test":
                skipped_test_shards += len(shards)
                continue
            for shard in shards:
                paths.add(Path(shard["path"]))
    for name in NILM_BUNDLE_FILES:
        paths.add((experiment_dir / name).resolve().relative_to(root))
    paths.add(Path("config/experiments/nilm_b0_b2_pilot.yaml"))
    return paths, skipped_test_shards


def write_manifest(records: list[dict], output: Path, *, protocol: str,
                   **extra) -> None:
    manifest = {
        "protocol": protocol,
        "project_relative_paths": True,
        "file_count": len(records),
        "total_bytes": sum(item["bytes"] for item in records),
        "files": records,
        **extra,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    with open(output.with_suffix(".txt"), "w", encoding="utf-8") as stream:
        stream.write("\n".join(item["path"] for item in records) + "\n")
    print(f"[bundle] files={len(records)}, total={manifest['total_bytes']:,} bytes")
    print(f"[bundle] manifest -> {output}")


def hash_records(paths: set[Path], root: Path) -> list[dict]:
    records = []
    for relative in sorted(paths, key=lambda item: item.as_posix()):
        path = root / relative
        if not path.exists():
            raise SystemExit(f"missing bundle file: {relative}")
        records.append({
            "path": relative.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
        print(f"[bundle] {relative} ({path.stat().st_size:,} bytes)", flush=True)
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--exclude-test", action="store_true",
                    help="build a train+validation development bundle; the "
                         "final test shards stay physically off the server")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    experiment_dir = Path(args.experiment_dir)
    paths, skipped = collect_nilm_bundle_paths(
        experiment_dir, root, exclude_test=args.exclude_test)
    records = hash_records(paths, root)
    write_manifest(records, Path(args.output),
                   protocol="nilm_server_transfer_bundle_v1",
                   exclude_test=bool(args.exclude_test),
                   skipped_test_shards=skipped)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
