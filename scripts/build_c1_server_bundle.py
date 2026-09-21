"""Build the C1 server data bundles (roadmap guide §4.4).

Two independent profiles, both train-safe by construction:

``state-discovery``
    Train-only cycle library, segment CSVs, source map, export manifest,
    split/leakage metadata and the frozen state-discovery config. This is
    everything C2 needs; validation/test waveforms are never included.

``nilm-b0b2``
    The existing B0/B1/B2 train+validation window inputs, reusing
    ``build_server_transfer_manifest.collect_nilm_bundle_paths`` so the two
    manifest builders cannot drift apart. Test shards are always excluded.

Every manifest records project-relative paths with per-file bytes and
SHA-256, and asserts that zero staged paths look like test data.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_server_transfer_manifest import (  # noqa: E402
    collect_nilm_bundle_paths,
    hash_records,
    write_manifest,
)

CORE_DIR = Path("reports/core_validation/ukdale_b1_washing_machine")
STATE_DISCOVERY_CONFIG = Path(
    "config/experiments/core_wm_state_discovery_detsec_pc.yaml")


def looks_like_test_path(relative: Path) -> bool:
    parts = [part.lower() for part in relative.parts]
    return "test" in parts


def collect_state_discovery_paths(root: Path,
                                  library_dir: Path) -> set[Path]:
    """Gather the train-only C2 inputs and assert they are actually train."""
    segments_dir = library_dir / "segments"
    map_path = segments_dir / "segment_source_map.csv"
    if not map_path.exists():
        raise SystemExit(f"segment source map not found: {map_path}")
    mapping = pd.read_csv(map_path)
    if set(mapping["partition"].astype(str)) != {"train"}:
        raise SystemExit("state-discovery bundle must be train-only")

    inventory_path = library_dir / "real_cycle_library.csv"
    inventory = pd.read_csv(inventory_path)
    if "partition" in inventory.columns and set(
            inventory["partition"].astype(str)) != {"train"}:
        raise SystemExit("cycle library must be train-only")

    paths: set[Path] = set()
    for relative in (
            inventory_path.name,
            "real_cycle_library_manifest.json",
            "segments/segment_source_map.csv",
            "segments/segment_export_manifest.json",
    ):
        candidate = library_dir / relative
        if not candidate.exists():
            raise SystemExit(f"missing state-discovery input: {candidate}")
        paths.add(candidate.resolve().relative_to(root))
    segment_files = sorted(segments_dir.glob("cycle_*.csv"))
    if not segment_files:
        raise SystemExit(f"no cycle CSVs under {segments_dir}")
    if len(segment_files) != len(mapping):
        raise SystemExit(
            f"segment files ({len(segment_files)}) and source map rows "
            f"({len(mapping)}) disagree")
    for segment in segment_files:
        paths.add(segment.resolve().relative_to(root))

    # Full-cycle waveforms: build_state_library.py reloads them to compute
    # per-state statistics. Missed in the first bundle (2026-09-21 C1 smoke
    # caught it: FileNotFoundError cycles/cycle_0000.npz on the server).
    cycles_dir = library_dir / "cycles"
    cycle_npz = sorted(cycles_dir.glob("cycle_*.npz"))
    if not cycle_npz:
        raise SystemExit(f"no cycle NPZ files under {cycles_dir}")
    recorded = dict(zip(inventory["path"].astype(str),
                        inventory["sha256"].astype(str)))
    if len(cycle_npz) != len(recorded):
        raise SystemExit(
            f"cycle NPZ count ({len(cycle_npz)}) and inventory rows "
            f"({len(recorded)}) disagree")
    for cycle in cycle_npz:
        relative = cycle.resolve().relative_to(root)
        library_relative = cycle.relative_to(library_dir).as_posix()
        if library_relative not in recorded:
            raise SystemExit(
                f"cycle file {cycle.name} missing from inventory; "
                "bundle must stay exactly train-only")
        digest = hash_records([relative], root)[0]["sha256"]
        if digest != recorded[library_relative]:
            raise SystemExit(
                f"cycle file {cycle.name} sha256 disagrees with inventory")
        paths.add(relative)

    # Train-only provenance metadata from the split layer.
    for relative in (
            "partition_manifest.json", "leakage_report.json",
            "cycle_inventory_with_split.csv",
    ):
        candidate = root / CORE_DIR / relative
        if candidate.exists():
            paths.add(candidate.resolve().relative_to(root))

    # Code and config always come from the repository itself, independent of
    # where the data library lives.
    for repo_relative in (
            STATE_DISCOVERY_CONFIG,
            Path("scripts/train_state_discovery.py"),
            Path("scripts/build_state_library.py"),
    ):
        candidate = PROJECT_ROOT / repo_relative
        if not candidate.exists():
            raise SystemExit(f"missing repo file required by C2: {candidate}")
        paths.add(candidate.resolve().relative_to(PROJECT_ROOT))
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", required=True,
                    choices=("state-discovery", "nilm-b0b2"))
    ap.add_argument("--output", required=True,
                    help="manifest JSON path, e.g. manifests/c1_..._manifest.json")
    ap.add_argument("--experiment-dir", default=None,
                    help="nilm-b0b2 only: prepared NILM inputs directory")
    ap.add_argument("--library-dir", default=None,
                    help="state-discovery only: train cycle library directory")
    args = ap.parse_args()
    root = PROJECT_ROOT

    if args.profile == "state-discovery":
        library_dir = Path(args.library_dir) if args.library_dir else (
            root / CORE_DIR / "real_cycle_library_train_v1")
        paths = collect_state_discovery_paths(root, library_dir)
        skipped = 0
        protocol = "c1_state_discovery_trainonly_bundle_v1"
        extra = {"train_only": True}
    else:
        if not args.experiment_dir:
            raise SystemExit("--profile nilm-b0b2 requires --experiment-dir")
        paths, skipped = collect_nilm_bundle_paths(
            Path(args.experiment_dir), root, exclude_test=True)
        protocol = "c1_nilm_b0_b2_trainval_bundle_v1"
        extra = {"exclude_test": True}

    test_paths = sorted(path.as_posix() for path in paths
                        if looks_like_test_path(path))
    if test_paths:
        raise SystemExit(
            f"refusing to bundle test-looking paths: {test_paths[:5]}")

    records = hash_records(paths, root)
    write_manifest(records, Path(args.output), protocol=protocol,
                   test_path_count=0, skipped_test_shards=skipped, **extra)


if __name__ == "__main__":
    main()
