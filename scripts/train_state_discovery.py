"""Formal train-only state discovery entry (Phase C2).

Reads train cycle segments, runs the existing workflow steps (prim-glr
segmentation -> detsec_pc or other feature extraction -> KMeans over
k=3/4/5 -> temporal state merge) and exports one traceable state library per
k plus a provenance summary (config hash, seed, git commit, input hashes).
The segment source map must be train-only; validation/test data are never
read. ``--smoke-cycles N`` restricts the run to the first N cycles for a tiny
local end-to-end check; such runs are marked ``status: smoke`` and must never
be frozen.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Mirror main.py's environment guards and import paths before heavy imports.
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
for _path in (str(PROJECT_ROOT), str(MODELS_DIR),
              str(MODELS_DIR / "time_segmentation"),
              str(MODELS_DIR / "feature_extract")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from scripts.build_state_library import (  # noqa: E402
    build_state_library_from_manifest,
)
from src.generation.provenance import (  # noqa: E402
    canonical_config_hash,
    git_commit,
    sha256_of_file,
    utc_now_string,
)

SEGMENT_MAP_NAME = "segment_source_map.csv"


def load_segment_map(segments_dir: Path, map_path: Path | None) -> tuple[pd.DataFrame, Path]:
    """Load the segment source map and enforce the train-only guarantee."""
    resolved = Path(map_path) if map_path else Path(segments_dir) / SEGMENT_MAP_NAME
    if not resolved.exists():
        raise SystemExit(f"segment source map not found: {resolved}")
    mapping = pd.read_csv(resolved)
    if "partition" not in mapping.columns:
        raise SystemExit(f"{resolved} has no partition column")
    partitions = set(mapping["partition"].astype(str))
    if partitions != {"train"}:
        raise SystemExit(
            f"state discovery must be train-only; map contains {sorted(partitions)}")
    return mapping, resolved


def build_smoke_segments(source_segments_dir: Path, destination: Path,
                         n_cycles: int, mapping: pd.DataFrame) -> Path:
    """Copy the first ``n_cycles`` cycle CSVs and their map rows verbatim.

    Filenames keep their original indices, and the workflow assigns csv_idx by
    sorted filename order, so the subset map stays aligned with the blocks.
    """
    source_segments_dir = Path(source_segments_dir)
    destination = Path(destination)
    subset = mapping[mapping["csv_idx"] < n_cycles].sort_values("csv_idx")
    if len(subset) < n_cycles:
        raise SystemExit(
            f"smoke requested {n_cycles} cycles but the map only covers "
            f"{len(subset)}")
    destination.mkdir(parents=True, exist_ok=True)
    for row in subset.itertuples(index=False):
        source_file = source_segments_dir / str(row.filename)
        if not source_file.exists():
            raise SystemExit(f"missing segment file: {source_file}")
        shutil_copy(source_file, destination / str(row.filename))
    subset.to_csv(destination / SEGMENT_MAP_NAME, index=False)
    return destination / SEGMENT_MAP_NAME


def shutil_copy(source: Path, destination: Path) -> None:
    import shutil

    shutil.copyfile(source, destination)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config",
                    default="config/experiments/core_wm_state_discovery_detsec_pc.yaml")
    ap.add_argument("--output-root", required=True,
                    help="new directory for the state libraries and summary")
    ap.add_argument("--run-id", required=True,
                    help="workflow run id (log/<run-id>/run_manifest.json)")
    ap.add_argument("--k", default="3,4,5", help="comma list, e.g. 3,4,5")
    ap.add_argument("--feature-model", default=None,
                    help="override config/state_discovery feature_model")
    ap.add_argument("--segment-method", default=None,
                    help="override config/state_discovery segment_method")
    ap.add_argument("--seed", type=int, default=None,
                    help="override the KMeans random_state from the config")
    ap.add_argument("--epochs-override", type=int, default=None,
                    help="override feature_extract.epochs (smoke runs)")
    ap.add_argument("--smoke-cycles", type=int, default=0,
                    help="use only the first N train cycles (0 = formal run)")
    ap.add_argument("--cycle-library-dir", default=None,
                    help="override config/state_discovery cycle_library_dir")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream) or {}
    discovery = cfg.setdefault("state_discovery", {})
    k_candidates = [int(value) for value in str(args.k).split(",") if value.strip()]
    feature_model = args.feature_model or discovery.get(
        "feature_model", "detsec_pc")
    segment_method = args.segment_method or discovery.get(
        "segment_method", "prim-glr")
    cluster_method = discovery.get("cluster_method", "kmeans")
    sample_seconds = int(discovery.get("sample_seconds", 6))
    seed = args.seed if args.seed is not None else int(
        discovery.get("seed", 42))
    appliance = (cfg.get("run", {}) or {}).get("appliance", "washing_machine")
    cycle_library_dir = Path(args.cycle_library_dir
                             or discovery["cycle_library_dir"])
    segments_dir = Path(cfg.get("paths", {}).get("segments_dir", ""))

    output_root = Path(args.output_root)
    if output_root.exists():
        raise SystemExit(
            f"refusing to overwrite an existing output directory: {output_root}")

    mapping, map_path = load_segment_map(segments_dir, None)
    smoke = args.smoke_cycles > 0
    if smoke:
        subset_map = build_smoke_segments(
            segments_dir, output_root / "segments_smoke_subset",
            args.smoke_cycles, mapping)
        segments_dir = subset_map.parent
        map_path = subset_map
    cfg.setdefault("paths", {})["segments_dir"] = str(segments_dir)
    if args.epochs_override is not None:
        cfg.setdefault("feature_extract", {})["epochs"] = args.epochs_override
        cfg["feature_extract"]["cache"] = False
    cfg.setdefault("time_clustering", {}).setdefault(
        "method_specific", {}).setdefault("kmeans", {})["random_state"] = seed

    status = (f"smoke_{feature_model}" if smoke
              else f"formal_candidate_{feature_model}")
    head_commit = git_commit(PROJECT_ROOT) or "unknown"
    config_hash = canonical_config_hash(cfg)

    # Reuse main.py's step builders so the formal pipeline cannot drift from
    # the workflow definition used everywhere else.
    from main import _build_cluster, _build_feature, _build_segment, \
        _build_state_merge
    from src.framework.workflow import Workflow

    selection = {
        "appliance": appliance,
        "segment_method": segment_method,
        "feature_model": feature_model,
        "cluster_method": cluster_method,
        "n_clusters": k_candidates,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"[state-discovery] run_id={args.run_id} feature={feature_model} "
          f"segment={segment_method} k={k_candidates} seed={seed} "
          f"smoke_cycles={args.smoke_cycles}", flush=True)
    workflow = Workflow(args.run_id, appliance, cfg)
    workflow.set_variants(segment_method=segment_method,
                          feature_model=feature_model,
                          cluster_method=cluster_method)
    for builder in (_build_segment, _build_feature, _build_cluster,
                    _build_state_merge):
        workflow.add(builder(cfg, selection))
    workflow.run()

    run_manifest_path = PROJECT_ROOT / "log" / args.run_id / "run_manifest.json"
    if not run_manifest_path.exists():
        raise SystemExit(f"workflow did not produce {run_manifest_path}")
    libraries = {}
    for k in k_candidates:
        tag = f"{cluster_method}_k{k}_merged"
        summary = build_state_library_from_manifest(
            run_manifest_path=run_manifest_path,
            cluster_tag=tag,
            cycle_library_dir=cycle_library_dir,
            segment_map_path=map_path,
            output_dir=output_root / f"state_library_k{k}",
            run_id=args.run_id,
            sample_seconds=sample_seconds,
            status=status,
            feature_model=feature_model,
            segment_method=segment_method,
            config_hash=config_hash,
            git_commit=head_commit,
        )
        if summary["state_blocks"] == 0:
            raise RuntimeError(f"{tag} produced zero state blocks; "
                               "segmentation or clustering likely failed")
        libraries[tag] = summary

    provenance = {
        "protocol": "state_discovery_v1",
        "status": "smoke" if smoke else "formal_candidate",
        "library_status": status,
        "run_id": args.run_id,
        "git_commit": head_commit,
        "seed": seed,
        "feature_model": feature_model,
        "segment_method": segment_method,
        "cluster_method": cluster_method,
        "k_candidates": k_candidates,
        "smoke_cycles": args.smoke_cycles,
        "train_only": True,
        "test_accessed": False,
        "config": cfg,
        "config_hash": config_hash,
        "input_hashes": {
            "segment_source_map_sha256": sha256_of_file(map_path),
            "segment_export_manifest_sha256": (
                sha256_of_file(segments_dir / "segment_export_manifest.json")
                if (segments_dir / "segment_export_manifest.json").exists()
                else None),
            "cycle_library_inventory_sha256": sha256_of_file(
                cycle_library_dir / "real_cycle_library.csv"),
        },
        "cycles_used": (args.smoke_cycles if smoke
                        else int(mapping["cycle_id"].nunique())),
        "libraries": libraries,
        "updated_utc": utc_now_string(),
    }
    summary_path = output_root / "discovery_summary.json"
    with open(summary_path, "w", encoding="utf-8") as stream:
        json.dump(provenance, stream, ensure_ascii=False, indent=2)
    print(f"[state-discovery] libraries -> {output_root}")
    print(f"[state-discovery] summary -> {summary_path}")
    print(f"[state-discovery] train_only=True test_accessed=False "
          f"status={provenance['status']}")


if __name__ == "__main__":
    main()
