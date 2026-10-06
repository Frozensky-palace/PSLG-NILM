"""PSLG-NILM-ADVANCED entry point.

One linear engine. Step selection and model/method variants are chosen via CLI
arguments; fixed parameters live in ``config/config.yaml``. No per-trajectory
RunKey cache, no ``.done`` skip-flags, no sed-edited temporary configs — slurm
scripts just loop over these CLI arguments.

Examples:
    # run everything implemented for the default appliance
    python main.py --steps all

    # compare two segmentation methods; feature extraction is cached (M2), so
    # the second invocation reuses detsec features produced for `clasp`
    python main.py --steps extract,segment,feature --segment-method clasp --feature-model detsec --run-id exp1
    python main.py --steps segment,feature             --segment-method fluss --feature-model detsec --run-id exp1

    # re-run ONLY feature extraction against an existing run's manifest
    python main.py --steps feature --segment-method clasp --feature-model bilstm_ae --run-id exp1
"""
from __future__ import annotations

import os
import sys

# Set threading / library env before heavy imports to avoid segfaults from
# conflicting threading layers (stumpy/numba vs tensorflow).
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

import argparse
import datetime

import yaml

# Make the project root and the vendored model packages importable.
project_root = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(project_root, "models")
for _p in (project_root, models_dir,
           os.path.join(models_dir, "time_segmentation"),
           os.path.join(models_dir, "feature_extract")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Canonical step order — all steps are implemented.
ALL_STEP_ORDER = ["extract", "segment", "feature", "cluster", "state_merge", "fewshot", "pam", "split"]
IMPLEMENTED_STEPS = ALL_STEP_ORDER


def parse_steps(spec: str, available):
    """Parse a --steps value into an ordered list of step ids.

    ``spec`` may be "all" or a comma list. Unknown ids raise.
    """
    s = (spec or "").strip().lower()
    if s in ("", "all"):
        return [t for t in ALL_STEP_ORDER if t in available]
    out = [t.strip() for t in s.split(",") if t.strip()]
    unknown = [t for t in out if t not in ALL_STEP_ORDER]
    if unknown:
        raise ValueError(f"unknown --steps id(s): {unknown}. known: {ALL_STEP_ORDER}")
    # keep canonical execution order regardless of user ordering
    return [t for t in ALL_STEP_ORDER if t in out]


def parse_int_list(spec: str):
    """Parse '3,4,5' -> [3, 4, 5]; empty/None -> None."""
    s = (spec or "").strip()
    if not s:
        return None
    return [int(t) for t in s.split(",") if t.strip()]


# ── step builders (lazy-import the step classes) ─────────────────────────────

def _build_extract(cfg, sel):
    from src.steps.extract_active_data_step import ExtractActiveDataStep
    c = cfg.get("extract_active_data", {})
    return ExtractActiveDataStep(
        method=c.get("method", "simple"),
        appliance_name=sel["appliance"],
        input_file=sel["raw_series"],
        resample_fs=c.get("resample_fs", 0),
        threshold=c.get("threshold", 5),
        t_drop=c.get("t_drop", 150),
        t_min_work=c.get("t_min_work", 180),
        context_seconds=c.get("context_seconds", 90),
        fs=c.get("fs", 0.1666667),
    )


def _build_segment(cfg, sel):
    from src.steps.time_segmentation import TimeSegmentationStep
    c = cfg.get("time_segmentation", {})
    return TimeSegmentationStep(
        segment_method=sel["segment_method"],
        appliance_name=sel["appliance"],
        window_size=c.get("window_size", 30),
        n_regimes=c.get("n_regimes", 2),
        excl_factor=c.get("excl_factor", 4),
        clasp_n_jobs=c.get("clasp_n_jobs", -1),
        clasp_n_segments=c.get("clasp_n_segments", "learn"),
        max_seg_len=c.get("max_seg_len", 0),
        input_dir=(cfg.get("paths", {}) or {}).get("segments_dir", ""),
    )


def _build_feature(cfg, sel):
    from src.steps.feature_extract_step import FeatureExtractStep
    c = cfg.get("feature_extract", {})
    return FeatureExtractStep(
        model_name=sel["feature_model"],
        segment_method=sel["segment_method"],
        latent_dim=c.get("latent_dim", 16),
        epochs=c.get("epochs", 50),
        batch_size=c.get("batch_size", 32),
        learning_rate=c.get("learning_rate", 0.0001),
        patience=c.get("patience", 5),
        attention_size=c.get("attention_size", 32),
        cache_enabled=bool(c.get("cache", True)),
        embed_dim=c.get("embed_dim", 32),
        lambda_phy=c.get("lambda_phy", 0.1),
        nonneg_channels=c.get("nonneg_channels", [0, 1, 2, 3]),
        norm_mode=c.get("norm_mode", "znorm"),
        embed_proj=c.get("embed_proj", "none"),
        nonneg_activation=c.get("nonneg_activation", "softplus"),
        tf_ratio=c.get("tf_ratio", 1.0),
        tf_schedule=c.get("tf_schedule", "constant"),
    )


def _build_cluster(cfg, sel):
    from src.steps.time_clustering_step import TimeClusteringStep
    c = cfg.get("time_clustering", {})
    ms = c.get("method_specific", {}) or {}
    km, db, hd = ms.get("kmeans", {}), ms.get("dbscan", {}), ms.get("hdbscan", {})
    dk = ms.get("dpc_kmeans", {}) or {}
    return TimeClusteringStep(
        cluster_method=sel["cluster_method"],
        feature_model=sel["feature_model"],
        segment_method=sel["segment_method"],
        n_clusters=sel["n_clusters"],
        metric=c.get("metric", "euclidean"),
        normalization_method=c.get("normalization_method", "zscore"),
        col_index=c.get("col_index", 2),
        kmeans_n_init=km.get("n_init", 30),
        kmeans_max_iter=km.get("max_iter", 300),
        kmeans_random_state=km.get("random_state", 42),
        dbscan_eps=db.get("eps", 1.25),
        dbscan_min_pts=db.get("min_pts", 20),
        hdbscan_min_cluster_size=hd.get("min_cluster_size", 20),
        hdbscan_min_samples=hd.get("min_samples"),
        hdbscan_cluster_selection_method=hd.get("cluster_selection_method", "eom"),
        hdbscan_cluster_selection_epsilon=hd.get("cluster_selection_epsilon", 0.0),
        dpc_percent=dk.get("percent", 2.0),
        dpc_min_dist_tau=dk.get("min_dist_tau"),
        dpc_random_state=dk.get("random_state", 0),
        dpc_k_nn=dk.get("k_nn", 5),
    )


def _build_state_merge(cfg, sel):
    from src.steps.temporal_state_merge_step import TemporalStateMergeStep
    c = cfg.get("temporal_state_merge", {}) or {}
    ex = cfg.get("extract_active_data", {}) or {}
    return TemporalStateMergeStep(
        cluster_method=sel["cluster_method"],
        feature_model=sel["feature_model"],
        segment_method=sel["segment_method"],
        merge_mode=c.get("merge_mode", "same_label_plus_similar"),
        enable_similar_merge=c.get("enable_similar_merge", True),
        similar_feature_tol=c.get("similar_feature_tol", 0.5),
        min_block_seconds=c.get("min_block_seconds", 30.0),
        fs=c.get("fs", ex.get("resample_fs", ex.get("fs", 0.1666667))),
    )


def _build_fewshot(cfg, sel):
    from src.steps.few_shot_cluster_extract_step import FewShotClusterExtractStep
    c = cfg.get("few_shot_cluster_extract", {})
    return FewShotClusterExtractStep(
        cluster_tag=sel["cluster_tag"],
        n_percent=c.get("n_percent", 50),
        adj_threshold=c.get("adj_threshold", 0.6),
        center_margin=c.get("center_margin", 0.1),
        center_support_threshold=c.get("center_support_threshold", 0.6),
        export_format=c.get("export_format", "csv"),
        normalization_method=cfg.get("time_clustering", {}).get("normalization_method", "zscore"),
    )


def _build_pam(cfg, sel):
    from src.steps.primitive_activity_mapping_step import PrimitiveActivityMappingStep
    return PrimitiveActivityMappingStep(cluster_tag=sel["cluster_tag"])


def _build_split(cfg, sel):
    from src.steps.dataset_split_step import DatasetSplitStep
    c = cfg.get("dataset_split", {})
    return DatasetSplitStep(
        # Keep the final split on the exact same branch series selected for
        # extraction.  In particular, honour a CLI --raw-series override.
        raw_series_path=sel["raw_series"],
        mains_series_path=c.get("mains_series"),
        few_train_ratio=c.get("few_train_ratio", 0.5),
        non_few_train_ratio=c.get("non_few_train_ratio", 0.8),
        random_seed=c.get("random_seed", 42),
        timestamp_tolerance_seconds=c.get("timestamp_tolerance_seconds", 0.0),
        clip_negative_mains_to_zero=c.get("clip_negative_mains_to_zero", True),
    )


STEP_BUILDERS = {
    "extract": _build_extract,
    "segment": _build_segment,
    "feature": _build_feature,
    "cluster": _build_cluster,
    "state_merge": _build_state_merge,
    "fewshot": _build_fewshot,
    "pam": _build_pam,
    "split": _build_split,
}


def resolve_selection(args, cfg):
    """Merge CLI overrides over config defaults into a single selection dict."""
    run = cfg.get("run", {}) or {}
    paths = cfg.get("paths", {}) or {}
    appliance = args.appliance or run.get("appliance") or "appliance"
    run_id = (args.run_id or run.get("run_id")
              or datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    return {
        "appliance": appliance,
        "run_id": run_id,
        "raw_series": args.raw_series or paths.get("raw_series", ""),
        "segment_method": args.segment_method or "clasp",
        "feature_model": args.feature_model or "detsec",
        "cluster_method": args.cluster_method or "kmeans",
        "n_clusters": parse_int_list(args.n_clusters) or [3, 4, 5],
        "cluster_tag": args.cluster_tag,
    }


def run(args):
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    sel = resolve_selection(args, cfg)
    selected = parse_steps(args.steps, IMPLEMENTED_STEPS)

    not_impl = [s for s in selected if s not in STEP_BUILDERS]
    if not_impl:
        raise NotImplementedError(
            f"step(s) {not_impl} are scheduled for a later milestone and not built yet. "
            f"Implemented: {sorted(STEP_BUILDERS)}")

    from src.framework.workflow import Workflow
    wf = Workflow(sel["run_id"], sel["appliance"], cfg)
    wf.set_variants(
        segment_method=sel["segment_method"],
        feature_model=sel["feature_model"],
        cluster_method=sel["cluster_method"],
    )

    for step_id in selected:
        wf.add(STEP_BUILDERS[step_id](cfg, sel))

    wf.run()


def main():
    p = argparse.ArgumentParser(
        description="PSLG-NILM-ADVANCED - linear workflow with CLI step/variant selection.")
    p.add_argument("--config", default="config/config.yaml",
                   help="Path to the fixed-parameter config file.")
    p.add_argument("--steps", default="extract,segment,feature",
                   help="Comma list or 'all'. Known order: " + ",".join(ALL_STEP_ORDER))
    p.add_argument("--segment-method", default=None,
                   help="clasp | fluss | espresso | clasp-origin | none (default: clasp)")
    p.add_argument("--feature-model", default=None,
                   help="detsec | detsec_pc | physical_stats | bilstm_ae | lstm_ae | cnn_ae | bilstm_ae_attention | autoencoder | dtw (default: detsec)")
    p.add_argument("--cluster-method", default=None,
                   help="kmeans | kmeans-scan | dpc-kmeans | dpc-kmeans-scan | dbscan | hdbscan (default: kmeans)")
    p.add_argument("--n-clusters", default=None,
                   help="Candidate cluster counts, e.g. '3,4,5'. Every k gets its own tagged result.")
    p.add_argument("--cluster-tag", default=None,
                   help="Which tagged clustering result downstream steps consume, e.g. 'kmeans_k4'.")
    p.add_argument("--appliance", default=None, help="Override run.appliance.")
    p.add_argument("--run-id", default=None, help="Reuse a run directory (enables manifest reuse).")
    p.add_argument("--raw-series", default=None, help="Override paths.raw_series.")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
