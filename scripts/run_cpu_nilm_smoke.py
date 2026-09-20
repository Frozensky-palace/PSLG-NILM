"""Run a fixed-budget CPU validation smoke test for B0/B1/B2 (no test use)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nilm.metrics import nilm_metrics  # noqa: E402
from src.nilm.window_dataset import ShardedWindowDataset  # noqa: E402


def _features_batch(windows: np.ndarray) -> np.ndarray:
    """Same 14 features as the original per-window ``_features``, vectorized.

    Row-wise reductions over a C-contiguous (N, L) matrix follow the identical
    pairwise-summation paths as the previous 1-D per-window calls, so values
    match the frozen baseline bit for bit.
    """
    x = np.asarray(windows, dtype=np.float32)
    middle = x.shape[1] // 2
    diff = np.diff(x, axis=1)
    return np.column_stack([
        x[:, middle],
        x.mean(axis=1),
        x.std(axis=1),
        x.min(axis=1),
        x.max(axis=1),
        np.quantile(x, 0.10, axis=1),
        np.quantile(x, 0.50, axis=1),
        np.quantile(x, 0.90, axis=1),
        x[:, 0],
        x[:, -1],
        np.abs(diff).mean(axis=1),
        x[:, middle - 2:middle + 3].mean(axis=1),
        x[:, middle - 12:middle + 13].mean(axis=1),
        x[:, middle - 50:middle + 51].mean(axis=1),
    ]).astype(np.float32)


def _extract(dataset: ShardedWindowDataset, indices: np.ndarray,
             *, label: str) -> tuple[np.ndarray, np.ndarray]:
    X = np.empty((len(indices), 14), dtype=np.float32)
    y = np.empty(len(indices), dtype=np.float32)
    indices = np.asarray(indices, dtype=np.int64)
    range_indices = np.searchsorted(dataset.cumulative, indices, side="right")
    previous = np.where(
        range_indices > 0, dataset.cumulative[np.maximum(range_indices - 1, 0)], 0)
    first = dataset.ranges["first_center"].to_numpy(dtype=np.int64)[range_indices]
    strides = dataset.ranges["stride"].to_numpy(dtype=np.int64)[range_indices]
    centers = first + (indices - previous) * strides
    shard_indices = dataset.ranges["shard_index"].to_numpy(dtype=np.int64)[range_indices]
    mains_cfg = dataset.normalization["mains_w"]
    app_cfg = dataset.normalization["appliance_w"]
    completed = 0
    for shard_index in np.unique(shard_indices):
        positions = np.flatnonzero(shard_indices == shard_index)
        path, mains_field, appliance_field = dataset._source(int(shard_index))
        with np.load(path) as data:
            mains = ((data[mains_field].astype(np.float32) - mains_cfg["mean"])
                     / mains_cfg["std"])
            appliance = data[appliance_field]
            shard_centers = centers[positions]
            windows = np.lib.stride_tricks.sliding_window_view(
                mains, dataset.window_length)[shard_centers - dataset.half]
            X[positions] = _features_batch(windows)
            y[positions] = (
                (appliance[shard_centers].astype(np.float32) - app_cfg["mean"])
                / app_cfg["std"])
        completed += len(positions)
        print(f"[cpu-smoke] {label}: {completed:,}/{len(indices):,}", flush=True)
    return X, y


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--train-per-class", type=int, default=25000)
    ap.add_argument("--validation-count", type=int, default=50000)
    args = ap.parse_args()
    experiment_dir = Path(args.experiment_dir)
    output_dir = Path(args.output_dir)
    with open(experiment_dir / "dataset_manifest.json", encoding="utf-8") as stream:
        manifest = json.load(stream)
    monitor = np.load(experiment_dir / manifest["validation_monitor"]["path"])
    validation_indices = monitor[:min(args.validation_count, len(monitor))]
    validation_set = ShardedWindowDataset(
        experiment_dir, arm="B0", partition="validation")
    X_val, y_val_norm = _extract(
        validation_set, validation_indices, label="validation")
    y_val = validation_set.denormalize_target(y_val_norm)
    validation_set.close()

    rng = np.random.default_rng(args.seed)
    results = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for arm in ("B0", "B1", "B2"):
        pool = manifest["train_sampling_pools"][arm]
        active = np.load(experiment_dir / pool["active_path"])
        inactive = np.load(experiment_dir / pool["inactive_path"])
        active_pick = np.sort(rng.choice(
            active, size=args.train_per_class,
            replace=args.train_per_class > len(active)))
        inactive_pick = np.sort(rng.choice(
            inactive, size=args.train_per_class,
            replace=args.train_per_class > len(inactive)))
        indices = np.concatenate([active_pick, inactive_pick])
        # Sort for fast shard access, then use all rows; order does not affect
        # this deterministic tree model.
        indices.sort()
        train_set = ShardedWindowDataset(
            experiment_dir, arm=arm, partition="train")
        X_train, y_train = _extract(train_set, indices, label=f"{arm} train")
        train_set.close()
        model = HistGradientBoostingRegressor(
            loss="squared_error", learning_rate=0.08, max_iter=80,
            max_leaf_nodes=31, l2_regularization=1.0,
            random_state=args.seed)
        model.fit(X_train, y_train)
        pred_norm = model.predict(X_val)
        pred = np.maximum(
            pred_norm * manifest_normalization(experiment_dir)["appliance_w"]["std"]
            + manifest_normalization(experiment_dir)["appliance_w"]["mean"], 0.0)
        metrics = nilm_metrics(y_val, pred, threshold_w=20.0)
        metrics.update({
            "arm": arm,
            "train_samples": int(len(indices)),
            "active_train_samples": int(len(active_pick)),
            "inactive_train_samples": int(len(inactive_pick)),
            "validation_samples": int(len(validation_indices)),
            "model": "HistGradientBoostingRegressor fixed CPU smoke",
            "not_final_seq2point": True,
            "test_accessed": False,
        })
        results[arm] = metrics
        np.savez_compressed(
            output_dir / f"{arm.lower()}_validation_predictions.npz",
            y_true=y_val.astype(np.float32), y_pred=pred.astype(np.float32),
            validation_indices=validation_indices)
        print(f"[cpu-smoke] {arm}: MAE={metrics['mae_w']:.3f}W "
              f"F1={metrics['f1']:.4f}")
    report = {
        "protocol": "cpu_nilm_smoke_v1",
        "purpose": "pipeline sanity check only; not the final Seq2Point result",
        "seed": args.seed,
        "same_training_sample_count": True,
        "same_real_validation_subset": True,
        "test_accessed": False,
        "results": results,
    }
    with open(output_dir / "cpu_smoke_results.json", "w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)


def manifest_normalization(experiment_dir: Path) -> dict:
    with open(experiment_dir / "normalization.json", encoding="utf-8") as stream:
        return json.load(stream)


if __name__ == "__main__":
    main()
