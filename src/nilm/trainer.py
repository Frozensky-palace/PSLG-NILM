"""Seq2Point training loop for the B0-B5 protocol (Phase B1).

All arms share the same model, sampling budget, optimizer and early-stopping
rule; only the training windows differ. Training reads train windows and the
real validation monitor only — test is never touched. Batching gathers windows
with the same sliding-window technique as the CPU smoke, grouped by shard.
"""
from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.nilm.checkpoint import (
    capture_rng_states,
    load_checkpoint,
    restore_rng_states,
    save_checkpoint,
)
from src.nilm.seq2point import Seq2PointCNN
from src.nilm.window_dataset import ShardedWindowDataset

ACTIVE_THRESHOLD_W = 20.0
MODEL_HISTORY_LENGTHS = (2, 12, 50)


def ensure_deterministic_cuda_env() -> None:
    """CuBLAS needs a fixed workspace for deterministic algorithms.

    ``torch.use_deterministic_algorithms(True)`` makes every CuBLAS call
    raise on CUDA >= 10.2 unless ``CUBLAS_WORKSPACE_CONFIG`` was set before
    the process started. Set it at import time so any entry point that
    imports this module is safe; never override an explicit user value.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


ensure_deterministic_cuda_env()


def update_early_stopping(val_mae: float, best_val_mae: float,
                          epochs_without_improvement: int
                          ) -> tuple[bool, float, int]:
    """Update early-stopping state without counting a new best as a miss."""
    improved = val_mae < best_val_mae
    if improved:
        return True, val_mae, 0
    return False, best_val_mae, epochs_without_improvement + 1


def gather_windows(dataset: ShardedWindowDataset, indices: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Gather normalized (N, L) windows and normalized targets for indices."""
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
    X = np.empty((len(indices), dataset.window_length), dtype=np.float32)
    y = np.empty(len(indices), dtype=np.float32)
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
            X[positions] = windows
            y[positions] = (
                (appliance[shard_centers].astype(np.float32) - app_cfg["mean"])
                / app_cfg["std"])
    return X, y


def sample_batch_indices(active_pool: np.ndarray, inactive_pool: np.ndarray,
                         batch_size: int, rng: np.random.Generator
                         ) -> np.ndarray:
    """Half active, half inactive; pools sampled with replacement if needed."""
    half = batch_size // 2
    active = rng.choice(active_pool, size=half, replace=len(active_pool) < half)
    inactive = rng.choice(inactive_pool, size=batch_size - half,
                          replace=len(inactive_pool) < batch_size - half)
    return np.sort(np.concatenate([active, inactive]))


def evaluate_validation_mae(dataset: ShardedWindowDataset, indices: np.ndarray,
                            model: nn.Module, app_norm: dict,
                            batch_size: int = 2048) -> tuple[float, np.ndarray, np.ndarray]:
    """MAE in watts plus raw arrays, on real validation windows."""
    model.eval()
    predictions = np.empty(len(indices), dtype=np.float32)
    targets = np.empty(len(indices), dtype=np.float32)
    device = next(model.parameters()).device
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            chunk = indices[start:start + batch_size]
            X, y = gather_windows(dataset, chunk)
            pred = model(torch.from_numpy(X).to(device)).detach().cpu().numpy(
            ).astype(np.float32)
            predictions[start:start + len(chunk)] = pred
            targets[start:start + len(chunk)] = y
    mae_w = float(np.mean(np.abs(
        (predictions - targets) * app_norm["std"])))
    return mae_w, predictions, targets


def train_seq2point(experiment_dir: Path, output_dir: Path, arm: str,
                    seed: int, batch_size: int, steps_per_epoch: int,
                    max_epochs: int, patience: int, learning_rate: float,
                    validation_count: int, resume: bool = False,
                    device_name: str = "auto") -> dict:
    """Train one arm end to end; returns the summary dict written to disk."""
    experiment_dir = Path(experiment_dir)
    output_dir = Path(output_dir)
    if arm not in ("B0", "B1", "B2"):
        raise ValueError(f"unsupported arm {arm}; test is not an arm")
    torch.manual_seed(seed)
    ensure_deterministic_cuda_env()
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(seed)
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "arm": arm, "seed": seed, "batch_size": batch_size,
        "steps_per_epoch": steps_per_epoch, "max_epochs": max_epochs,
        "patience": patience, "learning_rate": learning_rate,
        "validation_count": validation_count,
        "experiment_dir": experiment_dir.resolve().as_posix(),
        "active_threshold_w": ACTIVE_THRESHOLD_W,
        "device": str(device),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8")

    dataset = ShardedWindowDataset(experiment_dir, arm=arm, partition="train")
    validation = ShardedWindowDataset(
        experiment_dir, arm="B0", partition="validation")
    monitor_path = experiment_dir / "validation_monitor_indices.npy"
    monitor = np.load(monitor_path)
    validation_indices = monitor[:min(validation_count, len(monitor))]
    app_norm = dataset.normalization["appliance_w"]

    model = Seq2PointCNN(dataset.window_length).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    start_epoch = 1
    best_val_mae = float("inf")
    history: list[dict] = []
    checkpoint_path = output_dir / "last_checkpoint.pt"
    if resume and checkpoint_path.exists():
        payload = load_checkpoint(checkpoint_path, model=model,
                                  optimizer=optimizer,
                                  map_location=device)
        start_epoch = payload["epoch"] + 1
        best_val_mae = payload["best_val_mae"]
        history = payload["config"].get("_history", [])
        restore_rng_states(payload["rng_states"])
        print(f"[train] resumed from epoch {payload['epoch']}", flush=True)

    active_pool = np.load(experiment_dir / f"{arm.lower()}_train_active_indices.npy")
    inactive_pool = np.load(experiment_dir / f"{arm.lower()}_train_inactive_indices.npy")
    print(f"[train] arm={arm} train pools: active={len(active_pool):,} "
          f"inactive={len(inactive_pool):,}", flush=True)

    best_epoch = 0
    epochs_without_improvement = 0
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "deterministic_algorithms": True,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda" else None,
    }
    started = time.time()
    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        losses = []
        for _ in range(steps_per_epoch):
            batch = sample_batch_indices(active_pool, inactive_pool,
                                         batch_size, rng)
            X, y = gather_windows(dataset, batch)
            optimizer.zero_grad()
            loss = loss_fn(model(torch.from_numpy(X).to(device)),
                           torch.from_numpy(y).to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        val_mae, _, _ = evaluate_validation_mae(
            validation, validation_indices, model, app_norm)
        history.append({"epoch": epoch, "train_loss_mean": float(
            np.mean(losses)), "val_mae_w": val_mae})
        marker = ""
        improved, new_best_val_mae, new_epochs_without_improvement = (
            update_early_stopping(
                val_mae, best_val_mae, epochs_without_improvement))
        if improved:
            best_val_mae, best_epoch = new_best_val_mae, epoch
            marker = " *best*"
            save_checkpoint(output_dir / "best_checkpoint.pt", model=model,
                            optimizer=optimizer, epoch=epoch,
                            best_val_mae=best_val_mae, config=config,
                            rng_states=capture_rng_states())
        save_checkpoint(output_dir / "last_checkpoint.pt", model=model,
                        optimizer=optimizer, epoch=epoch,
                        best_val_mae=best_val_mae, config={
                            **config, "_history": history[-50:]},
                        rng_states=capture_rng_states())
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8")
        print(f"[train] epoch {epoch}: loss={np.mean(losses):.5f} "
              f"val_mae={val_mae:.3f}W{marker}", flush=True)
        epochs_without_improvement = new_epochs_without_improvement
        if not improved:
            if epochs_without_improvement >= patience:
                print(f"[train] early stop: no improvement for {patience} "
                      "epochs", flush=True)
                break

    summary = {
        "protocol": "seq2point_training_v1",
        "arm": arm,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_mae_w": best_val_mae,
        "epochs_run": len(history),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "environment": environment,
        "test_accessed": False,
        "wall_seconds": time.time() - started,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    dataset.close()
    validation.close()
    return summary
