"""Atomic checkpoint save/load with full RNG state for exact resume."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def save_checkpoint(path: Path, *, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer, epoch: int,
                    best_val_mae: float, config: dict,
                    rng_states: dict | None = None) -> None:
    """Write ``<name>.tmp`` first; a failed save never corrupts an old file."""
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_val_mae": float(best_val_mae),
        "config": config,
        "rng_states": rng_states or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path, *, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer | None = None) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload


def capture_rng_states() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }


def restore_rng_states(states: dict) -> None:
    if "python" in states:
        random.setstate(states["python"])
    if "numpy" in states:
        np.random.set_state(states["numpy"])
    if "torch" in states:
        torch.set_rng_state(states["torch"].cpu())
