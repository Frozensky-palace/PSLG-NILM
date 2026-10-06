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
                    optimizer: torch.optim.Optimizer | None = None,
                    map_location: str | torch.device = "cpu") -> dict:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload


def capture_rng_states() -> dict:
    states = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["torch_cuda"] = torch.cuda.get_rng_state_all()
    return states


def restore_rng_states(states: dict) -> None:
    if "python" in states:
        random.setstate(states["python"])
    if "numpy" in states:
        np.random.set_state(states["numpy"])
    if "torch" in states:
        torch.set_rng_state(states["torch"].cpu())
    if "torch_cuda" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [state.cpu() for state in states["torch_cuda"]])
