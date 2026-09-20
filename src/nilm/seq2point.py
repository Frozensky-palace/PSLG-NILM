"""Fixed Seq2Point regression network for the B0-B5 protocol.

The architecture is frozen: every comparison arm uses the same parameter count
and layer stack, so downstream differences come from the training data only.
Input is one normalized mains window (1, 599); the output is the normalized
appliance power at the window center.
"""
from __future__ import annotations

import torch
from torch import nn

WINDOW_LENGTH = 599


class Seq2PointCNN(nn.Module):
    """All-conv encoder plus a small regression head."""

    def __init__(self, window_length: int = WINDOW_LENGTH):
        super().__init__()
        self.window_length = int(window_length)
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=2, padding=4),
            nn.ReLU(),
            nn.Conv1d(32, 48, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(48, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 80, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(80, 96, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.window_length)
            flat = self.encoder(dummy).flatten(1).shape[1]
        self.head = nn.Sequential(
            nn.Linear(flat, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.flat_features = int(flat)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        """``windows``: (batch, window_length) normalized mains."""
        encoded = self.encoder(windows.unsqueeze(1))
        return self.head(encoded.flatten(1)).squeeze(1)


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))
