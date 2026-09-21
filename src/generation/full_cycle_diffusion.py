"""B3-D: conditional denoising diffusion for full cycles (roadmap D4).

A compact DDPM over bucketed waveform tensors with the same condition
treatment as the other routes. Training predicts the added noise; sampling
runs the standard ancestral loop with a linear beta schedule. The model is
intentionally small — the route is the high-cost baseline, so the local
smoke only has to prove correctness, not quality.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from src.generation.full_cycle_cvae import (
    BUCKET_MULTIPLE,
    LengthBucketizer,
    pad_and_mask,
)


def linear_beta_schedule(n_steps: int, beta_start: float = 1e-4,
                         beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, n_steps)


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10000.0) * torch.arange(half, dtype=torch.float32,
                                          device=timesteps.device) / half)
    angles = timesteps.float()[:, None] * frequencies[None, :]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


class DiffusionDenoiser(nn.Module):
    """Condition + noisy wave + timestep embedding -> predicted noise."""

    def __init__(self, wave_length: int, condition_dim: int, width: int = 32):
        super().__init__()
        if wave_length % BUCKET_MULTIPLE:
            raise ValueError(
                f"wave_length must be a multiple of {BUCKET_MULTIPLE}")
        self.time_dim = 32
        reduced = wave_length // 8
        self._width2 = width * 2
        self._reduced = reduced
        self.down = nn.Sequential(
            nn.Conv1d(1, width, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv1d(width, width * 2, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv1d(width * 2, width * 2, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Flatten(),
        )
        self.fc = nn.Linear(width * 2 * reduced + condition_dim
                            + self.time_dim, width * 2 * reduced)
        self.up = nn.Sequential(
            nn.ConvTranspose1d(width * 2, width, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose1d(width, width, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose1d(width, width, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv1d(width, 1, 5, padding=2),
        )

    def forward(self, noisy_wave: torch.Tensor, timesteps: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        hidden = self.down(noisy_wave)
        time_embedding = sinusoidal_embedding(timesteps, self.time_dim)
        joined = torch.cat([hidden, cond, time_embedding], dim=1)
        return self.up(self.fc(joined).view(-1, self._width2, self._reduced))


class GaussianDiffusion:
    """Linear-schedule DDPM around a denoiser."""

    def __init__(self, n_steps: int = 500, beta_start: float = 1e-4,
                 beta_end: float = 0.02):
        self.betas = linear_beta_schedule(n_steps, beta_start, beta_end)
        self.alphas_cumprod = torch.cumprod(1.0 - self.betas, dim=0)

    def add_noise(self, wave: torch.Tensor, timesteps: torch.Tensor
                  ) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(wave)
        alphas = self.alphas_cumprod.to(wave.device)[timesteps]
        scaled = alphas[:, None, None].sqrt()
        noisy = scaled * wave + (1.0 - scaled).sqrt() * noise
        return noisy, noise

    @torch.no_grad()
    def sample(self, denoiser: DiffusionDenoiser, shape: tuple[int, int],
               cond: torch.Tensor, device: str) -> torch.Tensor:
        denoiser.eval()
        wave = torch.randn(shape, device=device)
        for step in reversed(range(len(self.betas))):
            timesteps = torch.full((shape[0],), step, dtype=torch.long,
                                   device=device)
            predicted = denoiser(wave, timesteps, cond)
            beta = self.betas.to(device)[step]
            alpha = (1.0 - beta).sqrt()
            alpha_cumprod = self.alphas_cumprod.to(device)[step]
            wave = (wave - beta / (1 - alpha_cumprod).sqrt() * predicted) \
                / alpha
            if step > 0:
                wave = wave + beta.sqrt() * torch.randn_like(wave)
        return wave


def train_diffusion(denoiser: DiffusionDenoiser,
                    diffusion: GaussianDiffusion,
                    waves: list[np.ndarray], conditions: np.ndarray,
                    bucketizer: LengthBucketizer, *, epochs: int,
                    batch_size: int = 32, learning_rate: float = 1e-3,
                    seed: int = 0, device: str = "cpu") -> list[dict]:
    """Noise-prediction training; waves already divided by power scale."""
    torch.manual_seed(seed)
    denoiser.to(device)
    optimizer = torch.optim.Adam(denoiser.parameters(), lr=learning_rate)
    bucket_length = bucketizer.bucket_length(
        bucketizer.encode(max(len(w) for w in waves)))
    padded, _ = pad_and_mask(waves, bucket_length)
    cond = torch.from_numpy(np.stack(conditions).astype(np.float32))
    n = len(waves)
    generator = torch.Generator().manual_seed(seed)
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        order = torch.randperm(n, generator=generator)
        totals, batches = 0.0, 0
        for start in range(0, n, batch_size):
            picked = order[start:start + batch_size]
            wave = padded[picked].to(device)
            cnd = cond[picked].to(device)
            timesteps = torch.randint(
                0, len(diffusion.alphas_cumprod), (len(wave),),
                generator=generator)
            noisy, noise = diffusion.add_noise(wave, timesteps.to(device))
            optimizer.zero_grad()
            predicted = denoiser(noisy, timesteps.to(device), cnd)
            loss = torch.nn.functional.mse_loss(predicted, noise)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 5.0)
            optimizer.step()
            totals += float(loss.detach())
            batches += 1
        history.append({"epoch": epoch, "noise_mse": totals
                        / max(batches, 1)})
    return history
