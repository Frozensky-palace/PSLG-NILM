"""Conditional waveform CVAE shared by B3-V (full cycles) and B4 (primitives).

The waveforms of one bucket are padded to the bucket length; a mask keeps
padding out of the reconstruction loss. Everything here is deliberately
small: it must train a smoke run on CPU while remaining a legitimate
baseline when formally trained on the server.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.generation.base_generator import BaseGenerator
from src.generation.schema import StateSegmentRecord, SyntheticCycleRecord

BUCKET_MULTIPLE = 8
CONDITION_FEATURES = 4  # normalized length, mean power, energy, peak


class LengthBucketizer:
    """Quantile length buckets; every bucket length is a BUCKET_MULTIPLE."""

    def __init__(self, boundaries: list[int]):
        self.boundaries = sorted({int(b) for b in boundaries})
        if len(self.boundaries) < 1:
            raise ValueError("need at least one bucket boundary")

    @classmethod
    def fit(cls, lengths: np.ndarray, n_buckets: int = 4) -> "LengthBucketizer":
        lengths = np.asarray(lengths, dtype=np.int64)
        quantiles = np.quantile(lengths, np.linspace(0.0, 1.0, n_buckets + 1))
        boundaries = sorted({
            int(np.ceil((q + 7) / BUCKET_MULTIPLE) * BUCKET_MULTIPLE)
            for q in quantiles[1:]
        })
        return cls(boundaries)

    def encode(self, length_samples: int) -> int:
        for index, boundary in enumerate(self.boundaries):
            if length_samples <= boundary:
                return index
        return len(self.boundaries) - 1

    def bucket_length(self, bucket: int) -> int:
        return self.boundaries[min(bucket, len(self.boundaries) - 1)]

    @property
    def n_buckets(self) -> int:
        return len(self.boundaries)

    def to_dict(self) -> dict:
        return {"boundaries": self.boundaries}

    @classmethod
    def from_dict(cls, data: dict) -> "LengthBucketizer":
        return cls(data["boundaries"])


def condition_vector(length_samples: int, power: np.ndarray,
                     sample_seconds: int, mean_power_scale: float,
                     length_scale: int) -> np.ndarray:
    """Normalized [length, mean power, energy, peak] condition vector."""
    power = np.asarray(power, dtype=np.float64)
    mean_power = float(power.mean()) if len(power) else 0.0
    energy = float(power.sum() * sample_seconds / 3600.0)
    peak = float(power.max()) if len(power) else 0.0
    return np.array([
        length_samples / max(length_scale, 1),
        mean_power / max(mean_power_scale, 1e-9),
        energy / max(mean_power_scale * length_scale * sample_seconds
                     / 3600.0, 1e-9),
        peak / max(mean_power_scale * 3.0, 1e-9),
    ], dtype=np.float32)


def pad_and_mask(waves: list[np.ndarray], bucket_length: int
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad waveforms to bucket_length; mask marks real samples."""
    batch = torch.zeros(len(waves), 1, bucket_length, dtype=torch.float32)
    mask = torch.zeros(len(waves), bucket_length, dtype=torch.float32)
    for index, wave in enumerate(waves):
        trimmed = np.asarray(wave[:bucket_length], dtype=np.float32)
        batch[index, 0, :len(trimmed)] = torch.from_numpy(trimmed)
        mask[index, :len(trimmed)] = 1.0
    return batch, mask


class ConditionalWaveformCVAE(nn.Module):
    """Small conv CVAE; ``wave_length`` must be a BUCKET_MULTIPLE."""

    def __init__(self, wave_length: int, condition_dim: int,
                 latent_dim: int = 16, width: int = 32):
        super().__init__()
        if wave_length % BUCKET_MULTIPLE:
            raise ValueError(
                f"wave_length must be a multiple of {BUCKET_MULTIPLE}")
        self.wave_length = wave_length
        self.latent_dim = latent_dim
        reduced = wave_length // 8
        self._width2 = width * 2
        self._reduced = reduced
        self.encoder = nn.Sequential(
            nn.Conv1d(1, width, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(width, width * 2, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(width * 2, width * 2, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(width * 2 * reduced + condition_dim,
                               latent_dim)
        self.fc_logvar = nn.Linear(width * 2 * reduced + condition_dim,
                                   latent_dim)
        self.fc_decode = nn.Linear(latent_dim + condition_dim,
                                   width * 2 * reduced)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(width * 2, width, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(width, width, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(width, width, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(width, 1, 5, padding=2),
        )
        self._cond_dim = condition_dim

    def encode(self, wave: torch.Tensor, cond: torch.Tensor
               ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(wave)
        joined = torch.cat([hidden, cond], dim=1)
        return self.fc_mu(joined), self.fc_logvar(joined)

    def decode(self, latent: torch.Tensor, cond: torch.Tensor
               ) -> torch.Tensor:
        hidden = self.fc_decode(torch.cat([latent, cond], dim=1))
        return self.decoder(hidden.view(-1, self._width2, self._reduced))

    def forward(self, wave: torch.Tensor, cond: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(wave, cond)
        latent = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decode(latent, cond), mu, logvar


def elbo_loss(recon: torch.Tensor, wave: torch.Tensor, mask: torch.Tensor,
              mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0
              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Masked MSE reconstruction + KL; returns (loss, recon, kl)."""
    valid = mask.unsqueeze(1)
    recon_mse = ((recon - wave) ** 2 * valid).sum() / valid.sum().clamp(min=1)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_mse + beta * kl, recon_mse, kl


def train_cvae(model: ConditionalWaveformCVAE,
               waves: list[np.ndarray], conditions: np.ndarray,
               bucketizer: LengthBucketizer, *, epochs: int,
               batch_size: int = 32, learning_rate: float = 1e-3,
               beta: float = 0.5, seed: int = 0,
               device: str = "cpu") -> list[dict]:
    """Train on the given waveforms; returns the per-epoch history.

    ``waves`` are expected already divided by the caller's power scale —
    raw watt values make the reconstruction loss explode.
    """
    torch.manual_seed(seed)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    bucket_length = bucketizer.bucket_length(
        bucketizer.encode(max(len(w) for w in waves)))
    padded, mask = pad_and_mask(waves, bucket_length)
    cond = torch.from_numpy(np.stack(conditions).astype(np.float32))
    n = len(waves)
    generator = torch.Generator().manual_seed(seed)
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        order = torch.randperm(n, generator=generator)
        totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
        batches = 0
        for start in range(0, n, batch_size):
            picked = order[start:start + batch_size]
            wave, msk, cnd = (padded[picked].to(device),
                              mask[picked].to(device),
                              cond[picked].to(device))
            optimizer.zero_grad()
            recon, mu, logvar = model(wave, cnd)
            loss, recon_mse, kl = elbo_loss(recon, wave, msk, mu, logvar,
                                            beta=beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batches += 1
            totals["loss"] += float(loss.detach())
            totals["recon"] += float(recon_mse.detach())
            totals["kl"] += float(kl.detach())
        history.append({"epoch": epoch,
                        **{k: v / max(batches, 1) for k, v in totals.items()}})
    return history


def sample_cvae(model: ConditionalWaveformCVAE, bucketizer: LengthBucketizer,
                conditions: np.ndarray, lengths: list[int], *,
                seed: int, device: str = "cpu") -> list[np.ndarray]:
    """Deterministic sampling: latent ~ N(0,1), trimmed and clipped at 0."""
    torch.manual_seed(seed)
    model.to(device)
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        cond = torch.from_numpy(np.stack(conditions).astype(np.float32)
                                ).to(device)
        latent = torch.randn(len(lengths), model.latent_dim, device=device)
        decoded = model.decode(latent, cond).cpu().numpy()[:, 0, :]
    for row, length in zip(decoded, lengths):
        bucket_length = bucketizer.bucket_length(
            bucketizer.encode(length))
        trimmed = np.asarray(row[:min(length, bucket_length)],
                             dtype=np.float64).clip(min=0.0)
        outputs.append(trimmed)
    return outputs


class CVAESamplingGenerator(BaseGenerator):
    """Sampling wrapper for latent+cond decoders (CVAE; reused by WGAN)."""

    version = "1"

    def __init__(self, model, bucketizer: LengthBucketizer,
                 length_scale: float, mean_power_scale: float,
                 donor_waves: list[np.ndarray], sample_seconds: int = 6,
                 device: str = "cpu", power_scale: float = 1.0,
                 name: str = "b3_cvae"):
        import torch as _torch

        self.name = name
        self.model = model
        self.bucketizer = bucketizer
        self.length_scale = float(length_scale)
        self.mean_power_scale = float(mean_power_scale)
        self.power_scale = float(power_scale)
        self.donor_waves = donor_waves
        self.sample_seconds = int(sample_seconds)
        self.device = device
        self._torch = _torch
        model.to(device)
        model.eval()

    def config(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "buckets": self.bucketizer.to_dict(),
            "length_scale": self.length_scale,
            "mean_power_scale": self.mean_power_scale,
        }

    def generate_cycle(self, synthetic_cycle_id: str, seed: int,
                       rng: np.random.Generator
                       ) -> tuple[np.ndarray, SyntheticCycleRecord]:
        torch = self._torch
        donor_index = int(rng.integers(len(self.donor_waves)))
        donor = self.donor_waves[donor_index]
        condition = condition_vector(
            len(donor), donor, self.sample_seconds, self.mean_power_scale,
            self.length_scale)
        torch.manual_seed(int(rng.integers(1 << 31)))
        latent = torch.randn(1, self.model.latent_dim, device=self.device)
        condition_t = torch.from_numpy(condition).unsqueeze(0).to(self.device)
        with torch.no_grad():
            decoded = self.model.decode(latent, condition_t).cpu().numpy()
        power = np.asarray(decoded[0, 0, :len(donor)],
                           dtype=np.float64).clip(min=0.0) * self.power_scale
        energy = float(power.sum() * self.sample_seconds / 3600.0)
        segment = StateSegmentRecord(
            state_label=0,
            target_duration_seconds=len(donor) * self.sample_seconds,
            actual_duration_seconds=len(power) * self.sample_seconds,
            target_samples=len(donor),
            actual_samples=len(power),
            mean_power_w=float(power.mean()),
            energy_wh=energy,
        )
        record = SyntheticCycleRecord(
            synthetic_cycle_id=synthetic_cycle_id,
            route="B3",
            seed=seed,
            generator_name=self.name,
            generator_version=self.version,
            checkpoint_sha256=None,
            conditions={"condition_reference_index": donor_index},
            state_path=[0],
            segments=[segment],
            boundary_treatment="none",
            sample_seconds=self.sample_seconds,
            created_utc="",
            waveform_sha256="",
        )
        return power, record
