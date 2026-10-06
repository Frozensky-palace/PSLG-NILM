"""B3-G: conditional 1D WGAN-GP for full cycles (roadmap D3).

Same bucket/pad/condition treatment as the CVAE family; the critic is a
convolutional regressor trained with the gradient-penalty Wasserstein
objective. History records the (negative) critic score as the W-distance
proxy plus the penalty magnitude, so instability is visible in the curve
instead of discovered downstream.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.generation.full_cycle_cvae import (
    BUCKET_MULTIPLE,
    LengthBucketizer,
    pad_and_mask,
)


class WGANCritic(nn.Module):
    """Convolutional critic: (wave, condition) -> realness score."""

    def __init__(self, wave_length: int, condition_dim: int, width: int = 32):
        super().__init__()
        if wave_length % BUCKET_MULTIPLE:
            raise ValueError(
                f"wave_length must be a multiple of {BUCKET_MULTIPLE}")
        reduced = wave_length // 8
        self._width2 = width * 2
        self._reduced = reduced
        self.features = nn.Sequential(
            nn.Conv1d(1, width, 5, stride=2, padding=2), nn.LeakyReLU(0.2),
            nn.Conv1d(width, width * 2, 5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(width * 2, width * 2, 5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
        )
        self.head = nn.Linear(width * 2 * reduced + condition_dim, 1)

    def forward(self, wave: torch.Tensor, cond: torch.Tensor
                ) -> torch.Tensor:
        hidden = self.features(wave)
        return self.head(torch.cat([hidden, cond], dim=1)).squeeze(1)


class WGANGenerator(nn.Module):
    """Latent + condition -> waveform; mirrors the CVAE decoder shape."""

    def __init__(self, wave_length: int, condition_dim: int,
                 latent_dim: int = 16, width: int = 32):
        super().__init__()
        reduced = wave_length // 8
        self.latent_dim = latent_dim
        self._width2 = width * 2
        self._reduced = reduced
        self.fc = nn.Linear(latent_dim + condition_dim, width * 2 * reduced)
        self.blocks = nn.Sequential(
            nn.ConvTranspose1d(width * 2, width, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(width, width, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(width, width, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(width, 1, 5, padding=2),
        )

    def forward(self, latent: torch.Tensor, cond: torch.Tensor
                ) -> torch.Tensor:
        hidden = self.fc(torch.cat([latent, cond], dim=1))
        return self.blocks(hidden.view(-1, self._width2, self._reduced))


def gradient_penalty(critic: WGANCritic, real: torch.Tensor,
                     fake: torch.Tensor, cond: torch.Tensor
                     ) -> torch.Tensor:
    alpha = torch.rand(len(real), 1, 1, device=real.device)
    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    score = critic(interpolates, cond)
    gradients = torch.autograd.grad(
        outputs=score, inputs=interpolates,
        grad_outputs=torch.ones_like(score), create_graph=True,
        retain_graph=True)[0]
    return ((gradients.norm(2, dim=(1, 2)) - 1) ** 2).mean()


def apply_input_mask(wave: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Zero a padded wave batch beyond each sample's valid length.

    ``wave`` is (batch, 1, length); ``mask`` is the (batch, length) valid
    mask from ``pad_and_mask``. This is the B4WD segment-level mask
    semantics: the critic, the gradient penalty and the w-distance monitor
    all operate on masked inputs, never on zero padding.
    """
    return wave * mask.unsqueeze(1)


def train_wgan(generator: WGANGenerator, critic: WGANCritic,
               waves: list[np.ndarray], conditions: np.ndarray,
               bucketizer: LengthBucketizer, *, epochs: int,
               batch_size: int = 32, learning_rate: float = 1e-4,
               n_critic: int = 5, gp_weight: float = 10.0, seed: int = 0,
               device: str = "cpu") -> list[dict]:
    """WGAN-GP loop; waves must already be divided by the power scale."""
    torch.manual_seed(seed)
    generator.to(device)
    critic.to(device)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate,
                                   betas=(0.5, 0.9))
    c_optimizer = torch.optim.Adam(critic.parameters(), lr=learning_rate,
                                   betas=(0.5, 0.9))
    bucket_length = bucketizer.bucket_length(
        bucketizer.encode(max(len(w) for w in waves)))
    padded, _ = pad_and_mask(waves, bucket_length)
    cond = torch.from_numpy(np.stack(conditions).astype(np.float32))
    n = len(waves)
    generator_rng = torch.Generator().manual_seed(seed)
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        order = torch.randperm(n, generator=generator_rng)
        w_totals, gp_totals, batches = 0.0, 0.0, 0
        for start in range(0, n, batch_size):
            picked = order[start:start + batch_size]
            wave = padded[picked].to(device)
            cnd = cond[picked].to(device)
            for _ in range(n_critic):
                latent = torch.randn(len(wave), generator.latent_dim,
                                     device=device)
                fake = generator(latent, cnd).detach()
                c_optimizer.zero_grad()
                w_distance = (critic(wave, cnd).mean()
                              - critic(fake, cnd).mean())
                penalty = gradient_penalty(critic, wave, fake, cnd)
                critic_loss = -w_distance + gp_weight * penalty
                critic_loss.backward()
                c_optimizer.step()
            latent = torch.randn(len(wave), generator.latent_dim,
                                 device=device)
            g_optimizer.zero_grad()
            fake = generator(latent, cnd)
            generator_loss = -critic(fake, cnd).mean()
            generator_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 5.0)
            g_optimizer.step()
            w_totals += float(w_distance.detach())
            gp_totals += float(penalty.detach())
            batches += 1
        history.append({"epoch": epoch,
                        "w_distance": w_totals / max(batches, 1),
                        "gradient_penalty": gp_totals / max(batches, 1)})
    return history


def train_primitive_wgan(generator: WGANGenerator, critic: WGANCritic,
                         waves: list[np.ndarray], conditions: np.ndarray,
                         bucketizer: LengthBucketizer,
                         labels: list[int] | None = None, *, epochs: int,
                         batch_size: int = 32, learning_rate: float = 1e-4,
                         n_critic: int = 5, gp_weight: float = 10.0,
                         seed: int = 0, device: str = "cpu") -> list[dict]:
    """Segment-level masked WGAN-GP (B4WD prereg §2, preregistered masks).

    Unlike ``train_wgan`` (full cycles: single max bucket, mask discarded),
    real AND fake waves are zeroed beyond each real segment's valid length
    before entering the critic; the gradient penalty interpolates between
    the masked waves; the generator output is produced at full length and
    then masked with the real batch's mask. No loss term ever sees padding.
    Segment lengths span hundreds to thousands of samples, so without this
    the critic could separate real/fake by pad length alone.

    ``labels`` (state label per wave) only adds a per-state w-distance
    bucket to the history — the collapse monitor preregistered in §3
    (a single mixed-state scalar can hide one-state mode collapse).
    """
    torch.manual_seed(seed)
    generator.to(device)
    critic.to(device)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate,
                                   betas=(0.5, 0.9))
    c_optimizer = torch.optim.Adam(critic.parameters(), lr=learning_rate,
                                   betas=(0.5, 0.9))
    bucket_length = bucketizer.bucket_length(
        bucketizer.encode(max(len(w) for w in waves)))
    padded, mask = pad_and_mask(waves, bucket_length)
    cond = torch.from_numpy(np.stack(conditions).astype(np.float32))
    labels_t = (torch.tensor([int(label) for label in labels])
                if labels is not None else None)
    n = len(waves)
    generator_rng = torch.Generator().manual_seed(seed)
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        order = torch.randperm(n, generator=generator_rng)
        w_totals, gp_totals, batches = 0.0, 0.0, 0
        per_state: dict[int, list[float]] = {}
        for start in range(0, n, batch_size):
            picked = order[start:start + batch_size]
            msk = mask[picked].to(device)
            real = apply_input_mask(padded[picked].to(device), msk)
            cnd = cond[picked].to(device)
            for _ in range(n_critic):
                latent = torch.randn(len(real), generator.latent_dim,
                                     device=device)
                fake = apply_input_mask(
                    generator(latent, cnd).detach(), msk)
                c_optimizer.zero_grad()
                w_distance = (critic(real, cnd).mean()
                              - critic(fake, cnd).mean())
                penalty = gradient_penalty(critic, real, fake, cnd)
                critic_loss = -w_distance + gp_weight * penalty
                critic_loss.backward()
                c_optimizer.step()
            latent = torch.randn(len(real), generator.latent_dim,
                                 device=device)
            g_optimizer.zero_grad()
            fake = apply_input_mask(generator(latent, cnd), msk)
            generator_loss = -critic(fake, cnd).mean()
            generator_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 5.0)
            g_optimizer.step()
            w_totals += float(w_distance.detach())
            gp_totals += float(penalty.detach())
            batches += 1
            if labels_t is not None:
                with torch.no_grad():
                    per_sample = critic(real, cnd) - critic(fake, cnd)
                for row, label in enumerate(labels_t[picked].tolist()):
                    per_state.setdefault(int(label), []).append(
                        float(per_sample[row]))
        entry = {"epoch": epoch,
                 "w_distance": w_totals / max(batches, 1),
                 "gradient_penalty": gp_totals / max(batches, 1)}
        if per_state:
            entry["w_distance_by_state"] = {
                str(state): sum(values) / len(values)
                for state, values in sorted(per_state.items())}
        history.append(entry)
    return history
