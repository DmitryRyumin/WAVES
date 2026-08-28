"""
File: noise_conditioning.py

Author: Ksenia Zolina, Dmitry Ryumin, Denis Ivanko, and Alexey Karpov

Description: Noise conditioning modules for WAVES sparse Mixture-of-Experts routing.

License: MIT License
"""

from typing import cast

import torch
from torch import Tensor, nn


class MeanSpectrumSource(nn.Module):
    """Project the mean magnitude spectrum to a noise embedding."""

    def __init__(
        self,
        n_freq: int,
        noise_ctx_dim: int,
    ) -> None:
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(
                n_freq,
                noise_ctx_dim,
            ),
            nn.GELU(),
        )

    def forward(
        self,
        noisy_amp: Tensor,
        encoder_out: Tensor | None = None,
    ) -> Tensor:
        """Return the noise embedding."""

        del encoder_out

        return cast(
            Tensor,
            self.proj(noisy_amp.mean(dim=2)),
        )


class MeanStdSpectrumSource(nn.Module):
    """Project spectral mean and standard deviation to a noise embedding."""

    def __init__(
        self,
        n_freq: int,
        noise_ctx_dim: int,
    ) -> None:
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(
                2 * n_freq,
                noise_ctx_dim,
            ),
            nn.GELU(),
        )

    def forward(
        self,
        noisy_amp: Tensor,
        encoder_out: Tensor | None = None,
    ) -> Tensor:
        """Return the noise embedding."""

        del encoder_out

        mean = noisy_amp.mean(dim=2)

        std = noisy_amp.std(
            dim=2,
            unbiased=False,
        )

        features = torch.cat(
            [
                mean,
                std,
            ],
            dim=-1,
        )

        return cast(
            Tensor,
            self.proj(features),
        )


class SNREstimateSource(nn.Module):
    """Project spectral SNR descriptors to a noise embedding."""

    def __init__(
        self,
        noise_ctx_dim: int,
    ) -> None:
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(
                3,
                noise_ctx_dim,
            ),
            nn.GELU(),
        )

    def forward(
        self,
        noisy_amp: Tensor,
        encoder_out: Tensor | None = None,
    ) -> Tensor:
        """Return the noise embedding."""

        del encoder_out

        mean_spec = noisy_amp.mean(dim=2)

        peak_energy = mean_spec.max(
            dim=-1,
            keepdim=True,
        ).values.clamp(
            min=1e-8,
        )

        mean_energy = mean_spec.mean(
            dim=-1,
            keepdim=True,
        ).clamp(
            min=1e-8,
        )

        log_ratio = torch.log(peak_energy / mean_energy)

        log_spec = torch.log(
            mean_spec.clamp(
                min=1e-8,
            )
        )

        geo_mean = log_spec.mean(
            dim=-1,
            keepdim=True,
        ).exp()

        flatness = (geo_mean / mean_energy).clamp(
            min=1e-8,
            max=1.0,
        )

        num_frequency_bins = mean_spec.size(-1)

        frequency_bins = torch.arange(
            num_frequency_bins,
            device=mean_spec.device,
            dtype=mean_spec.dtype,
        )

        centroid = (mean_spec * frequency_bins).sum(
            dim=-1,
            keepdim=True,
        ) / (
            mean_spec.sum(
                dim=-1,
                keepdim=True,
            )
            + 1e-8
        )

        centroid = centroid / float(num_frequency_bins)

        features = torch.cat(
            [
                log_ratio,
                flatness,
                centroid,
            ],
            dim=-1,
        )

        return cast(
            Tensor,
            self.proj(features),
        )


class EncoderFeaturesSource(nn.Module):
    """Use global average pooled encoder features as noise context."""

    def __init__(
        self,
        encoder_channels: int,
        noise_ctx_dim: int,
    ) -> None:
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(
                encoder_channels,
                noise_ctx_dim,
            ),
            nn.GELU(),
        )

    def forward(
        self,
        noisy_amp: Tensor,
        encoder_out: Tensor | None = None,
    ) -> Tensor:
        """Return the noise embedding."""

        del noisy_amp

        if encoder_out is None:
            msg = "EncoderFeaturesSource requires encoder_out."
            raise ValueError(msg)

        pooled = encoder_out.mean(dim=(2, 3))

        return cast(
            Tensor,
            self.proj(pooled),
        )


class ConvSpectrumSource(nn.Module):
    """Encode the mean spectrum with a one-dimensional CNN."""

    def __init__(
        self,
        n_freq: int,
        noise_ctx_dim: int,
    ) -> None:
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv1d(
                1,
                16,
                kernel_size=9,
                padding=4,
            ),
            nn.GELU(),
            nn.Conv1d(
                16,
                16,
                kernel_size=5,
                padding=2,
            ),
            nn.GELU(),
        )

        self.proj = nn.Linear(
            16 * n_freq,
            noise_ctx_dim,
        )

    def forward(
        self,
        noisy_amp: Tensor,
        encoder_out: Tensor | None = None,
    ) -> Tensor:
        """Return the noise embedding."""

        del encoder_out

        mean = noisy_amp.mean(dim=2).unsqueeze(1)

        features = cast(
            Tensor,
            self.convs(mean),
        )

        flattened = features.reshape(
            features.size(0),
            -1,
        )

        return cast(
            Tensor,
            self.proj(flattened),
        )


class VQNoiseSource(nn.Module):
    """Quantize a continuous noise embedding using a learned codebook."""

    def __init__(
        self,
        n_freq: int,
        noise_ctx_dim: int,
        num_codes: int = 16,
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(
                n_freq,
                noise_ctx_dim,
            ),
            nn.GELU(),
            nn.Linear(
                noise_ctx_dim,
                noise_ctx_dim,
            ),
        )

        self.codebook = nn.Embedding(
            num_codes,
            noise_ctx_dim,
        )

        nn.init.uniform_(
            self.codebook.weight,
            -0.1,
            0.1,
        )

        self._last_commitment_loss: Tensor | float = 0.0

    def forward(
        self,
        noisy_amp: Tensor,
        encoder_out: Tensor | None = None,
    ) -> Tensor:
        """Return the straight-through quantized noise embedding."""

        del encoder_out

        mean = noisy_amp.mean(dim=2)

        encoded = cast(
            Tensor,
            self.encoder(mean),
        )

        distances = torch.cdist(
            encoded,
            self.codebook.weight,
        )

        indices = distances.argmin(dim=-1)

        quantized = cast(
            Tensor,
            self.codebook(indices),
        )

        self._last_commitment_loss = 0.25 * (encoded - quantized.detach()).pow(2).mean()

        return encoded + (quantized - encoded).detach()


def build_noise_source(
    noise_source: str,
    n_freq: int,
    noise_ctx_dim: int,
    encoder_channels: int = 64,
) -> nn.Module:
    """Create a noise conditioning module."""

    if noise_source == "mean_spectrum":
        return MeanSpectrumSource(
            n_freq,
            noise_ctx_dim,
        )

    if noise_source == "mean_std_spectrum":
        return MeanStdSpectrumSource(
            n_freq,
            noise_ctx_dim,
        )

    if noise_source == "snr_estimate":
        return SNREstimateSource(noise_ctx_dim)

    if noise_source == "encoder":
        return EncoderFeaturesSource(
            encoder_channels,
            noise_ctx_dim,
        )

    if noise_source == "conv_spectrum":
        return ConvSpectrumSource(
            n_freq,
            noise_ctx_dim,
        )

    if noise_source == "vq_spectrum":
        return VQNoiseSource(
            n_freq,
            noise_ctx_dim,
        )

    msg = f"Unknown noise source: {noise_source}"
    raise ValueError(msg)
