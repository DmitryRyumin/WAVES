"""
File: spectral.py

Author: Ksenia Zolina, Dmitry Ryumin, Denis Ivanko, and Alexey Karpov

Description: STFT and inverse STFT utilities for WAVES spectral processing.

License: MIT License
"""

import torch
from torch import Tensor

_HANN_WINDOW_CACHE: dict[
    tuple[
        int,
        str,
        torch.dtype,
    ],
    Tensor,
] = {}


def get_hann_window(
    win_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Return a cached Hann window."""

    cache_key = (
        win_size,
        str(device),
        dtype,
    )

    if cache_key not in _HANN_WINDOW_CACHE:
        _HANN_WINDOW_CACHE[cache_key] = torch.hann_window(
            win_size,
            device=device,
            dtype=dtype,
        )

    return _HANN_WINDOW_CACHE[cache_key]


def magnitude_phase_stft(
    waveform: Tensor,
    n_fft: int,
    hop_size: int,
    win_size: int,
    compress_factor: float = 1.0,
    *,
    center: bool = True,
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
]:
    """Compute compressed magnitude, phase, and complex STFT features."""

    hann_window = get_hann_window(
        win_size=win_size,
        device=waveform.device,
        dtype=waveform.dtype,
    )

    stft = torch.stft(
        waveform,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        return_complex=True,
    )

    stft_real = torch.view_as_real(stft)

    magnitude = torch.sqrt(
        torch.clamp(
            stft_real.pow(2).sum(-1),
            min=1e-9,
        )
    )

    phase = torch.atan2(
        stft_real[..., 1],
        stft_real[..., 0],
    )

    magnitude = torch.pow(
        magnitude,
        compress_factor,
    )

    complex_features = torch.stack(
        (
            magnitude * torch.cos(phase),
            magnitude * torch.sin(phase),
        ),
        dim=-1,
    )

    return (
        magnitude,
        phase,
        complex_features,
    )


def magnitude_phase_istft(
    magnitude: Tensor,
    phase: Tensor,
    n_fft: int,
    hop_size: int,
    win_size: int,
    compress_factor: float = 1.0,
    *,
    center: bool = True,
    length: int | None = None,
) -> Tensor:
    """Reconstruct a waveform from compressed magnitude and phase."""

    magnitude = torch.pow(
        magnitude,
        1.0 / compress_factor,
    )

    complex_spectrum = torch.complex(
        magnitude * torch.cos(phase),
        magnitude * torch.sin(phase),
    )

    hann_window = get_hann_window(
        win_size=win_size,
        device=complex_spectrum.device,
        dtype=magnitude.dtype,
    )

    return torch.istft(
        complex_spectrum,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        length=length,
    )
