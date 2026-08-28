"""
File: preprocessing.py

Author: Ksenia Zolina, Dmitry Ryumin, Denis Ivanko, and Alexey Karpov

Description: Audio preprocessing utilities for the WAVES Gradio application.

License: MIT License
"""

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from waves.audio.decoder import DecodedAudio
from waves.config import (
    get_config_bool,
    get_config_float,
)


@dataclass(frozen=True, slots=True)
class PreprocessedAudio:
    """Audio data preprocessed for speech enhancement."""

    path: str
    sample_rate: int
    num_channels: int
    duration_seconds: float
    num_samples: int
    waveform: Tensor
    normalization_factor: float


def ensure_channel_first(
    waveform: Tensor,
) -> Tensor:
    """Ensure waveform shape is channel-first."""

    if waveform.ndim == 1:
        return waveform.unsqueeze(0)

    if waveform.ndim == 2:
        return waveform

    msg = f"Expected waveform with one or two dimensions, got shape={tuple(waveform.shape)}."
    raise ValueError(msg)


def ensure_float32(
    waveform: Tensor,
) -> Tensor:
    """Convert waveform to contiguous float32."""

    return waveform.to(
        dtype=torch.float32,
    ).contiguous()


def validate_waveform(
    waveform: Tensor,
) -> None:
    """Validate waveform before model preprocessing."""

    if waveform.numel() == 0:
        msg = "Audio waveform is empty."
        raise ValueError(msg)

    if waveform.shape[-1] <= 0:
        msg = "Audio waveform does not contain any samples."
        raise ValueError(msg)

    if not bool(torch.isfinite(waveform).all().item()):
        msg = "Audio waveform contains non-finite values."
        raise ValueError(msg)


def normalize_waveform(
    waveform: Tensor,
    eps: float,
) -> tuple[
    Tensor,
    float,
]:
    """Normalize the waveform using the MP-SENet-style RMS normalization adopted in WAVES."""

    num_samples = int(waveform.numel())

    if num_samples <= 0:
        return (
            waveform,
            1.0,
        )

    signal_energy = torch.sum(
        waveform.square(),
    )

    energy = float(signal_energy.detach().cpu().item())

    if energy <= eps:
        return (
            waveform,
            1.0,
        )

    normalization_factor = math.sqrt(num_samples / energy)

    normalized_waveform = waveform * normalization_factor

    return (
        normalized_waveform,
        normalization_factor,
    )


def restore_waveform_scale(
    waveform: Tensor,
    normalization_factor: float,
) -> Tensor:
    """Restore waveform amplitude after model inference."""

    if normalization_factor <= 0.0:
        return waveform

    return waveform / normalization_factor


def preprocess_audio_for_enhancement(
    decoded_audio: DecodedAudio,
) -> PreprocessedAudio:
    """Preprocess decoded audio for speech enhancement."""

    enable_normalization = get_config_bool(
        "AudioPreprocessing_ENABLE_NORMALIZATION",
        True,
    )

    normalization_eps = get_config_float(
        "AudioPreprocessing_NORMALIZATION_EPS",
        1e-8,
    )

    waveform = ensure_channel_first(
        decoded_audio.waveform,
    )

    waveform = ensure_float32(
        waveform,
    )

    validate_waveform(waveform)

    if waveform.shape[0] != 1:
        msg = f"WAVES expects mono audio, got {waveform.shape[0]} channels."
        raise ValueError(msg)

    normalization_factor = 1.0

    if enable_normalization:
        (
            waveform,
            normalization_factor,
        ) = normalize_waveform(
            waveform=waveform,
            eps=normalization_eps,
        )

    return PreprocessedAudio(
        path=decoded_audio.path,
        sample_rate=decoded_audio.sample_rate,
        num_channels=int(waveform.shape[0]),
        duration_seconds=decoded_audio.duration_seconds,
        num_samples=int(waveform.shape[-1]),
        waveform=waveform,
        normalization_factor=normalization_factor,
    )
