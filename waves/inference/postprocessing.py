"""
File: postprocessing.py

Author: Ksenia Zolina, Dmitry Ryumin, Denis Ivanko, and Alexey Karpov

Description: Audio postprocessing utilities for WAVES.

License: MIT License
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from waves.audio.preprocessing import (
    PreprocessedAudio,
    restore_waveform_scale,
)
from waves.config import (
    get_config_bool,
    get_config_float,
)


@dataclass(frozen=True, slots=True)
class PostprocessedWaveform:
    """Postprocessed enhanced waveform and associated statistics."""

    waveform: Tensor
    input_rms: float
    output_rms_before_matching: float
    output_rms_after_matching: float
    peak_before_limiting: float
    peak_after_limiting: float
    was_rms_matched: bool
    was_peak_limited: bool


def compute_rms(
    waveform: Tensor,
) -> float:
    """Compute waveform RMS."""

    if waveform.numel() == 0:
        return 0.0

    waveform_float = waveform.float()

    rms = torch.sqrt(torch.mean(waveform_float.square()) + 1e-12)

    return float(rms.detach().cpu().item())


def compute_peak(
    waveform: Tensor,
) -> float:
    """Return the absolute waveform peak."""

    if waveform.numel() == 0:
        return 0.0

    peak = waveform.abs().max()

    return float(peak.detach().cpu().item())


def validate_output_waveform(
    waveform: Tensor,
) -> None:
    """Validate an enhanced waveform."""

    if waveform.ndim != 2:
        msg = f"Enhanced waveform must have shape [channels, samples], got {tuple(waveform.shape)}."
        raise ValueError(msg)

    if waveform.shape[0] != 1:
        msg = f"Enhanced waveform must be mono, got {waveform.shape[0]} channels."
        raise ValueError(msg)

    if waveform.shape[-1] <= 0:
        msg = "Enhanced waveform is empty."
        raise ValueError(msg)

    if not bool(torch.isfinite(waveform).all().item()):
        msg = "Enhanced waveform contains non-finite values."
        raise ValueError(msg)


def match_input_rms(
    waveform: Tensor,
    input_waveform: Tensor,
    rms_eps: float,
) -> tuple[Tensor, bool]:
    """Match enhanced waveform RMS to the input waveform RMS."""

    input_rms = compute_rms(input_waveform)

    output_rms = compute_rms(waveform)

    if input_rms <= rms_eps or output_rms <= rms_eps:
        return waveform, False

    scale = input_rms / output_rms

    return (
        waveform * scale,
        True,
    )


def limit_waveform_peak(
    waveform: Tensor,
    peak_limit: float,
) -> tuple[Tensor, bool]:
    """Scale waveform if its absolute peak exceeds the configured limit."""

    if not 0.0 < peak_limit <= 1.0:
        msg = "Audio peak limit must be in the range (0, 1]."
        raise ValueError(msg)

    peak = compute_peak(waveform)

    if peak <= peak_limit:
        return waveform, False

    return (
        waveform * (peak_limit / peak),
        True,
    )


def postprocess_enhanced_waveform(
    normalized_waveform: Tensor,
    preprocessed_audio: PreprocessedAudio,
) -> PostprocessedWaveform:
    """Restore and postprocess an enhanced normalized waveform."""

    waveform = (
        normalized_waveform.detach()
        .to(
            device="cpu",
            dtype=torch.float32,
        )
        .contiguous()
    )

    validate_output_waveform(waveform)

    if waveform.shape[-1] != preprocessed_audio.num_samples:
        msg = (
            "Enhanced waveform length does not match the input: "
            f"{waveform.shape[-1]} != {preprocessed_audio.num_samples}."
        )
        raise ValueError(msg)

    waveform = restore_waveform_scale(
        waveform=waveform,
        normalization_factor=preprocessed_audio.normalization_factor,
    )

    input_waveform = restore_waveform_scale(
        waveform=preprocessed_audio.waveform.detach().to(
            device="cpu",
            dtype=torch.float32,
        ),
        normalization_factor=preprocessed_audio.normalization_factor,
    )

    input_rms = compute_rms(input_waveform)

    output_rms_before_matching = compute_rms(waveform)

    match_rms = get_config_bool(
        "AudioPostprocessing_MATCH_INPUT_RMS",
        True,
    )

    rms_eps = get_config_float(
        "AudioPostprocessing_RMS_EPS",
        1e-6,
    )

    was_rms_matched = False

    if match_rms:
        waveform, was_rms_matched = match_input_rms(
            waveform=waveform,
            input_waveform=input_waveform,
            rms_eps=rms_eps,
        )

    output_rms_after_matching = compute_rms(waveform)

    peak_before_limiting = compute_peak(waveform)

    enable_peak_limiting = get_config_bool(
        "AudioPostprocessing_ENABLE_PEAK_LIMITING",
        True,
    )

    peak_limit = get_config_float(
        "AudioPostprocessing_PEAK_LIMIT",
        0.99,
    )

    was_peak_limited = False

    if enable_peak_limiting:
        (
            waveform,
            was_peak_limited,
        ) = limit_waveform_peak(
            waveform=waveform,
            peak_limit=peak_limit,
        )

    waveform = waveform.contiguous()

    validate_output_waveform(waveform)

    peak_after_limiting = compute_peak(waveform)

    return PostprocessedWaveform(
        waveform=waveform,
        input_rms=input_rms,
        output_rms_before_matching=output_rms_before_matching,
        output_rms_after_matching=output_rms_after_matching,
        peak_before_limiting=peak_before_limiting,
        peak_after_limiting=peak_after_limiting,
        was_rms_matched=was_rms_matched,
        was_peak_limited=was_peak_limited,
    )
