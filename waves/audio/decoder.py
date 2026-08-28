"""
File: decoder.py

Author: Ksenia Zolina, Dmitry Ryumin, Denis Ivanko, and Alexey Karpov

Description: Audio decoding utilities for the WAVES Gradio application.

License: MIT License
"""

from dataclasses import dataclass
from pathlib import Path

from torch import Tensor

from waves.audio.torchcodec_compat import AudioDecoder
from waves.config import get_config_int


@dataclass(frozen=True, slots=True)
class DecodedAudio:
    """Decoded audio data prepared for speech enhancement."""

    path: str
    sample_rate: int
    duration_seconds: float
    waveform: Tensor


def decode_audio_for_enhancement(
    audio_path: str,
) -> DecodedAudio:
    """Decode an audio file for speech enhancement."""

    path = Path(audio_path)

    if not path.is_file():
        msg = f"Audio file was not found: {path}"
        raise FileNotFoundError(msg)

    target_sample_rate = get_config_int(
        "AudioDecoding_TARGET_SAMPLE_RATE",
        16000,
    )

    target_num_channels = get_config_int(
        "AudioDecoding_TARGET_NUM_CHANNELS",
        1,
    )

    if target_sample_rate <= 0:
        msg = "AudioDecoding_TARGET_SAMPLE_RATE must be greater than zero."
        raise ValueError(msg)

    if target_num_channels <= 0:
        msg = "AudioDecoding_TARGET_NUM_CHANNELS must be greater than zero."
        raise ValueError(msg)

    decoder = AudioDecoder(
        str(path),
        sample_rate=target_sample_rate,
        num_channels=target_num_channels,
    )

    samples = decoder.get_all_samples()

    waveform = samples.data

    if waveform.ndim != 2:
        msg = f"TorchCodec returned an unexpected waveform shape: {tuple(waveform.shape)}."
        raise ValueError(msg)

    if waveform.shape[-1] <= 0:
        msg = "TorchCodec returned an empty audio waveform."
        raise ValueError(msg)

    sample_rate = int(samples.sample_rate)

    duration_seconds = float(samples.duration_seconds)

    return DecodedAudio(
        path=str(path),
        sample_rate=sample_rate,
        duration_seconds=duration_seconds,
        waveform=waveform,
    )
