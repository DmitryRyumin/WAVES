"""
File: metadata.py

Author: Ksenia Zolina, Dmitry Ryumin, Denis Ivanko, and Alexey Karpov

Description: Audio metadata utilities for the WAVES Gradio application.

License: MIT License
"""

from dataclasses import dataclass
from pathlib import Path

from waves.audio.torchcodec_compat import AudioDecoder


@dataclass(frozen=True, slots=True)
class AudioMetadata:
    """Metadata extracted from an audio file."""

    filename: str
    size_bytes: int
    duration_seconds: float | None
    bit_rate: float | None
    codec: str | None
    sample_rate: int | None
    num_channels: int | None
    sample_format: str | None


def read_audio_metadata(
    audio_path: str,
) -> AudioMetadata:
    """Read audio metadata using TorchCodec."""

    path = Path(audio_path)

    decoder = AudioDecoder(str(path))

    metadata = decoder.metadata

    return AudioMetadata(
        filename=path.name,
        size_bytes=path.stat().st_size,
        duration_seconds=metadata.duration_seconds,
        bit_rate=metadata.bit_rate,
        codec=metadata.codec,
        sample_rate=metadata.sample_rate,
        num_channels=metadata.num_channels,
        sample_format=metadata.sample_format,
    )
