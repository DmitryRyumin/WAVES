"""
File: encoder.py

Author: Ksenia Zolina, Dmitry Ryumin, Denis Ivanko, and Alexey Karpov

Description: Audio encoding utilities for the WAVES Gradio application.

License: MIT License
"""

from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Final

import torch
from torch import Tensor

from waves.audio.torchcodec_compat import AudioEncoder

TEMPORARY_AUDIO_PREFIX: Final = "waves_"
TEMPORARY_AUDIO_SUFFIX: Final = ".wav"


@dataclass(frozen=True, slots=True)
class EncodedAudio:
    """Encoded enhanced audio file."""

    path: str


def validate_audio_for_encoding(
    waveform: Tensor,
    sample_rate: int,
) -> None:
    """Validate a waveform before TorchCodec encoding."""

    if sample_rate <= 0:
        msg = "Audio sample rate must be greater than zero."
        raise ValueError(msg)

    if waveform.ndim not in {1, 2}:
        msg = f"Audio waveform must have one or two dimensions, got shape={tuple(waveform.shape)}."
        raise ValueError(msg)

    if waveform.numel() == 0:
        msg = "Audio waveform is empty."
        raise ValueError(msg)

    if waveform.shape[-1] <= 0:
        msg = "Audio waveform does not contain any samples."
        raise ValueError(msg)

    if not bool(torch.isfinite(waveform).all().item()):
        msg = "Audio waveform contains non-finite values."
        raise ValueError(msg)

    peak = float(waveform.abs().max().detach().cpu().item())

    if peak > 1.0:
        msg = f"TorchCodec requires floating-point audio samples in the range [-1, 1], got absolute peak={peak:.6f}."
        raise ValueError(msg)


def encode_audio_to_file(
    waveform: Tensor,
    sample_rate: int,
    output_path: str | Path,
) -> EncodedAudio:
    """Encode an audio waveform to a file using TorchCodec."""

    waveform = (
        waveform.detach()
        .to(
            device="cpu",
            dtype=torch.float32,
        )
        .contiguous()
    )

    validate_audio_for_encoding(
        waveform=waveform,
        sample_rate=sample_rate,
    )

    path = Path(output_path).expanduser()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    encoder = AudioEncoder(
        samples=waveform,
        sample_rate=sample_rate,
    )

    encoder.to_file(path)

    if not path.is_file():
        msg = f"TorchCodec did not create the expected output file: {path}"
        raise RuntimeError(msg)

    if path.stat().st_size <= 0:
        msg = f"TorchCodec created an empty output file: {path}"
        raise RuntimeError(msg)

    return EncodedAudio(path=str(path))


def encode_audio_to_temporary_wav(
    waveform: Tensor,
    sample_rate: int,
) -> EncodedAudio:
    """Encode an audio waveform to a temporary WAV file."""

    with tempfile.NamedTemporaryFile(
        prefix=TEMPORARY_AUDIO_PREFIX,
        suffix=TEMPORARY_AUDIO_SUFFIX,
        delete=False,
    ) as temporary_file:
        output_path = Path(temporary_file.name)

    try:
        return encode_audio_to_file(
            waveform=waveform,
            sample_rate=sample_rate,
            output_path=output_path,
        )
    except Exception:
        output_path.unlink(missing_ok=True)
        raise


def remove_temporary_encoded_audio_file(
    audio_path: str | Path | None,
) -> bool:
    """Remove a temporary WAV created by the WAVES encoder."""

    if audio_path is None:
        return False

    path = Path(audio_path).expanduser()

    resolved_path = path.resolve()

    temporary_directory = Path(tempfile.gettempdir()).resolve()

    if resolved_path.parent != temporary_directory:
        return False

    if not resolved_path.name.startswith(TEMPORARY_AUDIO_PREFIX):
        return False

    if resolved_path.suffix.lower() != TEMPORARY_AUDIO_SUFFIX:
        return False

    existed = resolved_path.is_file()

    resolved_path.unlink(missing_ok=True)

    return existed
