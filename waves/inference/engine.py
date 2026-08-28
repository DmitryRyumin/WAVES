"""
File: engine.py

Author: Ksenia Zolina, Dmitry Ryumin, Denis Ivanko, and Alexey Karpov

Description: WAVES sliding-window inference engine for speech enhancement.

License: MIT License
"""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from torch import Tensor
import torch.nn.functional as F

from waves.audio.decoder import (
    decode_audio_for_enhancement,
)
from waves.audio.encoder import (
    EncodedAudio,
    encode_audio_to_temporary_wav,
)
from waves.audio.preprocessing import (
    PreprocessedAudio,
    preprocess_audio_for_enhancement,
)
from waves.audio.spectral import (
    magnitude_phase_istft,
    magnitude_phase_stft,
)
from waves.config import (
    get_config_bool,
    get_config_float,
    get_config_str,
)
from waves.inference.postprocessing import (
    PostprocessedWaveform,
    postprocess_enhanced_waveform,
)
from waves.models.loader import (
    LoadedModel,
    load_model,
)
from waves.models.transformer import TransformerBlock
from waves.routing import (
    RoutingTelemetry,
    RoutingTelemetryCollector,
)


@dataclass(frozen=True, slots=True)
class EnhancedAudio:
    """Enhanced audio returned by the inference engine."""

    input_path: str
    model_name: str
    model_weights_path: str
    sample_rate: int
    duration_seconds: float
    num_samples: int
    waveform: Tensor
    window_samples: int
    hop_samples: int
    device: str
    postprocessing: PostprocessedWaveform
    routing: RoutingTelemetry | None


@dataclass(frozen=True, slots=True)
class EnhancedAudioFile:
    """Enhanced audio together with its encoded WAV file."""

    audio: EnhancedAudio
    encoded: EncodedAudio


def get_window_parameters(
    sample_rate: int,
) -> tuple[int, int]:
    """Return sliding-window and hop sizes in samples."""

    window_seconds = get_config_float(
        "Inference_WINDOW_SECONDS",
        2.0,
    )

    hop_seconds = get_config_float(
        "Inference_HOP_SECONDS",
        1.0,
    )

    if window_seconds <= 0.0:
        msg = "Inference window duration must be greater than zero."
        raise ValueError(msg)

    if hop_seconds <= 0.0:
        msg = "Inference hop duration must be greater than zero."
        raise ValueError(msg)

    window_samples = round(window_seconds * sample_rate)
    hop_samples = round(hop_seconds * sample_rate)

    if window_samples <= 0:
        msg = "Inference window size resolved to zero samples."
        raise ValueError(msg)

    if hop_samples <= 0:
        msg = "Inference hop size resolved to zero samples."
        raise ValueError(msg)

    if hop_samples > window_samples:
        msg = "Inference hop size cannot exceed the window size."
        raise ValueError(msg)

    return (
        window_samples,
        hop_samples,
    )


def resolve_amp_dtype() -> torch.dtype:
    """Resolve the configured CUDA AMP dtype."""

    dtype_name = (
        get_config_str(
            "Inference_AMP_DTYPE",
            "bfloat16",
        )
        .strip()
        .lower()
    )

    if dtype_name == "bfloat16":
        return torch.bfloat16

    if dtype_name == "float16":
        return torch.float16

    if dtype_name == "float32":
        return torch.float32

    msg = f"Unsupported AMP dtype: {dtype_name}"
    raise ValueError(msg)


def should_use_cuda_amp(
    device: torch.device,
) -> bool:
    """Return whether CUDA AMP should be used."""

    enable_amp = get_config_bool(
        "Inference_ENABLE_AMP",
        True,
    )

    if not enable_amp:
        return False

    if device.type != "cuda":
        return False

    amp_dtype = resolve_amp_dtype()

    return amp_dtype in {
        torch.float16,
        torch.bfloat16,
    }


def get_spectral_device(
    inference_device: torch.device,
) -> torch.device:
    """Return the device used for STFT and inverse STFT."""

    if inference_device.type == "mps":
        return torch.device("cpu")

    return inference_device


def build_window_positions(
    num_samples: int,
    window_samples: int,
    hop_samples: int,
) -> list[int]:
    """Build deterministic sliding-window start positions."""

    if num_samples <= window_samples:
        return [0]

    last_start = num_samples - window_samples

    positions = list(
        range(
            0,
            last_start + 1,
            hop_samples,
        )
    )

    if not positions:
        positions = [0]

    if positions[-1] != last_start:
        positions.append(last_start)

    return positions


def validate_preprocessed_audio(
    audio: PreprocessedAudio,
    loaded_model: LoadedModel,
) -> None:
    """Validate compatibility between audio and the loaded model."""

    if audio.sample_rate != loaded_model.config.sampling_rate:
        msg = (
            "Audio and model sample rates do not match: "
            f"{audio.sample_rate} Hz != "
            f"{loaded_model.config.sampling_rate} Hz."
        )
        raise ValueError(msg)

    if audio.num_channels != 1:
        msg = f"WAVES inference requires mono audio, got {audio.num_channels} channels."
        raise ValueError(msg)

    if audio.waveform.ndim != 2:
        msg = f"Preprocessed waveform must have shape [channels, samples], got {tuple(audio.waveform.shape)}."
        raise ValueError(msg)

    if audio.waveform.shape[0] != 1:
        msg = f"Preprocessed waveform must contain one channel, got {audio.waveform.shape[0]}."
        raise ValueError(msg)

    if audio.num_samples <= 0:
        msg = "Preprocessed waveform must contain at least one sample."
        raise ValueError(msg)

    if audio.waveform.shape[-1] != audio.num_samples:
        msg = f"Preprocessed waveform length is inconsistent: {audio.waveform.shape[-1]} != {audio.num_samples}."
        raise ValueError(msg)


def model_has_moe_routing(
    loaded_model: LoadedModel,
) -> bool:
    """Return whether the model contains routed Transformer blocks."""

    return any(
        isinstance(
            module,
            TransformerBlock,
        )
        and module.use_moe
        for module in loaded_model.model.modules()
    )


def create_routing_collector(
    loaded_model: LoadedModel,
) -> RoutingTelemetryCollector | None:
    """Create and attach a routing collector when routing is enabled."""

    routing_enabled = get_config_bool(
        "MoERouting_ENABLE",
        True,
    )

    if not routing_enabled:
        return None

    if not model_has_moe_routing(loaded_model):
        return None

    collector = RoutingTelemetryCollector(
        loaded_model.model,
    )

    collector.attach()

    return collector


def create_routing_snapshot(
    collector: RoutingTelemetryCollector | None,
) -> RoutingTelemetry | None:
    """Return collected routing telemetry when available."""

    if collector is None:
        return None

    telemetry = collector.snapshot()

    if telemetry.is_empty:
        return None

    return telemetry


def prepare_routing_window(
    collector: RoutingTelemetryCollector | None,
    *,
    window_index: int,
    start_sample: int,
    end_sample: int,
    process_full_audio: bool,
) -> None:
    """Prepare routing capture for one inference window."""

    if collector is None:
        return

    if not process_full_audio and window_index > 0:
        if collector.is_attached:
            collector.close()

        return

    collector.begin_window(
        index=window_index,
        start_sample=start_sample,
        end_sample=end_sample,
    )


def run_model_forward(
    loaded_model: LoadedModel,
    magnitude: Tensor,
    phase: Tensor,
) -> tuple[Tensor, Tensor]:
    """Run WAVES on magnitude and phase spectra."""

    model = loaded_model.model
    device = loaded_model.device

    magnitude = magnitude.to(
        device=device,
        dtype=torch.float32,
    )

    phase = phase.to(
        device=device,
        dtype=torch.float32,
    )

    use_amp = should_use_cuda_amp(device)

    with torch.inference_mode():
        if use_amp:
            amp_dtype = resolve_amp_dtype()

            with torch.amp.autocast(
                "cuda",
                dtype=amp_dtype,
            ):
                model_result = cast(
                    tuple[
                        Tensor,
                        Tensor,
                        Tensor,
                        Tensor,
                    ],
                    model(
                        magnitude,
                        phase,
                    ),
                )

        else:
            model_result = cast(
                tuple[
                    Tensor,
                    Tensor,
                    Tensor,
                    Tensor,
                ],
                model(
                    magnitude,
                    phase,
                ),
            )

    enhanced_magnitude = model_result[0]
    enhanced_phase = model_result[1]

    return (
        enhanced_magnitude,
        enhanced_phase,
    )


def infer_segment(
    loaded_model: LoadedModel,
    segment: Tensor,
) -> Tensor:
    """Enhance one normalized mono waveform segment."""

    if segment.ndim != 2:
        msg = f"Inference segment must have shape [channels, samples], got {tuple(segment.shape)}."
        raise ValueError(msg)

    if segment.shape[0] != 1:
        msg = "Inference segment must be mono."
        raise ValueError(msg)

    segment_length = int(segment.shape[-1])

    spectral_device = get_spectral_device(
        loaded_model.device,
    )

    spectral_segment = segment.to(
        device=spectral_device,
        dtype=torch.float32,
    )

    (
        magnitude,
        phase,
        _,
    ) = magnitude_phase_stft(
        waveform=spectral_segment,
        n_fft=loaded_model.config.n_fft,
        hop_size=loaded_model.config.hop_size,
        win_size=loaded_model.config.win_size,
        compress_factor=loaded_model.config.compress_factor,
    )

    (
        enhanced_magnitude,
        enhanced_phase,
    ) = run_model_forward(
        loaded_model=loaded_model,
        magnitude=magnitude,
        phase=phase,
    )

    enhanced_magnitude = enhanced_magnitude.to(
        device=spectral_device,
        dtype=torch.float32,
    )

    enhanced_phase = enhanced_phase.to(
        device=spectral_device,
        dtype=torch.float32,
    )

    waveform = magnitude_phase_istft(
        magnitude=enhanced_magnitude,
        phase=enhanced_phase,
        n_fft=loaded_model.config.n_fft,
        hop_size=loaded_model.config.hop_size,
        win_size=loaded_model.config.win_size,
        compress_factor=loaded_model.config.compress_factor,
        length=segment_length,
    )

    waveform = (
        waveform.detach()
        .to(
            device="cpu",
            dtype=torch.float32,
        )
        .contiguous()
    )

    if waveform.ndim != 2:
        msg = f"Inverse STFT returned an unexpected shape: {tuple(waveform.shape)}."
        raise ValueError(msg)

    if waveform.shape[-1] != segment_length:
        msg = f"Inverse STFT returned an unexpected length: {waveform.shape[-1]} != {segment_length}."
        raise ValueError(msg)

    return waveform


def sliding_window_inference(
    preprocessed_audio: PreprocessedAudio,
    loaded_model: LoadedModel,
) -> tuple[
    Tensor,
    int,
    int,
    RoutingTelemetry | None,
]:
    """Enhance arbitrary-length audio using overlapping waveform windows."""

    validate_preprocessed_audio(
        audio=preprocessed_audio,
        loaded_model=loaded_model,
    )

    waveform = (
        preprocessed_audio.waveform.detach()
        .to(
            device="cpu",
            dtype=torch.float32,
        )
        .contiguous()
    )

    num_samples = preprocessed_audio.num_samples

    (
        window_samples,
        hop_samples,
    ) = get_window_parameters(
        preprocessed_audio.sample_rate,
    )

    routing_collector = create_routing_collector(
        loaded_model,
    )

    process_full_audio = get_config_bool(
        "MoERouting_PROCESS_FULL_AUDIO",
        True,
    )

    try:
        if num_samples <= window_samples:
            padding = window_samples - num_samples

            padded = F.pad(
                waveform,
                (
                    0,
                    padding,
                ),
            )

            prepare_routing_window(
                routing_collector,
                window_index=0,
                start_sample=0,
                end_sample=num_samples,
                process_full_audio=process_full_audio,
            )

            enhanced_waveform = infer_segment(
                loaded_model=loaded_model,
                segment=padded,
            )

            enhanced_waveform = enhanced_waveform[
                :,
                :num_samples,
            ].contiguous()

        else:
            positions = build_window_positions(
                num_samples=num_samples,
                window_samples=window_samples,
                hop_samples=hop_samples,
            )

            output = torch.zeros(
                num_samples,
                dtype=torch.float32,
            )

            weight = torch.zeros(
                num_samples,
                dtype=torch.float32,
            )

            fade = torch.hamming_window(
                window_samples,
                periodic=False,
                dtype=torch.float32,
            )

            source = waveform.squeeze(0)

            for (
                window_index,
                position,
            ) in enumerate(positions):
                end = position + window_samples

                segment = source[position:end].unsqueeze(0)

                if segment.shape[-1] != window_samples:
                    msg = f"Internal sliding-window segment has unexpected length {segment.shape[-1]}."
                    raise RuntimeError(msg)

                prepare_routing_window(
                    routing_collector,
                    window_index=window_index,
                    start_sample=position,
                    end_sample=min(
                        end,
                        num_samples,
                    ),
                    process_full_audio=process_full_audio,
                )

                enhanced_segment = infer_segment(
                    loaded_model=loaded_model,
                    segment=segment,
                ).squeeze(0)

                output[position:end] += enhanced_segment * fade

                weight[position:end] += fade

            enhanced_waveform = (output / weight.clamp(min=1e-8)).unsqueeze(0)

            enhanced_waveform = enhanced_waveform.contiguous()

    finally:
        if routing_collector is not None and routing_collector.is_attached:
            routing_collector.close()

    routing = create_routing_snapshot(
        routing_collector,
    )

    return (
        enhanced_waveform,
        window_samples,
        hop_samples,
        routing,
    )


def enhance_preprocessed_audio(
    preprocessed_audio: PreprocessedAudio,
    loaded_model: LoadedModel,
) -> EnhancedAudio:
    """Enhance already decoded and preprocessed audio."""

    (
        normalized_enhanced,
        window_samples,
        hop_samples,
        routing,
    ) = sliding_window_inference(
        preprocessed_audio=preprocessed_audio,
        loaded_model=loaded_model,
    )

    postprocessed = postprocess_enhanced_waveform(
        normalized_waveform=normalized_enhanced,
        preprocessed_audio=preprocessed_audio,
    )

    waveform = (
        postprocessed.waveform.detach()
        .to(
            device="cpu",
            dtype=torch.float32,
        )
        .contiguous()
    )

    return EnhancedAudio(
        input_path=preprocessed_audio.path,
        model_name=loaded_model.info.name,
        model_weights_path=str(loaded_model.info.weights_path),
        sample_rate=preprocessed_audio.sample_rate,
        duration_seconds=(preprocessed_audio.duration_seconds),
        num_samples=preprocessed_audio.num_samples,
        waveform=waveform,
        window_samples=window_samples,
        hop_samples=hop_samples,
        device=str(loaded_model.device),
        postprocessing=postprocessed,
        routing=routing,
    )


def enhance_audio_file(
    audio_path: str | Path,
    model_name: str,
) -> EnhancedAudio:
    """Decode, preprocess, and enhance one audio file."""

    decoded_audio = decode_audio_for_enhancement(str(audio_path))

    preprocessed_audio = preprocess_audio_for_enhancement(decoded_audio)

    loaded_model = load_model(
        model_name,
    )

    return enhance_preprocessed_audio(
        preprocessed_audio=preprocessed_audio,
        loaded_model=loaded_model,
    )


def enhance_audio_to_file(
    audio_path: str | Path,
    model_name: str,
) -> EnhancedAudioFile:
    """Enhance an audio file and encode the result as a temporary WAV."""

    enhanced_audio = enhance_audio_file(
        audio_path=audio_path,
        model_name=model_name,
    )

    encoded_audio = encode_audio_to_temporary_wav(
        waveform=enhanced_audio.waveform,
        sample_rate=enhanced_audio.sample_rate,
    )

    return EnhancedAudioFile(
        audio=enhanced_audio,
        encoded=encoded_audio,
    )
