"""
File: pipeline.py

Author: Ksenia Zolina, Dmitry Ryumin, Denis Ivanko, and Alexey Karpov

Description: Observable WAVES speech-enhancement processing pipeline.

License: MIT License
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch

from waves.audio.decoder import (
    decode_audio_for_enhancement,
)
from waves.audio.encoder import (
    encode_audio_to_temporary_wav,
)
from waves.audio.preprocessing import (
    preprocess_audio_for_enhancement,
)
from waves.audio.validation import (
    validate_audio_file,
)
from waves.inference.engine import (
    EnhancedAudio,
    EnhancedAudioFile,
    SlidingWindowInferenceResult,
    iter_sliding_window_inference,
)
from waves.inference.postprocessing import (
    postprocess_enhanced_waveform,
)
from waves.inference.progress import (
    EnhancementProgressEvent,
    EnhancementProgressTracker,
    EnhancementStage,
)
from waves.models.loader import (
    load_model_with_status,
)


@dataclass(frozen=True, slots=True)
class EnhancementPipelineResult:
    """Completed WAVES enhancement pipeline result."""

    output: EnhancedAudioFile
    progress_tracker: EnhancementProgressTracker


def iter_enhancement_pipeline(
    audio_path: str | Path,
    model_name: str,
) -> Iterator[EnhancementProgressEvent | EnhancementPipelineResult]:
    """Run WAVES enhancement while emitting observable progress events."""

    path = str(audio_path)

    tracker = EnhancementProgressTracker()

    yield tracker.begin_stage(EnhancementStage.VALIDATION)

    validation_result = validate_audio_file(path)

    if not validation_result.is_valid:
        msg = "The input audio failed validation."
        raise ValueError(msg)

    yield tracker.complete_stage(EnhancementStage.VALIDATION)

    yield tracker.begin_stage(EnhancementStage.AUDIO_DECODING)

    decoded_audio = decode_audio_for_enhancement(path)

    yield tracker.complete_stage(EnhancementStage.AUDIO_DECODING)

    yield tracker.begin_stage(EnhancementStage.PREPROCESSING)

    preprocessed_audio = preprocess_audio_for_enhancement(decoded_audio)

    yield tracker.complete_stage(EnhancementStage.PREPROCESSING)

    yield tracker.begin_stage(EnhancementStage.MODEL_LOADING)

    model_load_result = load_model_with_status(model_name)

    loaded_model = model_load_result.loaded_model

    yield tracker.complete_stage(
        EnhancementStage.MODEL_LOADING,
        cached=model_load_result.cached,
    )

    yield tracker.begin_stage(EnhancementStage.ENHANCEMENT)

    inference_result: SlidingWindowInferenceResult | None = None

    completed_windows = 0
    total_windows = 0

    for update in iter_sliding_window_inference(
        preprocessed_audio=(preprocessed_audio),
        loaded_model=loaded_model,
    ):
        if update.result is not None:
            inference_result = update.result
            continue

        completed_windows = update.completed_windows

        total_windows = update.total_windows

        stage_progress = completed_windows / total_windows

        yield tracker.update_stage(
            EnhancementStage.ENHANCEMENT,
            stage_progress,
            completed_units=(completed_windows),
            total_units=(total_windows),
        )

    if inference_result is None:
        msg = "WAVES inference did not produce an enhanced waveform."
        raise RuntimeError(msg)

    yield tracker.complete_stage(
        EnhancementStage.ENHANCEMENT,
        completed_units=(completed_windows),
        total_units=(total_windows),
    )

    yield tracker.begin_stage(EnhancementStage.POSTPROCESSING)

    postprocessed = postprocess_enhanced_waveform(
        normalized_waveform=(inference_result.waveform),
        preprocessed_audio=(preprocessed_audio),
    )

    waveform = (
        postprocessed.waveform.detach()
        .to(
            device="cpu",
            dtype=torch.float32,
        )
        .contiguous()
    )

    enhanced_audio = EnhancedAudio(
        input_path=(preprocessed_audio.path),
        model_name=(loaded_model.info.name),
        model_weights_path=str(loaded_model.info.weights_path),
        sample_rate=(preprocessed_audio.sample_rate),
        duration_seconds=(preprocessed_audio.duration_seconds),
        num_samples=(preprocessed_audio.num_samples),
        waveform=waveform,
        window_samples=(inference_result.window_samples),
        hop_samples=(inference_result.hop_samples),
        device=str(loaded_model.device),
        postprocessing=postprocessed,
        routing=(inference_result.routing),
    )

    yield tracker.complete_stage(EnhancementStage.POSTPROCESSING)

    yield tracker.begin_stage(EnhancementStage.WAV_ENCODING)

    encoded_audio = encode_audio_to_temporary_wav(
        waveform=(enhanced_audio.waveform),
        sample_rate=(enhanced_audio.sample_rate),
    )

    output = EnhancedAudioFile(
        audio=enhanced_audio,
        encoded=encoded_audio,
    )

    yield tracker.complete_stage(EnhancementStage.WAV_ENCODING)

    yield EnhancementPipelineResult(
        output=output,
        progress_tracker=tracker,
    )
