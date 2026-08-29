"""
File: application.py

Author: Ksenia Zolina, Dmitry Ryumin, Denis Ivanko, and Alexey Karpov

Description: Application event handlers for WAVES.

License: MIT License
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr

from waves.audio.encoder import (
    remove_temporary_encoded_audio_file,
)
from waves.audio.validation import validate_audio_file
from waves.config import get_config_str
from waves.events.audio_state import (
    create_audio_component_label,
    create_audio_state_content,
    get_audio_display_filename,
)
from waves.events.visualization import (
    RoutingPlotUpdates,
    create_hidden_plot_update,
    create_hidden_routing_plot_updates,
    create_routing_plot_updates,
    create_spectrogram_plot_update_from_enhancement,
)
from waves.inference import enhance_audio_to_file
from waves.localization import (
    get_language_index,
    get_localized_text,
)
from waves.logger import get_logger
from waves.models import discover_models
from waves.routing import RoutingTelemetry

LOGGER = get_logger(__name__)

ERROR_STATUS_MARKER = '<span class="application-status-error-marker"></span>'

RECORDED_AUDIO_FILENAME = "recording.wav"


@dataclass(frozen=True, slots=True)
class AudioChangeUpdates:
    """Named UI updates produced by an input-audio change."""

    audio_input: Any
    audio_filename_state: str | None
    status: Any
    run_button: Any
    clear_button: Any
    audio_info_button_column: Any
    audio_info_button: Any
    audio_info_modal: Any
    audio_info_modal_content: Any
    enhanced_audio: Any
    spectrogram_plot: Any
    routing_state: RoutingTelemetry | None
    routing_plots: RoutingPlotUpdates


@dataclass(frozen=True, slots=True)
class EnhancementUpdates:
    """Named UI updates produced while running speech enhancement."""

    audio_input: Any
    status: Any
    run_button: Any
    clear_button: Any
    enhanced_audio: Any
    spectrogram_plot: Any
    routing_state: RoutingTelemetry | None
    routing_plots: RoutingPlotUpdates


@dataclass(frozen=True, slots=True)
class ClearApplicationUpdates:
    """Named UI updates produced when clearing the application."""

    audio_input: Any
    audio_filename_state: str | None
    status: Any
    run_button: Any
    clear_button: Any
    audio_info_button_column: Any
    audio_info_button: Any
    audio_info_modal: Any
    audio_info_modal_content: Any
    enhanced_audio: Any
    spectrogram_plot: Any
    routing_state: RoutingTelemetry | None
    routing_plots: RoutingPlotUpdates


def resolve_audio_display_filename(
    audio_path: str | None,
    audio_filename: str | None,
) -> str | None:
    """Resolve the display filename for the current audio sample."""

    if not audio_path:
        return None

    if audio_filename:
        normalized_filename = Path(audio_filename).name.strip()

        if normalized_filename:
            return normalized_filename

    return get_audio_display_filename(audio_path)


def resolve_application_model() -> str:
    """Resolve the production model used for speech enhancement."""

    configured_model = get_config_str(
        "Model_DEFAULT_MODEL",
        "",
    ).strip()

    if configured_model:
        return configured_model

    models = discover_models()

    if not models:
        msg = "No production models were found."
        raise RuntimeError(msg)

    if len(models) > 1:
        msg = "Multiple production models were found. Set Model.DEFAULT_MODEL in config.toml."
        raise RuntimeError(msg)

    return models[0].name


def handle_audio_change(
    audio_path: str | None,
    enhanced_audio_path: str | None,
    language: str,
    *,
    audio_filename: str | None = None,
) -> AudioChangeUpdates:
    """Handle an uploaded, recorded, selected, or removed audio sample."""

    language_index = get_language_index(language)

    remove_temporary_encoded_audio_file(enhanced_audio_path)

    resolved_audio_filename = resolve_audio_display_filename(
        audio_path=audio_path,
        audio_filename=audio_filename,
    )

    audio_state = create_audio_state_content(
        audio_path=audio_path,
        language_index=language_index,
    )

    noisy_audio_label = create_audio_component_label(
        "Labels_NOISY_AUDIO",
        language_index,
        resolved_audio_filename,
    )

    enhanced_audio_label = create_audio_component_label(
        "Labels_ENHANCED_AUDIO",
        language_index,
        resolved_audio_filename,
    )

    routing_plot_updates = create_hidden_routing_plot_updates()

    if not audio_path:
        return AudioChangeUpdates(
            audio_input=gr.update(
                label=noisy_audio_label,
            ),
            audio_filename_state=None,
            status=gr.update(
                value=audio_state.status_text,
            ),
            run_button=gr.update(
                interactive=False,
            ),
            clear_button=gr.update(
                interactive=False,
            ),
            audio_info_button_column=gr.update(
                visible=False,
            ),
            audio_info_button=gr.update(
                interactive=False,
            ),
            audio_info_modal=gr.update(
                visible=False,
            ),
            audio_info_modal_content=gr.update(
                value="",
            ),
            enhanced_audio=gr.update(
                value=None,
                label=enhanced_audio_label,
                visible=False,
            ),
            spectrogram_plot=(create_hidden_plot_update()),
            routing_state=None,
            routing_plots=routing_plot_updates,
        )

    return AudioChangeUpdates(
        audio_input=gr.update(
            label=noisy_audio_label,
        ),
        audio_filename_state=(resolved_audio_filename),
        status=gr.update(
            value=audio_state.status_text,
        ),
        run_button=gr.update(
            interactive=audio_state.is_valid,
        ),
        clear_button=gr.update(
            interactive=True,
        ),
        audio_info_button_column=gr.update(
            visible=True,
        ),
        audio_info_button=gr.update(
            interactive=True,
        ),
        audio_info_modal=gr.update(
            visible=False,
        ),
        audio_info_modal_content=gr.update(
            value=audio_state.audio_info_html,
        ),
        enhanced_audio=gr.update(
            value=None,
            label=enhanced_audio_label,
            visible=False,
        ),
        spectrogram_plot=(create_hidden_plot_update()),
        routing_state=None,
        routing_plots=routing_plot_updates,
    )


def handle_run_enhancement(
    audio_path: str | None,
    enhanced_audio_path: str | None,
    language: str,
    audio_filename: str | None,
) -> Iterator[EnhancementUpdates]:
    """Run speech enhancement and stream named UI state updates."""

    language_index = get_language_index(language)

    resolved_audio_filename = resolve_audio_display_filename(
        audio_path=audio_path,
        audio_filename=audio_filename,
    )

    noisy_audio_label = create_audio_component_label(
        "Labels_NOISY_AUDIO",
        language_index,
        resolved_audio_filename,
    )

    enhanced_audio_label = create_audio_component_label(
        "Labels_ENHANCED_AUDIO",
        language_index,
        resolved_audio_filename,
    )

    yield EnhancementUpdates(
        audio_input=gr.update(
            interactive=False,
            label=noisy_audio_label,
        ),
        status=gr.update(
            value=get_localized_text(
                "Texts_STATUS_PROCESSING",
                language_index,
            ),
        ),
        run_button=gr.update(
            interactive=False,
        ),
        clear_button=gr.update(
            interactive=False,
        ),
        enhanced_audio=gr.update(
            value=None,
            label=enhanced_audio_label,
            visible=False,
        ),
        spectrogram_plot=(create_hidden_plot_update()),
        routing_state=None,
        routing_plots=(create_hidden_routing_plot_updates()),
    )

    try:
        if not audio_path:
            msg = "No input audio was provided."
            raise ValueError(msg)

        validation_result = validate_audio_file(audio_path)

        if not validation_result.is_valid:
            msg = "The input audio failed validation."
            raise ValueError(msg)

        model_name = resolve_application_model()

        remove_temporary_encoded_audio_file(enhanced_audio_path)

        result = enhance_audio_to_file(
            audio_path=audio_path,
            model_name=model_name,
        )

        spectrogram_update = create_spectrogram_plot_update_from_enhancement(
            audio_path=audio_path,
            enhanced_waveform=(result.audio.waveform),
            sample_rate=(result.audio.sample_rate),
            language_index=language_index,
        )

        routing = result.audio.routing

        routing_plot_updates = create_routing_plot_updates(
            routing=routing,
            language_index=language_index,
            sample_rate=(result.audio.sample_rate),
        )

        LOGGER.info(
            (
                "Speech enhancement completed: "
                "input='%s', output='%s', "
                "model='%s', weights='%s', "
                "device='%s', samples=%d"
            ),
            audio_path,
            result.encoded.path,
            result.audio.model_name,
            result.audio.model_weights_path,
            result.audio.device,
            result.audio.num_samples,
        )

        yield EnhancementUpdates(
            audio_input=gr.update(
                interactive=True,
                label=noisy_audio_label,
            ),
            status=gr.update(
                value=get_localized_text(
                    "Texts_STATUS_COMPLETED",
                    language_index,
                ),
            ),
            run_button=gr.update(
                interactive=True,
            ),
            clear_button=gr.update(
                interactive=True,
            ),
            enhanced_audio=gr.update(
                value=result.encoded.path,
                label=enhanced_audio_label,
                visible=True,
            ),
            spectrogram_plot=(spectrogram_update),
            routing_state=routing,
            routing_plots=(routing_plot_updates),
        )

    except Exception:
        LOGGER.exception("Speech enhancement failed.")

        status_text = f"{ERROR_STATUS_MARKER}\n{get_localized_text('Texts_STATUS_ENHANCEMENT_FAILED', language_index)}"

        yield EnhancementUpdates(
            audio_input=gr.update(
                interactive=True,
                label=noisy_audio_label,
            ),
            status=gr.update(
                value=status_text,
            ),
            run_button=gr.update(
                interactive=True,
            ),
            clear_button=gr.update(
                interactive=True,
            ),
            enhanced_audio=gr.update(
                value=None,
                label=enhanced_audio_label,
                visible=False,
            ),
            spectrogram_plot=(create_hidden_plot_update()),
            routing_state=None,
            routing_plots=(create_hidden_routing_plot_updates()),
        )


def handle_clear_application(
    language: str,
    enhanced_audio_path: str | None,
) -> ClearApplicationUpdates:
    """Clear the current application audio state."""

    language_index = get_language_index(language)

    noisy_audio_label = create_audio_component_label(
        "Labels_NOISY_AUDIO",
        language_index,
        None,
    )

    enhanced_audio_label = create_audio_component_label(
        "Labels_ENHANCED_AUDIO",
        language_index,
        None,
    )

    remove_temporary_encoded_audio_file(enhanced_audio_path)

    audio_state = create_audio_state_content(
        audio_path=None,
        language_index=language_index,
    )

    return ClearApplicationUpdates(
        audio_input=gr.update(
            value=None,
            label=noisy_audio_label,
        ),
        audio_filename_state=None,
        status=gr.update(
            value=audio_state.status_text,
        ),
        run_button=gr.update(
            interactive=False,
        ),
        clear_button=gr.update(
            interactive=False,
        ),
        audio_info_button_column=gr.update(
            visible=False,
        ),
        audio_info_button=gr.update(
            interactive=False,
        ),
        audio_info_modal=gr.update(
            visible=False,
        ),
        audio_info_modal_content=gr.update(
            value="",
        ),
        enhanced_audio=gr.update(
            value=None,
            label=enhanced_audio_label,
            visible=False,
        ),
        spectrogram_plot=(create_hidden_plot_update()),
        routing_state=None,
        routing_plots=(create_hidden_routing_plot_updates()),
    )


def handle_show_audio_info() -> Any:
    """Open the audio information modal."""

    return gr.update(
        visible=True,
    )


def handle_hide_audio_info() -> Any:
    """Close the audio information modal."""

    return gr.update(
        visible=False,
    )
