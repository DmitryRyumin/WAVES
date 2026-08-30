"""
File: application.py

Author: Ksenia Zolina, Dmitry Ryumin, Denis Ivanko, and Alexey Karpov

Description: Application event handlers for WAVES.

License: MIT License
"""

from collections.abc import Iterator
from pathlib import Path
from time import sleep
from typing import Any

import gradio as gr

from waves.audio.encoder import (
    remove_temporary_encoded_audio_file,
)
from waves.config import get_config_str
from waves.events.application_updates import (
    AudioChangeUpdates,
    ClearApplicationUpdates,
    EnhancementUpdates,
    ProcessingModalUpdates,
    create_hidden_visualization_info_button_updates,
    create_result_visualization_info_button_updates,
    create_unchanged_routing_plot_updates,
    create_unchanged_visualization_info_button_updates,
)
from waves.events.audio_state import (
    create_audio_component_label,
    create_audio_state_content,
    get_audio_display_filename,
)
from waves.events.visualization import (
    create_hidden_plot_update,
    create_hidden_routing_plot_updates,
    create_routing_plot_updates,
    create_spectrogram_plot_update_from_enhancement,
)
from waves.inference import (
    EnhancementPipelineResult,
    EnhancementProgressEvent,
    EnhancementProgressTracker,
    EnhancementStage,
    iter_enhancement_pipeline,
)
from waves.localization import (
    get_language_index,
    get_localized_text,
)
from waves.logger import get_logger
from waves.models import discover_models
from waves.ui.progress_modal import (
    ProcessingSummary,
    create_completed_processing_modal_html,
    create_processing_modal_html,
    create_processing_time_button_label,
)

LOGGER = get_logger(__name__)

ERROR_STATUS_MARKER = '<span class="application-status-error-marker"></span>'

RECORDED_AUDIO_FILENAME = "recording.wav"

PROCESSING_COMPLETION_TRANSITION_SECONDS = 0.58


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
            processing_time_button_column=(
                gr.update(
                    visible=False,
                )
            ),
            processing_time_button=gr.update(
                value=get_localized_text(
                    "Labels_PROCESSING_TIME",
                    language_index,
                ),
                interactive=False,
            ),
            processing_modal=gr.update(
                visible=False,
            ),
            processing_modal_content=gr.update(
                value="",
            ),
            processing_modal_close_button=(
                gr.update(
                    visible=False,
                )
            ),
            processing_summary_state=None,
            visualization_info_key_state=None,
            visualization_info_modal=gr.update(
                visible=False,
            ),
            visualization_info_modal_content=(
                gr.update(
                    value="",
                )
            ),
            visualization_info_modal_close_button=(
                gr.update(
                    visible=False,
                )
            ),
            visualization_info_buttons=(create_hidden_visualization_info_button_updates()),
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
        processing_time_button_column=(
            gr.update(
                visible=False,
            )
        ),
        processing_time_button=gr.update(
            value=get_localized_text(
                "Labels_PROCESSING_TIME",
                language_index,
            ),
            interactive=False,
        ),
        processing_modal=gr.update(
            visible=False,
        ),
        processing_modal_content=gr.update(
            value="",
        ),
        processing_modal_close_button=(
            gr.update(
                visible=False,
            )
        ),
        processing_summary_state=None,
        visualization_info_key_state=None,
        visualization_info_modal=gr.update(
            visible=False,
        ),
        visualization_info_modal_content=(
            gr.update(
                value="",
            )
        ),
        visualization_info_modal_close_button=(
            gr.update(
                visible=False,
            )
        ),
        visualization_info_buttons=(create_hidden_visualization_info_button_updates()),
        enhanced_audio=gr.update(
            value=None,
            label=enhanced_audio_label,
            visible=False,
        ),
        spectrogram_plot=(create_hidden_plot_update()),
        routing_state=None,
        routing_plots=routing_plot_updates,
    )


def handle_enhancement_started(
    audio_path: str | None,
    enhanced_audio_path: str | None,
    language: str,
    audio_filename: str | None,
) -> EnhancementUpdates:
    """Immediately lock the interface and show initial processing state."""

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

    remove_temporary_encoded_audio_file(enhanced_audio_path)

    initial_tracker = EnhancementProgressTracker()

    initial_event = initial_tracker.begin_stage(EnhancementStage.VALIDATION)

    return EnhancementUpdates(
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
        audio_info_button=gr.update(
            interactive=False,
        ),
        audio_info_modal=gr.update(
            visible=False,
        ),
        processing_time_button_column=(
            gr.update(
                visible=False,
            )
        ),
        processing_time_button=gr.update(
            interactive=False,
        ),
        processing_modal=gr.update(
            visible=True,
        ),
        processing_modal_content=gr.update(
            value=create_processing_modal_html(
                initial_event,
                language_index,
            ),
        ),
        processing_modal_close_button=(
            gr.update(
                visible=False,
            )
        ),
        processing_summary_state=None,
        visualization_info_key_state=None,
        visualization_info_modal=gr.update(
            visible=False,
        ),
        visualization_info_modal_content=(
            gr.update(
                value="",
            )
        ),
        visualization_info_modal_close_button=(
            gr.update(
                visible=False,
            )
        ),
        visualization_info_buttons=(create_hidden_visualization_info_button_updates()),
        enhanced_audio=gr.update(
            value=None,
            label=enhanced_audio_label,
            visible=False,
        ),
        spectrogram_plot=(create_hidden_plot_update()),
        routing_state=None,
        routing_plots=(create_hidden_routing_plot_updates()),
    )


def create_running_enhancement_update(
    event: EnhancementProgressEvent,
    language_index: int,
) -> EnhancementUpdates:
    """Create one streamed running-state UI update."""

    return EnhancementUpdates(
        audio_input=gr.update(),
        status=gr.update(),
        run_button=gr.update(),
        clear_button=gr.update(),
        audio_info_button=gr.update(),
        audio_info_modal=gr.update(),
        processing_time_button_column=(gr.update()),
        processing_time_button=gr.update(),
        processing_modal=gr.update(
            visible=True,
        ),
        processing_modal_content=gr.update(
            value=create_processing_modal_html(
                event,
                language_index,
            ),
        ),
        processing_modal_close_button=(
            gr.update(
                visible=False,
            )
        ),
        processing_summary_state=None,
        visualization_info_key_state=None,
        visualization_info_modal=gr.update(),
        visualization_info_modal_content=(gr.update()),
        visualization_info_modal_close_button=(gr.update()),
        visualization_info_buttons=(create_unchanged_visualization_info_button_updates()),
        enhanced_audio=gr.update(),
        spectrogram_plot=gr.update(),
        routing_state=None,
        routing_plots=(create_unchanged_routing_plot_updates()),
    )


def handle_run_enhancement(
    audio_path: str | None,
    language: str,
    audio_filename: str | None,
) -> Iterator[EnhancementUpdates]:
    """Run speech enhancement and stream progress updates."""

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

    try:
        if not audio_path:
            msg = "No input audio was provided."
            raise ValueError(msg)

        model_name = resolve_application_model()

        pipeline_result: EnhancementPipelineResult | None = None

        for pipeline_update in iter_enhancement_pipeline(
            audio_path=audio_path,
            model_name=model_name,
        ):
            if isinstance(
                pipeline_update,
                EnhancementProgressEvent,
            ):
                yield (
                    create_running_enhancement_update(
                        pipeline_update,
                        language_index,
                    )
                )

                continue

            pipeline_result = pipeline_update

        if pipeline_result is None:
            msg = "Speech enhancement pipeline did not produce a result."
            raise RuntimeError(msg)

        result = pipeline_result.output

        tracker = pipeline_result.progress_tracker

        spectrogram_begin = tracker.begin_stage(EnhancementStage.SPECTROGRAM_RENDERING)

        yield (
            create_running_enhancement_update(
                spectrogram_begin,
                language_index,
            )
        )

        spectrogram_update = create_spectrogram_plot_update_from_enhancement(
            audio_path=audio_path,
            enhanced_waveform=(result.audio.waveform),
            sample_rate=(result.audio.sample_rate),
            language_index=language_index,
        )

        spectrogram_complete = tracker.complete_stage(EnhancementStage.SPECTROGRAM_RENDERING)

        yield (
            create_running_enhancement_update(
                spectrogram_complete,
                language_index,
            )
        )

        routing_begin = tracker.begin_stage(EnhancementStage.ROUTING_VISUALIZATION)

        yield (
            create_running_enhancement_update(
                routing_begin,
                language_index,
            )
        )

        routing = result.audio.routing

        routing_plot_updates = create_routing_plot_updates(
            routing=routing,
            language_index=language_index,
            sample_rate=(result.audio.sample_rate),
        )

        final_event = tracker.complete_stage(EnhancementStage.ROUTING_VISUALIZATION)

        summary = ProcessingSummary(
            event=final_event,
            device=result.audio.device,
        )

        LOGGER.info(
            (
                "Speech enhancement completed: "
                "input='%s', output='%s', "
                "model='%s', weights='%s', "
                "device='%s', samples=%d, "
                "elapsed=%.3f s"
            ),
            audio_path,
            result.encoded.path,
            result.audio.model_name,
            result.audio.model_weights_path,
            result.audio.device,
            result.audio.num_samples,
            final_event.elapsed_seconds,
        )

        # Show the completed state briefly so the user sees
        # the final stage transition, green progress bar,
        # and completion animation before the overlay closes.
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
            audio_info_button=gr.update(
                interactive=True,
            ),
            audio_info_modal=gr.update(
                visible=False,
            ),
            processing_time_button_column=(
                gr.update(
                    visible=True,
                )
            ),
            processing_time_button=gr.update(
                value=(
                    create_processing_time_button_label(
                        summary,
                        language_index,
                    )
                ),
                interactive=True,
            ),
            processing_modal=gr.update(
                visible=True,
            ),
            processing_modal_content=gr.update(
                value=(
                    create_completed_processing_modal_html(
                        summary,
                        language_index,
                        auto_close=True,
                    )
                ),
            ),
            processing_modal_close_button=(
                gr.update(
                    visible=False,
                )
            ),
            processing_summary_state=summary,
            visualization_info_key_state=None,
            visualization_info_modal=gr.update(
                visible=False,
            ),
            visualization_info_modal_content=(
                gr.update(
                    value="",
                )
            ),
            visualization_info_modal_close_button=(
                gr.update(
                    visible=False,
                )
            ),
            visualization_info_buttons=(
                create_result_visualization_info_button_updates(
                    spectrogram_plot=(spectrogram_update),
                    routing_plots=(routing_plot_updates),
                )
            ),
            enhanced_audio=gr.update(
                value=result.encoded.path,
                label=enhanced_audio_label,
                visible=True,
            ),
            spectrogram_plot=spectrogram_update,
            routing_state=routing,
            routing_plots=routing_plot_updates,
        )

        sleep(PROCESSING_COMPLETION_TRANSITION_SECONDS)

        # Hide the processing overlay while retaining the
        # completed summary for the Processing time button.
        yield EnhancementUpdates(
            audio_input=gr.update(),
            status=gr.update(),
            run_button=gr.update(),
            clear_button=gr.update(),
            audio_info_button=gr.update(),
            audio_info_modal=gr.update(),
            processing_time_button_column=(gr.update()),
            processing_time_button=gr.update(),
            processing_modal=gr.update(
                visible=False,
            ),
            processing_modal_content=gr.update(
                value="",
            ),
            processing_modal_close_button=(
                gr.update(
                    visible=False,
                )
            ),
            processing_summary_state=summary,
            visualization_info_key_state=None,
            visualization_info_modal=gr.update(),
            visualization_info_modal_content=(gr.update()),
            visualization_info_modal_close_button=(gr.update()),
            visualization_info_buttons=(create_unchanged_visualization_info_button_updates()),
            enhanced_audio=gr.update(),
            spectrogram_plot=gr.update(),
            routing_state=routing,
            routing_plots=(create_unchanged_routing_plot_updates()),
        )

    except Exception:
        LOGGER.exception("Speech enhancement failed.")

        status_text = (
            f"{ERROR_STATUS_MARKER}\n"
            f"{
                get_localized_text(
                    'Texts_STATUS_ENHANCEMENT_FAILED',
                    language_index,
                )
            }"
        )

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
            audio_info_button=gr.update(
                interactive=True,
            ),
            audio_info_modal=gr.update(
                visible=False,
            ),
            processing_time_button_column=(
                gr.update(
                    visible=False,
                )
            ),
            processing_time_button=gr.update(
                interactive=False,
            ),
            processing_modal=gr.update(
                visible=False,
            ),
            processing_modal_content=gr.update(
                value="",
            ),
            processing_modal_close_button=(
                gr.update(
                    visible=False,
                )
            ),
            processing_summary_state=None,
            visualization_info_key_state=None,
            visualization_info_modal=gr.update(
                visible=False,
            ),
            visualization_info_modal_content=(
                gr.update(
                    value="",
                )
            ),
            visualization_info_modal_close_button=(
                gr.update(
                    visible=False,
                )
            ),
            visualization_info_buttons=(create_hidden_visualization_info_button_updates()),
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
        processing_time_button_column=(
            gr.update(
                visible=False,
            )
        ),
        processing_time_button=gr.update(
            value=get_localized_text(
                "Labels_PROCESSING_TIME",
                language_index,
            ),
            interactive=False,
        ),
        processing_modal=gr.update(
            visible=False,
        ),
        processing_modal_content=gr.update(
            value="",
        ),
        processing_modal_close_button=(
            gr.update(
                visible=False,
            )
        ),
        processing_summary_state=None,
        visualization_info_key_state=None,
        visualization_info_modal=gr.update(
            visible=False,
        ),
        visualization_info_modal_content=(
            gr.update(
                value="",
            )
        ),
        visualization_info_modal_close_button=(
            gr.update(
                visible=False,
            )
        ),
        visualization_info_buttons=(create_hidden_visualization_info_button_updates()),
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


def handle_show_processing_timing(
    summary: ProcessingSummary | None,
    language: str,
) -> ProcessingModalUpdates:
    """Open the completed processing timing modal."""

    if summary is None:
        return ProcessingModalUpdates(
            modal=gr.update(
                visible=False,
            ),
            content=gr.update(
                value="",
            ),
            close_button=gr.update(
                visible=False,
            ),
        )

    language_index = get_language_index(language)

    return ProcessingModalUpdates(
        modal=gr.update(
            visible=True,
        ),
        content=gr.update(
            value=(
                create_completed_processing_modal_html(
                    summary,
                    language_index,
                )
            ),
        ),
        close_button=gr.update(
            visible=True,
        ),
    )


def handle_hide_processing_modal() -> Any:
    """Close the processing modal."""

    return gr.update(
        visible=False,
    )
