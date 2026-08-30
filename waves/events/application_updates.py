"""
File: application_updates.py

Author: Ksenia Zolina, Dmitry Ryumin, Denis Ivanko, and Alexey Karpov

Description: Named Gradio update payloads for the WAVES application.

License: MIT License
"""

from dataclasses import dataclass
from typing import Any

import gradio as gr

from waves.events.visualization import RoutingPlotUpdates
from waves.routing import RoutingTelemetry
from waves.ui.progress_modal import ProcessingSummary


@dataclass(frozen=True, slots=True)
class VisualizationInfoButtonUpdates:
    """Named visibility updates for visualization information buttons."""

    spectrogram: Any
    expert_occupancy: Any
    layer_routing: Any
    frequency_routing: Any
    load_over_time: Any


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

    processing_time_button_column: Any
    processing_time_button: Any
    processing_modal: Any
    processing_modal_content: Any
    processing_modal_close_button: Any
    processing_summary_state: ProcessingSummary | None

    visualization_info_key_state: str | None
    visualization_info_modal: Any
    visualization_info_modal_content: Any
    visualization_info_modal_close_button: Any
    visualization_info_buttons: VisualizationInfoButtonUpdates

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

    audio_info_button: Any
    audio_info_modal: Any

    processing_time_button_column: Any
    processing_time_button: Any

    processing_modal: Any
    processing_modal_content: Any
    processing_modal_close_button: Any
    processing_summary_state: ProcessingSummary | None

    visualization_info_key_state: str | None
    visualization_info_modal: Any
    visualization_info_modal_content: Any
    visualization_info_modal_close_button: Any
    visualization_info_buttons: VisualizationInfoButtonUpdates

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

    processing_time_button_column: Any
    processing_time_button: Any

    processing_modal: Any
    processing_modal_content: Any
    processing_modal_close_button: Any
    processing_summary_state: ProcessingSummary | None

    visualization_info_key_state: str | None
    visualization_info_modal: Any
    visualization_info_modal_content: Any
    visualization_info_modal_close_button: Any
    visualization_info_buttons: VisualizationInfoButtonUpdates

    enhanced_audio: Any
    spectrogram_plot: Any

    routing_state: RoutingTelemetry | None
    routing_plots: RoutingPlotUpdates


@dataclass(frozen=True, slots=True)
class ProcessingModalUpdates:
    """Named updates used to show the completed processing modal."""

    modal: Any
    content: Any
    close_button: Any


def create_unchanged_routing_plot_updates() -> RoutingPlotUpdates:
    """Create no-op updates for all routing plots."""

    return RoutingPlotUpdates(
        expert_occupancy_plot=gr.update(),
        layer_routing_plot=gr.update(),
        frequency_routing_plot=gr.update(),
        load_over_time_plot=gr.update(),
    )


def _create_uniform_visualization_info_button_updates(
    *,
    visible: bool | None,
) -> VisualizationInfoButtonUpdates:
    """Create identical updates for all visualization information buttons."""

    def create_update() -> Any:
        if visible is None:
            return gr.update()

        return gr.update(
            visible=visible,
        )

    return VisualizationInfoButtonUpdates(
        spectrogram=create_update(),
        expert_occupancy=create_update(),
        layer_routing=create_update(),
        frequency_routing=create_update(),
        load_over_time=create_update(),
    )


def create_hidden_visualization_info_button_updates() -> VisualizationInfoButtonUpdates:
    """Hide all visualization information buttons."""

    return _create_uniform_visualization_info_button_updates(
        visible=False,
    )


def create_unchanged_visualization_info_button_updates() -> VisualizationInfoButtonUpdates:
    """Create no-op updates for all visualization information buttons."""

    return _create_uniform_visualization_info_button_updates(
        visible=None,
    )


def _plot_update_is_visible(
    update: Any,
) -> bool:
    """Return whether a Gradio plot update explicitly makes a plot visible."""

    return (
        isinstance(
            update,
            dict,
        )
        and update.get("visible") is True
    )


def create_result_visualization_info_button_updates(
    *,
    spectrogram_plot: Any,
    routing_plots: RoutingPlotUpdates,
) -> VisualizationInfoButtonUpdates:
    """Match information-button visibility to successfully rendered plots."""

    return VisualizationInfoButtonUpdates(
        spectrogram=gr.update(
            visible=_plot_update_is_visible(spectrogram_plot),
        ),
        expert_occupancy=gr.update(
            visible=_plot_update_is_visible(routing_plots.expert_occupancy_plot),
        ),
        layer_routing=gr.update(
            visible=_plot_update_is_visible(routing_plots.layer_routing_plot),
        ),
        frequency_routing=gr.update(
            visible=_plot_update_is_visible(routing_plots.frequency_routing_plot),
        ),
        load_over_time=gr.update(
            visible=_plot_update_is_visible(routing_plots.load_over_time_plot),
        ),
    )
