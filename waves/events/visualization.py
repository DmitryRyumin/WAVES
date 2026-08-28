"""
File: visualization.py

Author: Ksenia Zolina, Dmitry Ryumin, Denis Ivanko, and Alexey Karpov

Description: Shared visualization update builders for WAVES application events.

License: MIT License
"""

from dataclasses import dataclass
from typing import Any

import gradio as gr
from torch import Tensor

from waves.audio import decode_audio_for_enhancement
from waves.config import get_config_bool
from waves.logger import get_logger
from waves.routing import RoutingTelemetry
from waves.visualization import (
    create_expert_occupancy_figure,
    create_frequency_routing_figure,
    create_layer_routing_figure,
    create_load_over_time_figure,
    create_spectrogram_comparison_figure,
)

LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RoutingPlotUpdates:
    """Named updates for all Mixture-of-Experts routing plots."""

    expert_occupancy_plot: Any
    layer_routing_plot: Any
    frequency_routing_plot: Any
    load_over_time_plot: Any


def create_hidden_plot_update() -> Any:
    """Create a hidden empty Plot update."""

    return gr.update(
        value=None,
        visible=False,
    )


def create_hidden_routing_plot_updates() -> RoutingPlotUpdates:
    """Create hidden updates for all routing plots."""

    return RoutingPlotUpdates(
        expert_occupancy_plot=create_hidden_plot_update(),
        layer_routing_plot=create_hidden_plot_update(),
        frequency_routing_plot=create_hidden_plot_update(),
        load_over_time_plot=create_hidden_plot_update(),
    )


def _create_spectrogram_plot_update(
    noisy_waveform: Tensor,
    enhanced_waveform: Tensor,
    sample_rate: int,
    language_index: int,
) -> Any:
    """Create a visible localized spectrogram update from prepared waveforms."""

    try:
        figure = create_spectrogram_comparison_figure(
            noisy_waveform=noisy_waveform,
            enhanced_waveform=enhanced_waveform,
            sample_rate=sample_rate,
            language_index=language_index,
        )
    except Exception:
        LOGGER.exception("Failed to create spectrogram comparison.")

        return create_hidden_plot_update()

    return gr.update(
        value=figure,
        visible=True,
    )


def create_spectrogram_plot_update_from_enhancement(
    audio_path: str,
    enhanced_waveform: Tensor,
    sample_rate: int,
    language_index: int,
) -> Any:
    """Create a spectrogram update after speech enhancement."""

    if not get_config_bool(
        "Visualization_SHOW_SPECTROGRAM",
        True,
    ):
        return create_hidden_plot_update()

    try:
        noisy_audio = decode_audio_for_enhancement(audio_path)
    except Exception:
        LOGGER.exception("Failed to decode noisy audio for spectrogram comparison.")

        return create_hidden_plot_update()

    return _create_spectrogram_plot_update(
        noisy_waveform=noisy_audio.waveform,
        enhanced_waveform=enhanced_waveform,
        sample_rate=sample_rate,
        language_index=language_index,
    )


def create_spectrogram_plot_update_from_paths(
    audio_path: str | None,
    enhanced_audio_path: str | None,
    language_index: int,
) -> Any:
    """Recreate a localized spectrogram update from stored audio paths."""

    if (
        not audio_path
        or not enhanced_audio_path
        or not get_config_bool(
            "Visualization_SHOW_SPECTROGRAM",
            True,
        )
    ):
        return create_hidden_plot_update()

    try:
        noisy_audio = decode_audio_for_enhancement(audio_path)
        enhanced_audio = decode_audio_for_enhancement(enhanced_audio_path)
    except Exception:
        LOGGER.exception("Failed to decode audio for localized spectrogram comparison.")

        return create_hidden_plot_update()

    return _create_spectrogram_plot_update(
        noisy_waveform=noisy_audio.waveform,
        enhanced_waveform=enhanced_audio.waveform,
        sample_rate=noisy_audio.sample_rate,
        language_index=language_index,
    )


def create_routing_plot_updates(
    routing: RoutingTelemetry | None,
    language_index: int,
    sample_rate: int | None = None,
) -> RoutingPlotUpdates:
    """Create localized updates for all configured routing plots."""

    updates = create_hidden_routing_plot_updates()

    if routing is None or not get_config_bool(
        "Visualization_SHOW_ROUTING",
        True,
    ):
        return updates

    expert_occupancy_update = updates.expert_occupancy_plot
    layer_routing_update = updates.layer_routing_plot
    frequency_routing_update = updates.frequency_routing_plot
    load_over_time_update = updates.load_over_time_plot

    if get_config_bool(
        "MoERouting_SHOW_EXPERT_OCCUPANCY",
        True,
    ):
        try:
            figure = create_expert_occupancy_figure(
                telemetry=routing,
                language_index=language_index,
            )

            expert_occupancy_update = gr.update(
                value=figure,
                visible=True,
            )
        except Exception:
            LOGGER.exception("Failed to create expert occupancy visualization.")

    if get_config_bool(
        "MoERouting_SHOW_LAYER_ROUTING",
        True,
    ):
        try:
            figure = create_layer_routing_figure(
                telemetry=routing,
                language_index=language_index,
            )

            layer_routing_update = gr.update(
                value=figure,
                visible=True,
            )
        except Exception:
            LOGGER.exception("Failed to create layer routing visualization.")

    if get_config_bool(
        "MoERouting_SHOW_EXPERT_LOAD_BY_FREQUENCY",
        True,
    ):
        try:
            figure = create_frequency_routing_figure(
                telemetry=routing,
                language_index=language_index,
                sample_rate=sample_rate,
            )

            frequency_routing_update = gr.update(
                value=figure,
                visible=True,
            )
        except Exception:
            LOGGER.exception("Failed to create frequency routing visualization.")

    if get_config_bool(
        "MoERouting_SHOW_EXPERT_LOAD_OVER_TIME",
        True,
    ):
        try:
            figure = create_load_over_time_figure(
                telemetry=routing,
                language_index=language_index,
                sample_rate=sample_rate,
            )

            load_over_time_update = gr.update(
                value=figure,
                visible=True,
            )
        except Exception:
            LOGGER.exception("Failed to create temporal routing visualization.")

    return RoutingPlotUpdates(
        expert_occupancy_plot=expert_occupancy_update,
        layer_routing_plot=layer_routing_update,
        frequency_routing_plot=frequency_routing_update,
        load_over_time_plot=load_over_time_update,
    )
