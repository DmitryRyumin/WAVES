"""
File: language.py

Author: Ksenia Zolina, Dmitry Ryumin, Denis Ivanko, and Alexey Karpov

Description: Language switching event handlers for the WAVES Gradio application.

License: MIT License
"""

from dataclasses import dataclass
from typing import Any

import gradio as gr

from waves.config import get_config_str_mapping
from waves.events.audio_state import (
    create_audio_component_label,
    create_audio_state_content,
)
from waves.events.visualization import (
    RoutingPlotUpdates,
    create_routing_plot_updates,
    create_spectrogram_plot_update_from_paths,
)
from waves.localization import (
    get_language_index,
    get_localized_text,
)
from waves.routing import RoutingTelemetry
from waves.ui.application import create_application_title_markdown
from waves.ui.language_selector import create_language_flag_html
from waves.ui.progress_modal import (
    ProcessingSummary,
    create_processing_time_button_label,
)


@dataclass(frozen=True, slots=True)
class LanguageChangeUpdates:
    """Named UI updates produced by a language change."""

    flag: Any
    tabs: dict[str, Any]

    title: Any
    status: Any
    audio_input: Any
    examples_title: Any
    examples_placeholder: Any
    run_button: Any
    clear_button: Any
    audio_info_button: Any
    processing_time_button: Any
    audio_info_modal_title: Any
    audio_info_modal_content: Any
    enhanced_audio: Any

    spectrogram_plot: Any
    routing_plots: RoutingPlotUpdates

    settings_title: Any
    settings_description: Any
    settings_placeholder: Any

    about_app_title: Any
    about_app_description: Any
    about_app_placeholder: Any

    requirements_title: Any
    requirements_placeholder: Any


def create_tab_language_updates(
    language_index: int,
) -> dict[str, Any]:
    """Create localized updates for dynamically configured tabs."""

    tab_creators = get_config_str_mapping("TabCreators")

    return {
        tab_name: gr.update(
            label=get_localized_text(
                f"Tabs_{tab_name}",
                language_index,
            ),
        )
        for tab_name in tab_creators
    }


def handle_language_change(
    language: str,
    audio_path: str | None,
    enhanced_audio_path: str | None,
    routing: RoutingTelemetry | None,
    audio_filename: str | None,
    processing_summary: ProcessingSummary | None,
) -> LanguageChangeUpdates:
    """Create named UI updates after language selection."""

    language_index = get_language_index(language)

    audio_state = create_audio_state_content(
        audio_path=audio_path,
        enhanced_audio_path=enhanced_audio_path,
        language_index=language_index,
    )

    spectrogram_update = create_spectrogram_plot_update_from_paths(
        audio_path=audio_path,
        enhanced_audio_path=enhanced_audio_path,
        language_index=language_index,
    )

    routing_updates = create_routing_plot_updates(
        routing=routing,
        language_index=language_index,
    )

    processing_time_button_value = (
        create_processing_time_button_label(
            processing_summary,
            language_index,
        )
        if processing_summary is not None
        else get_localized_text(
            "Labels_PROCESSING_TIME",
            language_index,
        )
    )

    return LanguageChangeUpdates(
        flag=gr.update(
            value=create_language_flag_html(language_index),
        ),
        tabs=create_tab_language_updates(language_index),
        title=gr.update(
            value=create_application_title_markdown(language_index),
        ),
        status=gr.update(
            value=audio_state.status_text,
        ),
        audio_input=gr.update(
            label=create_audio_component_label(
                "Labels_NOISY_AUDIO",
                language_index,
                audio_filename,
            ),
        ),
        examples_title=gr.update(
            value=f"### {get_localized_text('Labels_EXAMPLES', language_index)}",
        ),
        examples_placeholder=gr.update(
            value=get_localized_text(
                "Texts_EXAMPLES_PLACEHOLDER",
                language_index,
            ),
        ),
        run_button=gr.update(
            value=get_localized_text(
                "Labels_RUN",
                language_index,
            ),
        ),
        clear_button=gr.update(
            value=get_localized_text(
                "Labels_CLEAR",
                language_index,
            ),
        ),
        audio_info_button=gr.update(
            value=get_localized_text(
                "Labels_AUDIO_INFO_TITLE",
                language_index,
            ),
        ),
        processing_time_button=gr.update(
            value=processing_time_button_value,
        ),
        audio_info_modal_title=gr.update(
            value=f"### {get_localized_text('Labels_AUDIO_INFO_TITLE', language_index)}",
        ),
        audio_info_modal_content=gr.update(
            value=audio_state.audio_info_html,
        ),
        enhanced_audio=gr.update(
            label=create_audio_component_label(
                "Labels_ENHANCED_AUDIO",
                language_index,
                audio_filename,
            ),
        ),
        spectrogram_plot=spectrogram_update,
        routing_plots=routing_updates,
        settings_title=gr.update(
            value=f"### {get_localized_text('Texts_SETTINGS_TITLE', language_index)}",
        ),
        settings_description=gr.update(
            value=get_localized_text(
                "Texts_SETTINGS_DESCRIPTION",
                language_index,
            ),
        ),
        settings_placeholder=gr.update(
            value=get_localized_text(
                "Texts_SETTINGS_PLACEHOLDER",
                language_index,
            ),
        ),
        about_app_title=gr.update(
            value=f"# {get_localized_text('Texts_ABOUT_TITLE', language_index)}",
        ),
        about_app_description=gr.update(
            value=get_localized_text(
                "Texts_ABOUT_DESCRIPTION",
                language_index,
            ),
        ),
        about_app_placeholder=gr.update(
            value=get_localized_text(
                "Texts_ABOUT_PLACEHOLDER",
                language_index,
            ),
        ),
        requirements_title=gr.update(
            value=f"### {get_localized_text('Texts_REQUIREMENTS_TITLE', language_index)}",
        ),
        requirements_placeholder=gr.update(
            value=get_localized_text(
                "Texts_REQUIREMENTS_PLACEHOLDER",
                language_index,
            ),
        ),
    )
