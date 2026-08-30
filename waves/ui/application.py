"""
File: application.py

Author: Ksenia Zolina, Dmitry Ryumin, Denis Ivanko, and Alexey Karpov

Description: Main application tab UI components for WAVES.

License: MIT License
"""

from dataclasses import dataclass
from html import escape
from pathlib import Path

import gradio as gr
from gradio.helpers import Examples as GradioExamples

from waves.config import (
    PROJECT_ROOT,
    get_config_str,
    get_config_str_list,
)
from waves.localization import get_localized_text

EXAMPLES_PER_PAGE = 4


@dataclass(frozen=True, slots=True)
class ApplicationTabComponents:
    """Components created inside the main WAVES application tab."""

    title: gr.Markdown
    audio_input: gr.Audio
    audio_filename_state: gr.State

    examples: GradioExamples | None
    examples_title: gr.Markdown
    examples_placeholder: gr.Markdown

    run_button: gr.Button
    clear_button: gr.Button

    status: gr.Markdown

    audio_info_button_column: gr.Column
    audio_info_button: gr.Button

    processing_time_button_column: gr.Column
    processing_time_button: gr.Button

    audio_info_modal: gr.Column
    audio_info_modal_title: gr.Markdown
    audio_info_modal_content: gr.HTML
    audio_info_modal_close_button: gr.Button

    processing_modal: gr.Column
    processing_modal_content: gr.HTML
    processing_modal_close_button: gr.Button

    visualization_info_key_state: gr.State
    visualization_info_modal: gr.Column
    visualization_info_modal_content: gr.HTML
    visualization_info_modal_close_button: gr.Button

    enhanced_audio: gr.Audio

    spectrogram_download_button: gr.DownloadButton
    spectrogram_info_button: gr.Button
    spectrogram_plot: gr.Plot

    routing_state: gr.State
    processing_summary_state: gr.State

    expert_occupancy_download_button: gr.DownloadButton
    expert_occupancy_info_button: gr.Button
    expert_occupancy_plot: gr.Plot

    load_over_time_download_button: gr.DownloadButton
    load_over_time_info_button: gr.Button
    load_over_time_plot: gr.Plot

    frequency_routing_download_button: gr.DownloadButton
    frequency_routing_info_button: gr.Button
    frequency_routing_plot: gr.Plot

    layer_routing_download_button: gr.DownloadButton
    layer_routing_info_button: gr.Button
    layer_routing_plot: gr.Plot


def create_application_title_markdown(
    language_index: int,
) -> str:
    """Create the WAVES title markdown with the application version."""

    title_text = get_localized_text(
        "Texts_APP_TITLE",
        language_index,
    ).strip()

    subtitle_text = get_localized_text(
        "Texts_APP_SUBTITLE",
        language_index,
    ).strip()

    subtitle_prefix = f"{title_text}:"

    if subtitle_text.startswith(subtitle_prefix):
        subtitle_text = subtitle_text[len(subtitle_prefix) :].strip()

    app_version = get_config_str(
        "App_VERSION",
        "0.0.0",
    ).strip()

    return (
        f"### {escape(title_text)}: "
        f"{escape(subtitle_text)} "
        '<span class="application-version">'
        f"v{escape(app_version)}"
        "</span>"
    )


def get_example_audio_labels(
    example_audio_paths: list[list[str]],
) -> list[str]:
    """Return display labels for example audio files."""

    labels: list[str] = []

    for example in example_audio_paths:
        stem = Path(example[0]).stem

        parts = stem.split(
            "_",
            maxsplit=1,
        )

        if len(parts) == 2:
            index, name = parts

            labels.append(f"{index} · {name.replace('_', ' ').title()}")
        else:
            labels.append(
                stem.replace(
                    "_",
                    " ",
                ).title()
            )

    return labels


def get_existing_example_audio_paths() -> list[list[str]]:
    """Return existing example audio files configured for WAVES."""

    examples_dir = get_config_str(
        "StaticPaths_EXAMPLES",
        "examples",
    )

    example_files = get_config_str_list(
        "Examples_AUDIO",
        [],
    )

    existing_examples: list[list[str]] = []

    for example_file in example_files:
        path = PROJECT_ROOT / examples_dir / example_file

        if path.is_file():
            existing_examples.append(
                [
                    str(path),
                ]
            )

    return existing_examples


def create_application_tab(
    language_index: int = 0,
) -> ApplicationTabComponents:
    """Create the main WAVES application tab."""

    title = gr.Markdown(
        value=create_application_title_markdown(language_index),
        elem_classes="application-compact-title",
    )

    audio_filename_state = gr.State(
        value=None,
    )

    routing_state = gr.State(
        value=None,
    )

    processing_summary_state = gr.State(
        value=None,
    )

    visualization_info_key_state = gr.State(
        value=None,
    )

    with gr.Row(
        elem_classes="application-input-row",
    ):
        with gr.Column(
            scale=5,
            min_width=420,
            elem_classes="application-audio-column",
        ):
            audio_input = gr.Audio(
                label=get_localized_text(
                    "Labels_NOISY_AUDIO",
                    language_index,
                ),
                type="filepath",
                sources=[
                    "upload",
                    "microphone",
                ],
                format="wav",
                interactive=True,
                buttons=[
                    "download",
                ],
                elem_id="application-audio-input",
                elem_classes="application-audio-input",
            )

        with gr.Column(
            scale=1,
            min_width=300,
            elem_classes="application-examples-column",
        ):
            example_audio_paths = get_existing_example_audio_paths()

            examples_label = get_localized_text(
                "Labels_EXAMPLES",
                language_index,
            )

            examples_title = gr.Markdown(
                value=f"### {examples_label}",
                elem_classes="application-examples-title",
            )

            examples: GradioExamples | None = None

            if example_audio_paths:
                examples = gr.Examples(
                    examples=example_audio_paths,
                    inputs=[
                        audio_input,
                    ],
                    cache_examples=False,
                    examples_per_page=EXAMPLES_PER_PAGE,
                    label="",
                    example_labels=get_example_audio_labels(example_audio_paths),
                    elem_id="application-examples",
                )

                examples_placeholder = gr.Markdown(
                    value="",
                    visible=False,
                    elem_classes=("application-examples-placeholder"),
                )
            else:
                examples_placeholder = gr.Markdown(
                    value=get_localized_text(
                        "Texts_EXAMPLES_PLACEHOLDER",
                        language_index,
                    ),
                    visible=True,
                    elem_classes=("application-examples-placeholder"),
                )

    with gr.Row(
        elem_classes="application-action-row",
    ):
        run_button = gr.Button(
            value=get_localized_text(
                "Labels_RUN",
                language_index,
            ),
            variant="primary",
            size="lg",
            interactive=False,
            elem_id="run-button",
            elem_classes="application-run-button",
        )

        clear_button = gr.Button(
            value=get_localized_text(
                "Labels_CLEAR",
                language_index,
            ),
            variant="secondary",
            size="lg",
            interactive=False,
            elem_id="clear-button",
            elem_classes="application-clear-button",
        )

    with gr.Row(
        elem_classes="application-status-row",
    ):
        with (
            gr.Column(
                scale=7,
                min_width=560,
                elem_classes=("application-status-panel-column"),
            ),
            gr.Row(
                elem_classes="application-status-panel",
            ),
        ):
            with gr.Column(
                scale=1,
                min_width=0,
                elem_classes="application-status-column",
            ):
                status = gr.Markdown(
                    value=get_localized_text(
                        "Texts_STATUS_READY",
                        language_index,
                    ),
                    elem_classes="application-status",
                )

            with gr.Column(
                scale=0,
                min_width=360,
                visible=False,
                elem_id="processing-time-button-column",
                elem_classes=("application-processing-time-button-column"),
            ) as processing_time_button_column:
                processing_time_button = gr.Button(
                    value=get_localized_text(
                        "Labels_PROCESSING_TIME",
                        language_index,
                    ),
                    variant="secondary",
                    size="lg",
                    interactive=False,
                    visible=True,
                    elem_id="processing-time-button",
                    elem_classes=("application-processing-time-button"),
                )

        with gr.Column(
            scale=3,
            min_width=300,
            visible=False,
            elem_id="audio-info-button-column",
            elem_classes=("application-audio-info-button-column"),
        ) as audio_info_button_column:
            audio_info_button = gr.Button(
                value=get_localized_text(
                    "Labels_AUDIO_INFO_TITLE",
                    language_index,
                ),
                variant="secondary",
                size="lg",
                interactive=False,
                visible=True,
                elem_id="audio-info-button",
                elem_classes="application-audio-info-button",
            )

    with (
        gr.Column(
            visible=False,
            elem_id="audio-info-modal",
            elem_classes="audio-info-modal-backdrop",
        ) as audio_info_modal,
        gr.Column(
            elem_classes="audio-info-modal-card",
        ),
    ):
        with gr.Row(
            elem_classes="audio-info-modal-header",
        ):
            with gr.Column(
                scale=1,
                min_width=0,
                elem_classes="audio-info-modal-title-column",
            ):
                audio_info_title = get_localized_text(
                    "Labels_AUDIO_INFO_TITLE",
                    language_index,
                )

                audio_info_modal_title = gr.Markdown(
                    value=f"### {audio_info_title}",
                    elem_classes="audio-info-modal-title",
                )

            with gr.Column(
                scale=0,
                min_width=42,
                elem_classes="audio-info-modal-close-column",
            ):
                audio_info_modal_close_button = gr.Button(
                    value="Close",
                    variant="secondary",
                    size="sm",
                    elem_id=("audio-info-modal-close-button"),
                    elem_classes=("audio-info-modal-close-button"),
                )

        audio_info_modal_content = gr.HTML(
            value="",
            elem_classes="audio-info-modal-content",
        )

    with (
        gr.Column(
            visible=False,
            elem_id="processing-modal",
            elem_classes="processing-modal-backdrop",
        ) as processing_modal,
        gr.Column(
            elem_classes="processing-modal-card",
        ),
    ):
        processing_modal_close_button = gr.Button(
            value="Close",
            variant="secondary",
            size="sm",
            visible=False,
            interactive=True,
            elem_id="processing-modal-close-button",
            elem_classes="processing-modal-close-button",
        )

        processing_modal_content = gr.HTML(
            value="",
            elem_classes="processing-modal-content",
        )

    with (
        gr.Column(
            visible=False,
            elem_id="visualization-info-modal",
            elem_classes=("visualization-info-modal-backdrop"),
        ) as visualization_info_modal,
        gr.Column(
            elem_classes="visualization-info-modal-card",
        ),
    ):
        visualization_info_modal_close_button = gr.Button(
            value="Close",
            variant="secondary",
            size="sm",
            visible=True,
            interactive=True,
            elem_id=("visualization-info-modal-close-button"),
            elem_classes=("visualization-info-modal-close-button"),
        )

        visualization_info_modal_content = gr.HTML(
            value="",
            elem_classes="visualization-info-modal-content",
        )

    enhanced_audio = gr.Audio(
        label=get_localized_text(
            "Labels_ENHANCED_AUDIO",
            language_index,
        ),
        type="filepath",
        interactive=False,
        visible=False,
        buttons=[
            "download",
        ],
        elem_id="enhanced-audio-output",
        elem_classes="application-audio-output",
    )

    with gr.Column(
        elem_classes=[
            "application-visualization-card",
            "application-spectrogram-card",
        ],
    ):
        spectrogram_download_button = gr.DownloadButton(
            label="PDF",
            value=None,
            variant="secondary",
            size="sm",
            interactive=True,
            visible=False,
            elem_id="spectrogram-download-button",
            elem_classes=("application-visualization-download-button"),
        )

        spectrogram_info_button = gr.Button(
            value="ⓘ",
            variant="secondary",
            size="sm",
            interactive=True,
            visible=False,
            elem_id="spectrogram-info-button",
            elem_classes=("application-visualization-info-button"),
        )

        spectrogram_plot = gr.Plot(
            value=None,
            visible=False,
            show_label=False,
            container=False,
            elem_id="spectrogram-comparison-plot",
            elem_classes="application-spectrogram-plot",
        )

    with gr.Row(
        elem_classes="application-routing-row",
    ):
        with gr.Column(
            scale=1,
            min_width=420,
            elem_classes=[
                "application-routing-column",
                "application-visualization-card",
            ],
        ):
            expert_occupancy_download_button = gr.DownloadButton(
                label="PDF",
                value=None,
                variant="secondary",
                size="sm",
                interactive=True,
                visible=False,
                elem_id="expert-occupancy-download-button",
                elem_classes=("application-visualization-download-button"),
            )

            expert_occupancy_info_button = gr.Button(
                value="ⓘ",
                variant="secondary",
                size="sm",
                interactive=True,
                visible=False,
                elem_id="expert-occupancy-info-button",
                elem_classes=("application-visualization-info-button"),
            )

            expert_occupancy_plot = gr.Plot(
                value=None,
                visible=False,
                show_label=False,
                container=False,
                elem_id="expert-occupancy-plot",
                elem_classes=[
                    "application-routing-plot",
                    "application-expert-occupancy-plot",
                ],
            )

        with gr.Column(
            scale=1,
            min_width=420,
            elem_classes=[
                "application-routing-column",
                "application-visualization-card",
            ],
        ):
            load_over_time_download_button = gr.DownloadButton(
                label="PDF",
                value=None,
                variant="secondary",
                size="sm",
                interactive=True,
                visible=False,
                elem_id="load-over-time-download-button",
                elem_classes=("application-visualization-download-button"),
            )

            load_over_time_info_button = gr.Button(
                value="ⓘ",
                variant="secondary",
                size="sm",
                interactive=True,
                visible=False,
                elem_id="load-over-time-info-button",
                elem_classes=("application-visualization-info-button"),
            )

            load_over_time_plot = gr.Plot(
                value=None,
                visible=False,
                show_label=False,
                container=False,
                elem_id="load-over-time-plot",
                elem_classes=[
                    "application-routing-plot",
                    "application-load-over-time-plot",
                ],
            )

    with gr.Row(
        elem_classes="application-routing-row",
    ):
        with gr.Column(
            scale=1,
            min_width=420,
            elem_classes=[
                "application-routing-column",
                "application-visualization-card",
            ],
        ):
            frequency_routing_download_button = gr.DownloadButton(
                label="PDF",
                value=None,
                variant="secondary",
                size="sm",
                interactive=True,
                visible=False,
                elem_id="frequency-routing-download-button",
                elem_classes=("application-visualization-download-button"),
            )

            frequency_routing_info_button = gr.Button(
                value="ⓘ",
                variant="secondary",
                size="sm",
                interactive=True,
                visible=False,
                elem_id="frequency-routing-info-button",
                elem_classes=("application-visualization-info-button"),
            )

            frequency_routing_plot = gr.Plot(
                value=None,
                visible=False,
                show_label=False,
                container=False,
                elem_id="frequency-routing-plot",
                elem_classes=[
                    "application-routing-plot",
                    "application-frequency-routing-plot",
                ],
            )

        with gr.Column(
            scale=1,
            min_width=420,
            elem_classes=[
                "application-routing-column",
                "application-visualization-card",
            ],
        ):
            layer_routing_download_button = gr.DownloadButton(
                label="PDF",
                value=None,
                variant="secondary",
                size="sm",
                interactive=True,
                visible=False,
                elem_id="layer-routing-download-button",
                elem_classes=("application-visualization-download-button"),
            )

            layer_routing_info_button = gr.Button(
                value="ⓘ",
                variant="secondary",
                size="sm",
                interactive=True,
                visible=False,
                elem_id="layer-routing-info-button",
                elem_classes=("application-visualization-info-button"),
            )

            layer_routing_plot = gr.Plot(
                value=None,
                visible=False,
                show_label=False,
                container=False,
                elem_id="layer-routing-plot",
                elem_classes=[
                    "application-routing-plot",
                    "application-layer-routing-plot",
                ],
            )

    return ApplicationTabComponents(
        title=title,
        audio_input=audio_input,
        audio_filename_state=audio_filename_state,
        examples=examples,
        examples_title=examples_title,
        examples_placeholder=examples_placeholder,
        run_button=run_button,
        clear_button=clear_button,
        status=status,
        audio_info_button_column=audio_info_button_column,
        audio_info_button=audio_info_button,
        processing_time_button_column=(processing_time_button_column),
        processing_time_button=processing_time_button,
        audio_info_modal=audio_info_modal,
        audio_info_modal_title=audio_info_modal_title,
        audio_info_modal_content=audio_info_modal_content,
        audio_info_modal_close_button=(audio_info_modal_close_button),
        processing_modal=processing_modal,
        processing_modal_content=processing_modal_content,
        processing_modal_close_button=(processing_modal_close_button),
        visualization_info_key_state=(visualization_info_key_state),
        visualization_info_modal=visualization_info_modal,
        visualization_info_modal_content=(visualization_info_modal_content),
        visualization_info_modal_close_button=(visualization_info_modal_close_button),
        enhanced_audio=enhanced_audio,
        spectrogram_download_button=(spectrogram_download_button),
        spectrogram_info_button=spectrogram_info_button,
        spectrogram_plot=spectrogram_plot,
        routing_state=routing_state,
        processing_summary_state=processing_summary_state,
        expert_occupancy_download_button=(expert_occupancy_download_button),
        expert_occupancy_info_button=(expert_occupancy_info_button),
        expert_occupancy_plot=expert_occupancy_plot,
        load_over_time_download_button=(load_over_time_download_button),
        load_over_time_info_button=(load_over_time_info_button),
        load_over_time_plot=load_over_time_plot,
        frequency_routing_download_button=(frequency_routing_download_button),
        frequency_routing_info_button=(frequency_routing_info_button),
        frequency_routing_plot=frequency_routing_plot,
        layer_routing_download_button=(layer_routing_download_button),
        layer_routing_info_button=(layer_routing_info_button),
        layer_routing_plot=layer_routing_plot,
    )
