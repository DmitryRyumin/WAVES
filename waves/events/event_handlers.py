"""
File: event_handlers.py

Author: Ksenia Zolina, Dmitry Ryumin, Denis Ivanko, and Alexey Karpov

Description: Event handler registration for the WAVES Gradio application.

License: MIT License
"""

from collections.abc import Iterator
from functools import partial
from typing import Any, cast

import gradio as gr
from gradio.components.plot import PlotData

from waves.events.application import (
    RECORDED_AUDIO_FILENAME,
    handle_audio_change,
    handle_clear_application,
    handle_enhancement_started,
    handle_hide_audio_info,
    handle_hide_processing_modal,
    handle_run_enhancement,
    handle_show_audio_info,
    handle_show_processing_timing,
)
from waves.events.application_updates import (
    AudioChangeUpdates,
    ClearApplicationUpdates,
    EnhancementUpdates,
    ProcessingModalUpdates,
    VisualizationInfoButtonUpdates,
)
from waves.events.audio_state import (
    get_audio_display_filename,
)
from waves.events.language import handle_language_change
from waves.events.visualization import RoutingPlotUpdates
from waves.events.visualization_info import (
    VisualizationInfoModalUpdates,
    handle_hide_visualization_info,
    handle_refresh_visualization_info,
    handle_show_visualization_info,
)
from waves.logger import get_logger
from waves.routing import RoutingTelemetry
from waves.ui.application import ApplicationTabComponents
from waves.ui.client_scripts import (
    CLEAR_EXAMPLE_SELECTION_JS,
    EXAMPLES_UI_JS,
    MODAL_CLOSE_ANIMATION_JS,
)
from waves.ui.language_selector import (
    LanguageSelectorComponents,
)
from waves.ui.progress_modal import ProcessingSummary
from waves.ui.settings import SettingsTabComponents
from waves.ui.tabs import (
    AboutAppTabComponents,
    AppTabsComponents,
    RequirementsTabComponents,
)
from waves.ui.visualization_info import VisualizationInfoKey
from waves.visualization.export import (
    VisualizationExportKey,
    VisualizationPdfExports,
    create_visualization_pdf_exports_from_plot_json,
    remove_visualization_pdf_exports,
)

LOGGER = get_logger(__name__)


def setup_app_event_handlers(
    gradio_app: gr.Blocks,
    language_selector: LanguageSelectorComponents,
    app_tabs: AppTabsComponents,
) -> None:
    """Register WAVES application event handlers."""

    app_content = cast(
        ApplicationTabComponents,
        app_tabs.tab_contents["APP"],
    )

    settings_content = cast(
        SettingsTabComponents,
        app_tabs.tab_contents["SETTINGS"],
    )

    about_app_content = cast(
        AboutAppTabComponents,
        app_tabs.tab_contents["ABOUT_APP"],
    )

    requirements_content = cast(
        RequirementsTabComponents,
        app_tabs.tab_contents["REQUIREMENTS"],
    )

    routing_output_components = [
        app_content.expert_occupancy_plot,
        app_content.layer_routing_plot,
        app_content.frequency_routing_plot,
        app_content.load_over_time_plot,
    ]

    visualization_info_button_components = [
        app_content.spectrogram_info_button,
        app_content.expert_occupancy_info_button,
        app_content.layer_routing_info_button,
        app_content.frequency_routing_info_button,
        app_content.load_over_time_info_button,
    ]

    visualization_download_button_components = [
        app_content.spectrogram_download_button,
        app_content.expert_occupancy_download_button,
        app_content.layer_routing_download_button,
        app_content.frequency_routing_download_button,
        app_content.load_over_time_download_button,
    ]

    visualization_info_modal_components = [
        app_content.visualization_info_key_state,
        app_content.visualization_info_modal,
        app_content.visualization_info_modal_content,
        app_content.visualization_info_modal_close_button,
    ]

    def map_visualization_info_button_updates(
        updates: VisualizationInfoButtonUpdates,
    ) -> dict[Any, Any]:
        """Map visualization information button updates."""

        return {
            app_content.spectrogram_info_button: (updates.spectrogram),
            app_content.expert_occupancy_info_button: (updates.expert_occupancy),
            app_content.layer_routing_info_button: (updates.layer_routing),
            app_content.frequency_routing_info_button: (updates.frequency_routing),
            app_content.load_over_time_info_button: (updates.load_over_time),
        }

    def map_routing_updates(
        updates: RoutingPlotUpdates,
    ) -> dict[Any, Any]:
        """Map named routing updates to their Gradio components."""

        return {
            app_content.expert_occupancy_plot: (updates.expert_occupancy_plot),
            app_content.layer_routing_plot: (updates.layer_routing_plot),
            app_content.frequency_routing_plot: (updates.frequency_routing_plot),
            app_content.load_over_time_plot: (updates.load_over_time_plot),
        }

    def handle_language_change_event(
        language: str,
        audio_path: str | None,
        enhanced_audio_path: str | None,
        routing: RoutingTelemetry | None,
        audio_filename: str | None,
        processing_summary: ProcessingSummary | None,
    ) -> dict[Any, Any]:
        """Map named language updates to their Gradio components."""

        updates = handle_language_change(
            language=language,
            audio_path=audio_path,
            enhanced_audio_path=enhanced_audio_path,
            routing=routing,
            audio_filename=audio_filename,
            processing_summary=processing_summary,
        )

        component_updates: dict[
            Any,
            Any,
        ] = {
            language_selector.flag: (updates.flag),
            app_content.title: (updates.title),
            app_content.status: (updates.status),
            app_content.audio_input: (updates.audio_input),
            app_content.examples_title: (updates.examples_title),
            app_content.examples_placeholder: (updates.examples_placeholder),
            app_content.run_button: (updates.run_button),
            app_content.clear_button: (updates.clear_button),
            app_content.audio_info_button: (updates.audio_info_button),
            app_content.processing_time_button: (updates.processing_time_button),
            app_content.audio_info_modal_title: (updates.audio_info_modal_title),
            app_content.audio_info_modal_content: (updates.audio_info_modal_content),
            app_content.enhanced_audio: (updates.enhanced_audio),
            app_content.spectrogram_plot: (updates.spectrogram_plot),
            settings_content.title: (updates.settings_title),
            settings_content.description: (updates.settings_description),
            settings_content.placeholder: (updates.settings_placeholder),
            about_app_content.title: (updates.about_app_title),
            about_app_content.description: (updates.about_app_description),
            about_app_content.placeholder: (updates.about_app_placeholder),
            requirements_content.title: (updates.requirements_title),
            requirements_content.placeholder: (updates.requirements_placeholder),
        }

        component_updates.update(map_routing_updates(updates.routing_plots))

        for (
            tab_name,
            tab_update,
        ) in updates.tabs.items():
            tab_component = app_tabs.tab_components.get(tab_name)

            if tab_component is not None:
                component_updates[tab_component] = tab_update

        return component_updates

    def map_audio_change_updates(
        updates: AudioChangeUpdates,
    ) -> dict[Any, Any]:
        """Map named input-audio updates to their Gradio components."""

        component_updates: dict[
            Any,
            Any,
        ] = {
            app_content.audio_input: (updates.audio_input),
            app_content.audio_filename_state: (updates.audio_filename_state),
            app_content.status: (updates.status),
            app_content.run_button: (updates.run_button),
            app_content.clear_button: (updates.clear_button),
            app_content.audio_info_button_column: (updates.audio_info_button_column),
            app_content.audio_info_button: (updates.audio_info_button),
            app_content.audio_info_modal: (updates.audio_info_modal),
            app_content.audio_info_modal_content: (updates.audio_info_modal_content),
            app_content.processing_time_button_column: (updates.processing_time_button_column),
            app_content.processing_time_button: (updates.processing_time_button),
            app_content.processing_modal: (updates.processing_modal),
            app_content.processing_modal_content: (updates.processing_modal_content),
            app_content.processing_modal_close_button: (updates.processing_modal_close_button),
            app_content.processing_summary_state: (updates.processing_summary_state),
            app_content.visualization_info_key_state: (updates.visualization_info_key_state),
            app_content.visualization_info_modal: (updates.visualization_info_modal),
            app_content.visualization_info_modal_content: (updates.visualization_info_modal_content),
            app_content.visualization_info_modal_close_button: (updates.visualization_info_modal_close_button),
            app_content.enhanced_audio: (updates.enhanced_audio),
            app_content.spectrogram_plot: (updates.spectrogram_plot),
            app_content.routing_state: (updates.routing_state),
        }

        component_updates.update(map_routing_updates(updates.routing_plots))

        component_updates.update(map_visualization_info_button_updates(updates.visualization_info_buttons))

        return component_updates

    def handle_file_audio_change_event(
        audio_path: str | None,
        enhanced_audio_path: str | None,
        language: str,
    ) -> dict[Any, Any]:
        """Handle an audio sample loaded from a file."""

        updates = handle_audio_change(
            audio_path=audio_path,
            enhanced_audio_path=enhanced_audio_path,
            language=language,
            audio_filename=(get_audio_display_filename(audio_path)),
        )

        return map_audio_change_updates(updates)

    def handle_recorded_audio_change_event(
        audio_path: str | None,
        enhanced_audio_path: str | None,
        language: str,
    ) -> dict[Any, Any]:
        """Handle an audio sample recorded in the browser."""

        updates = handle_audio_change(
            audio_path=audio_path,
            enhanced_audio_path=enhanced_audio_path,
            language=language,
            audio_filename=(RECORDED_AUDIO_FILENAME),
        )

        return map_audio_change_updates(updates)

    def map_enhancement_updates(
        updates: EnhancementUpdates,
    ) -> dict[Any, Any]:
        """Map enhancement updates to Gradio components."""

        component_updates: dict[
            Any,
            Any,
        ] = {
            app_content.audio_input: (updates.audio_input),
            app_content.status: (updates.status),
            app_content.run_button: (updates.run_button),
            app_content.clear_button: (updates.clear_button),
            app_content.audio_info_button: (updates.audio_info_button),
            app_content.audio_info_modal: (updates.audio_info_modal),
            app_content.processing_time_button_column: (updates.processing_time_button_column),
            app_content.processing_time_button: (updates.processing_time_button),
            app_content.processing_modal: (updates.processing_modal),
            app_content.processing_modal_content: (updates.processing_modal_content),
            app_content.processing_modal_close_button: (updates.processing_modal_close_button),
            app_content.processing_summary_state: (updates.processing_summary_state),
            app_content.visualization_info_key_state: (updates.visualization_info_key_state),
            app_content.visualization_info_modal: (updates.visualization_info_modal),
            app_content.visualization_info_modal_content: (updates.visualization_info_modal_content),
            app_content.visualization_info_modal_close_button: (updates.visualization_info_modal_close_button),
            app_content.enhanced_audio: (updates.enhanced_audio),
            app_content.spectrogram_plot: (updates.spectrogram_plot),
            app_content.routing_state: (updates.routing_state),
        }

        component_updates.update(map_routing_updates(updates.routing_plots))

        component_updates.update(map_visualization_info_button_updates(updates.visualization_info_buttons))

        return component_updates

    def handle_enhancement_started_event(
        audio_path: str | None,
        enhanced_audio_path: str | None,
        language: str,
        audio_filename: str | None,
    ) -> dict[Any, Any]:
        """Immediately lock the interface and show the processing modal."""

        updates = handle_enhancement_started(
            audio_path=audio_path,
            enhanced_audio_path=(enhanced_audio_path),
            language=language,
            audio_filename=audio_filename,
        )

        return map_enhancement_updates(updates)

    def handle_run_enhancement_event(
        audio_path: str | None,
        language: str,
        audio_filename: str | None,
    ) -> Iterator[
        dict[
            Any,
            Any,
        ]
    ]:
        """Map streamed enhancement updates to their Gradio components."""

        for updates in handle_run_enhancement(
            audio_path=audio_path,
            language=language,
            audio_filename=audio_filename,
        ):
            yield map_enhancement_updates(updates)

    def handle_clear_application_event(
        language: str,
        enhanced_audio_path: str | None,
    ) -> dict[Any, Any]:
        """Map named clear-state updates to their Gradio components."""

        updates: ClearApplicationUpdates = handle_clear_application(
            language=language,
            enhanced_audio_path=(enhanced_audio_path),
        )

        component_updates: dict[
            Any,
            Any,
        ] = {
            app_content.audio_input: (updates.audio_input),
            app_content.audio_filename_state: (updates.audio_filename_state),
            app_content.status: (updates.status),
            app_content.run_button: (updates.run_button),
            app_content.clear_button: (updates.clear_button),
            app_content.audio_info_button_column: (updates.audio_info_button_column),
            app_content.audio_info_button: (updates.audio_info_button),
            app_content.audio_info_modal: (updates.audio_info_modal),
            app_content.audio_info_modal_content: (updates.audio_info_modal_content),
            app_content.processing_time_button_column: (updates.processing_time_button_column),
            app_content.processing_time_button: (updates.processing_time_button),
            app_content.processing_modal: (updates.processing_modal),
            app_content.processing_modal_content: (updates.processing_modal_content),
            app_content.processing_modal_close_button: (updates.processing_modal_close_button),
            app_content.processing_summary_state: (updates.processing_summary_state),
            app_content.visualization_info_key_state: (updates.visualization_info_key_state),
            app_content.visualization_info_modal: (updates.visualization_info_modal),
            app_content.visualization_info_modal_content: (updates.visualization_info_modal_content),
            app_content.visualization_info_modal_close_button: (updates.visualization_info_modal_close_button),
            app_content.enhanced_audio: (updates.enhanced_audio),
            app_content.spectrogram_plot: (updates.spectrogram_plot),
            app_content.routing_state: (updates.routing_state),
        }

        component_updates.update(map_routing_updates(updates.routing_plots))

        component_updates.update(map_visualization_info_button_updates(updates.visualization_info_buttons))

        return component_updates

    def handle_show_processing_timing_event(
        summary: ProcessingSummary | None,
        language: str,
    ) -> dict[Any, Any]:
        """Open the completed processing timing modal."""

        updates: ProcessingModalUpdates = handle_show_processing_timing(
            summary=summary,
            language=language,
        )

        return {
            app_content.processing_modal: (updates.modal),
            app_content.processing_modal_content: (updates.content),
            app_content.processing_modal_close_button: (updates.close_button),
        }

    def map_visualization_info_modal_updates(
        updates: VisualizationInfoModalUpdates,
    ) -> dict[Any, Any]:
        """Map shared visualization information modal updates."""

        return {
            app_content.visualization_info_key_state: (updates.key_state),
            app_content.visualization_info_modal: (updates.modal),
            app_content.visualization_info_modal_content: (updates.content),
            app_content.visualization_info_modal_close_button: (updates.close_button),
        }

    def handle_show_visualization_info_event(
        visualization_key: VisualizationInfoKey,
        language: str,
    ) -> dict[Any, Any]:
        """Open information for one scientific visualization."""

        updates = handle_show_visualization_info(
            visualization_key=(visualization_key),
            language=language,
        )

        return map_visualization_info_modal_updates(updates)

    def handle_hide_visualization_info_event() -> dict[
        Any,
        Any,
    ]:
        """Close the shared visualization information modal."""

        updates = handle_hide_visualization_info()

        return map_visualization_info_modal_updates(updates)

    def map_visualization_pdf_exports(
        exports: VisualizationPdfExports,
    ) -> dict[Any, Any]:
        """Map generated PDF paths to visualization download buttons."""

        return {
            app_content.spectrogram_download_button: (
                gr.update(
                    value=exports.spectrogram,
                    visible=(exports.spectrogram is not None),
                )
            ),
            app_content.expert_occupancy_download_button: (
                gr.update(
                    value=(exports.expert_occupancy),
                    visible=(exports.expert_occupancy is not None),
                )
            ),
            app_content.layer_routing_download_button: (
                gr.update(
                    value=(exports.layer_routing),
                    visible=(exports.layer_routing is not None),
                )
            ),
            app_content.frequency_routing_download_button: (
                gr.update(
                    value=(exports.frequency_routing),
                    visible=(exports.frequency_routing is not None),
                )
            ),
            app_content.load_over_time_download_button: (
                gr.update(
                    value=(exports.load_over_time),
                    visible=(exports.load_over_time is not None),
                )
            ),
        }

    def hide_visualization_pdf_exports() -> dict[
        Any,
        Any,
    ]:
        """Hide all visualization PDF download buttons."""

        return map_visualization_pdf_exports(VisualizationPdfExports())

    def handle_clear_visualization_pdf_exports(
        spectrogram_path: str | None,
        expert_occupancy_path: str | None,
        layer_routing_path: str | None,
        frequency_routing_path: str | None,
        load_over_time_path: str | None,
    ) -> dict[Any, Any]:
        """Delete generated PDF exports and hide their download buttons."""

        remove_visualization_pdf_exports(
            (
                spectrogram_path,
                expert_occupancy_path,
                layer_routing_path,
                frequency_routing_path,
                load_over_time_path,
            )
        )

        return hide_visualization_pdf_exports()

    def handle_refresh_visualization_pdf_exports(
        spectrogram_plot: PlotData | None,
        expert_occupancy_plot: PlotData | None,
        layer_routing_plot: PlotData | None,
        frequency_routing_plot: PlotData | None,
        load_over_time_plot: PlotData | None,
        spectrogram_path: str | None,
        expert_occupancy_path: str | None,
        layer_routing_path: str | None,
        frequency_routing_path: str | None,
        load_over_time_path: str | None,
    ) -> dict[Any, Any]:
        """Regenerate white-theme PDF exports for visible Plotly figures."""

        remove_visualization_pdf_exports(
            (
                spectrogram_path,
                expert_occupancy_path,
                layer_routing_path,
                frequency_routing_path,
                load_over_time_path,
            )
        )

        plot_data = {
            VisualizationExportKey.SPECTROGRAM: (
                spectrogram_plot.plot if (spectrogram_plot is not None and spectrogram_plot.type == "plotly") else None
            ),
            VisualizationExportKey.EXPERT_OCCUPANCY: (
                expert_occupancy_plot.plot
                if (expert_occupancy_plot is not None and expert_occupancy_plot.type == "plotly")
                else None
            ),
            VisualizationExportKey.LAYER_ROUTING: (
                layer_routing_plot.plot
                if (layer_routing_plot is not None and layer_routing_plot.type == "plotly")
                else None
            ),
            VisualizationExportKey.FREQUENCY_ROUTING: (
                frequency_routing_plot.plot
                if (frequency_routing_plot is not None and frequency_routing_plot.type == "plotly")
                else None
            ),
            VisualizationExportKey.LOAD_OVER_TIME: (
                load_over_time_plot.plot
                if (load_over_time_plot is not None and load_over_time_plot.type == "plotly")
                else None
            ),
        }

        try:
            exports = create_visualization_pdf_exports_from_plot_json(plot_data)
        except Exception:
            LOGGER.exception("Failed to export WAVES visualizations to PDF.")

            return hide_visualization_pdf_exports()

        return map_visualization_pdf_exports(exports)

    language_outputs = [
        language_selector.flag,
        *app_tabs.tab_components.values(),
        app_content.title,
        app_content.status,
        app_content.audio_input,
        app_content.examples_title,
        app_content.examples_placeholder,
        app_content.run_button,
        app_content.clear_button,
        app_content.audio_info_button,
        app_content.processing_time_button,
        app_content.audio_info_modal_title,
        app_content.audio_info_modal_content,
        app_content.enhanced_audio,
        app_content.spectrogram_plot,
        *routing_output_components,
        settings_content.title,
        settings_content.description,
        settings_content.placeholder,
        about_app_content.title,
        about_app_content.description,
        about_app_content.placeholder,
        requirements_content.title,
        requirements_content.placeholder,
    ]

    audio_change_outputs = [
        app_content.audio_input,
        app_content.audio_filename_state,
        app_content.status,
        app_content.run_button,
        app_content.clear_button,
        app_content.audio_info_button_column,
        app_content.audio_info_button,
        app_content.audio_info_modal,
        app_content.audio_info_modal_content,
        app_content.processing_time_button_column,
        app_content.processing_time_button,
        app_content.processing_modal,
        app_content.processing_modal_content,
        app_content.processing_modal_close_button,
        app_content.processing_summary_state,
        *visualization_info_modal_components,
        *visualization_info_button_components,
        app_content.enhanced_audio,
        app_content.spectrogram_plot,
        app_content.routing_state,
        *routing_output_components,
    ]

    enhancement_outputs = [
        app_content.audio_input,
        app_content.status,
        app_content.run_button,
        app_content.clear_button,
        app_content.audio_info_button,
        app_content.audio_info_modal,
        app_content.processing_time_button_column,
        app_content.processing_time_button,
        app_content.processing_modal,
        app_content.processing_modal_content,
        app_content.processing_modal_close_button,
        app_content.processing_summary_state,
        *visualization_info_modal_components,
        *visualization_info_button_components,
        app_content.enhanced_audio,
        app_content.spectrogram_plot,
        app_content.routing_state,
        *routing_output_components,
    ]

    clear_outputs = audio_change_outputs

    visualization_pdf_export_inputs = [
        app_content.spectrogram_plot,
        app_content.expert_occupancy_plot,
        app_content.layer_routing_plot,
        app_content.frequency_routing_plot,
        app_content.load_over_time_plot,
        *visualization_download_button_components,
    ]

    visualization_pdf_cleanup_inputs = [
        *visualization_download_button_components,
    ]

    language_dropdown = cast(
        Any,
        language_selector.dropdown,
    )

    audio_input = cast(
        Any,
        app_content.audio_input,
    )

    run_button = cast(
        Any,
        app_content.run_button,
    )

    clear_button = cast(
        Any,
        app_content.clear_button,
    )

    audio_info_button = cast(
        Any,
        app_content.audio_info_button,
    )

    audio_info_modal_close_button = cast(
        Any,
        app_content.audio_info_modal_close_button,
    )

    processing_time_button = cast(
        Any,
        app_content.processing_time_button,
    )

    processing_modal_close_button = cast(
        Any,
        app_content.processing_modal_close_button,
    )

    visualization_info_modal_close_button = cast(
        Any,
        app_content.visualization_info_modal_close_button,
    )

    spectrogram_info_button = cast(
        Any,
        app_content.spectrogram_info_button,
    )

    expert_occupancy_info_button = cast(
        Any,
        app_content.expert_occupancy_info_button,
    )

    layer_routing_info_button = cast(
        Any,
        app_content.layer_routing_info_button,
    )

    frequency_routing_info_button = cast(
        Any,
        app_content.frequency_routing_info_button,
    )

    load_over_time_info_button = cast(
        Any,
        app_content.load_over_time_info_button,
    )

    gradio_app.load(
        fn=None,
        inputs=[
            language_selector.dropdown,
        ],
        outputs=[],
        js=EXAMPLES_UI_JS,
        queue=False,
        show_progress="hidden",
    )

    language_change_event = language_dropdown.change(
        fn=handle_language_change_event,
        inputs=[
            language_selector.dropdown,
            app_content.audio_input,
            app_content.enhanced_audio,
            app_content.routing_state,
            app_content.audio_filename_state,
            app_content.processing_summary_state,
        ],
        outputs=language_outputs,
        queue=False,
        show_progress="hidden",
    )

    language_change_event.then(
        fn=handle_refresh_visualization_pdf_exports,
        inputs=visualization_pdf_export_inputs,
        outputs=(visualization_download_button_components),
        queue=True,
        concurrency_limit=1,
        concurrency_id=("visualization-pdf-export"),
        show_progress="hidden",
    )

    language_dropdown.change(
        fn=handle_refresh_visualization_info,
        inputs=[
            app_content.visualization_info_key_state,
            language_selector.dropdown,
        ],
        outputs=[
            app_content.visualization_info_modal_content,
        ],
        queue=False,
        show_progress="hidden",
    )

    language_dropdown.change(
        fn=None,
        inputs=[
            language_selector.dropdown,
        ],
        outputs=[],
        js=EXAMPLES_UI_JS,
        queue=False,
        show_progress="hidden",
    )

    if app_content.examples is not None:
        example_audio_change_event = app_content.examples.load_input_event.then(
            fn=handle_file_audio_change_event,
            inputs=[
                app_content.audio_input,
                app_content.enhanced_audio,
                language_selector.dropdown,
            ],
            outputs=audio_change_outputs,
            queue=False,
            show_progress="hidden",
        )

        example_audio_change_event.then(
            fn=handle_clear_visualization_pdf_exports,
            inputs=(visualization_pdf_cleanup_inputs),
            outputs=(visualization_download_button_components),
            queue=False,
            show_progress="hidden",
        )

    upload_audio_change_event = audio_input.upload(
        fn=handle_file_audio_change_event,
        inputs=[
            app_content.audio_input,
            app_content.enhanced_audio,
            language_selector.dropdown,
        ],
        outputs=audio_change_outputs,
        js=CLEAR_EXAMPLE_SELECTION_JS,
        queue=False,
        show_progress="hidden",
    )

    upload_audio_change_event.then(
        fn=handle_clear_visualization_pdf_exports,
        inputs=visualization_pdf_cleanup_inputs,
        outputs=(visualization_download_button_components),
        queue=False,
        show_progress="hidden",
    )

    recorded_audio_change_event = audio_input.stop_recording(
        fn=handle_recorded_audio_change_event,
        inputs=[
            app_content.audio_input,
            app_content.enhanced_audio,
            language_selector.dropdown,
        ],
        outputs=audio_change_outputs,
        js=CLEAR_EXAMPLE_SELECTION_JS,
        queue=False,
        show_progress="hidden",
    )

    recorded_audio_change_event.then(
        fn=handle_clear_visualization_pdf_exports,
        inputs=visualization_pdf_cleanup_inputs,
        outputs=(visualization_download_button_components),
        queue=False,
        show_progress="hidden",
    )

    enhancement_event = run_button.click(
        fn=handle_enhancement_started_event,
        inputs=[
            app_content.audio_input,
            app_content.enhanced_audio,
            language_selector.dropdown,
            app_content.audio_filename_state,
        ],
        outputs=enhancement_outputs,
        queue=False,
        show_progress="hidden",
    )

    enhancement_cleanup_event = enhancement_event.then(
        fn=handle_clear_visualization_pdf_exports,
        inputs=(visualization_pdf_cleanup_inputs),
        outputs=(visualization_download_button_components),
        queue=False,
        show_progress="hidden",
    )

    enhancement_run_event = enhancement_cleanup_event.then(
        fn=handle_run_enhancement_event,
        inputs=[
            app_content.audio_input,
            language_selector.dropdown,
            app_content.audio_filename_state,
        ],
        outputs=enhancement_outputs,
        queue=True,
        concurrency_limit=1,
        show_progress="hidden",
    )

    enhancement_run_event.then(
        fn=handle_refresh_visualization_pdf_exports,
        inputs=visualization_pdf_export_inputs,
        outputs=(visualization_download_button_components),
        queue=True,
        concurrency_limit=1,
        concurrency_id=("visualization-pdf-export"),
        show_progress="hidden",
    )

    clear_application_event = clear_button.click(
        fn=handle_clear_application_event,
        inputs=[
            language_selector.dropdown,
            app_content.enhanced_audio,
        ],
        outputs=clear_outputs,
        js=CLEAR_EXAMPLE_SELECTION_JS,
        queue=False,
        show_progress="hidden",
    )

    clear_application_event.then(
        fn=handle_clear_visualization_pdf_exports,
        inputs=visualization_pdf_cleanup_inputs,
        outputs=(visualization_download_button_components),
        queue=False,
        show_progress="hidden",
    )

    audio_clear_event = audio_input.clear(
        fn=handle_clear_application_event,
        inputs=[
            language_selector.dropdown,
            app_content.enhanced_audio,
        ],
        outputs=clear_outputs,
        js=CLEAR_EXAMPLE_SELECTION_JS,
        queue=False,
        show_progress="hidden",
    )

    audio_clear_event.then(
        fn=handle_clear_visualization_pdf_exports,
        inputs=visualization_pdf_cleanup_inputs,
        outputs=(visualization_download_button_components),
        queue=False,
        show_progress="hidden",
    )

    audio_info_button.click(
        fn=handle_show_audio_info,
        inputs=[],
        outputs=[
            app_content.audio_info_modal,
        ],
        queue=False,
        show_progress="hidden",
    )

    audio_info_modal_close_button.click(
        fn=handle_hide_audio_info,
        inputs=[],
        outputs=[
            app_content.audio_info_modal,
        ],
        js=MODAL_CLOSE_ANIMATION_JS,
        queue=False,
        show_progress="hidden",
    )

    processing_time_button.click(
        fn=handle_show_processing_timing_event,
        inputs=[
            app_content.processing_summary_state,
            language_selector.dropdown,
        ],
        outputs=[
            app_content.processing_modal,
            app_content.processing_modal_content,
            app_content.processing_modal_close_button,
        ],
        queue=False,
        show_progress="hidden",
    )

    visualization_info_events = (
        (
            spectrogram_info_button,
            VisualizationInfoKey.SPECTROGRAM,
        ),
        (
            expert_occupancy_info_button,
            VisualizationInfoKey.EXPERT_OCCUPANCY,
        ),
        (
            layer_routing_info_button,
            VisualizationInfoKey.LAYER_ROUTING,
        ),
        (
            frequency_routing_info_button,
            VisualizationInfoKey.FREQUENCY_ROUTING,
        ),
        (
            load_over_time_info_button,
            VisualizationInfoKey.LOAD_OVER_TIME,
        ),
    )

    for (
        info_button,
        visualization_key,
    ) in visualization_info_events:
        info_button.click(
            fn=partial(
                handle_show_visualization_info_event,
                visualization_key,
            ),
            inputs=[
                language_selector.dropdown,
            ],
            outputs=(visualization_info_modal_components),
            queue=False,
            show_progress="hidden",
        )

    visualization_info_modal_close_button.click(
        fn=handle_hide_visualization_info_event,
        inputs=[],
        outputs=(visualization_info_modal_components),
        js=MODAL_CLOSE_ANIMATION_JS,
        queue=False,
        show_progress="hidden",
    )

    processing_modal_close_button.click(
        fn=handle_hide_processing_modal,
        inputs=[],
        outputs=[
            app_content.processing_modal,
        ],
        js=MODAL_CLOSE_ANIMATION_JS,
        queue=False,
        show_progress="hidden",
    )
