"""
File: event_handlers.py

Author: Ksenia Zolina, Dmitry Ryumin, Denis Ivanko, and Alexey Karpov

Description: Event handler registration for the WAVES Gradio application.

License: MIT License
"""

from collections.abc import Iterator
from typing import Any, cast

from waves.events.application import (
    handle_audio_change,
    handle_clear_application,
    handle_hide_audio_info,
    handle_run_enhancement,
    handle_show_audio_info,
)
from waves.events.language import handle_language_change
from waves.events.visualization import RoutingPlotUpdates
from waves.routing import RoutingTelemetry
from waves.ui.application import ApplicationTabComponents
from waves.ui.language_selector import LanguageSelectorComponents
from waves.ui.settings import SettingsTabComponents
from waves.ui.tabs import (
    AboutAppTabComponents,
    AppTabsComponents,
    RequirementsTabComponents,
)

LANGUAGE_PAGINATION_SYNC_JS = """
(language) => {
    const pagesLabel = language === "Русский" ? "Страницы:" : "Pages:";

    const syncPaginationLabel = () => {
        const paginations = document.querySelectorAll(
            "#application-examples div.paginate"
        );

        paginations.forEach((pagination) => {
            for (const node of pagination.childNodes) {
                if (
                    node.nodeType === Node.TEXT_NODE &&
                    node.textContent &&
                    node.textContent.trim().length > 0
                ) {
                    node.textContent = `${pagesLabel} `;
                    break;
                }
            }
        });
    };

    syncPaginationLabel();
    requestAnimationFrame(syncPaginationLabel);

    setTimeout(syncPaginationLabel, 50);
    setTimeout(syncPaginationLabel, 250);

    const examples = document.querySelector("#application-examples");

    if (examples) {
        if (window.__wavesPaginationObserver) {
            window.__wavesPaginationObserver.disconnect();
        }

        const observer = new MutationObserver(syncPaginationLabel);

        observer.observe(
            examples,
            {
                childList: true,
                subtree: true,
            }
        );

        window.__wavesPaginationObserver = observer;
    }

    return [];
}
"""


def setup_app_event_handlers(
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

    def map_routing_updates(
        updates: RoutingPlotUpdates,
    ) -> dict[Any, Any]:
        """Map named routing updates to their Gradio components."""

        return {
            app_content.expert_occupancy_plot: updates.expert_occupancy_plot,
            app_content.layer_routing_plot: updates.layer_routing_plot,
            app_content.frequency_routing_plot: updates.frequency_routing_plot,
            app_content.load_over_time_plot: updates.load_over_time_plot,
        }

    def handle_language_change_event(
        language: str,
        audio_path: str | None,
        enhanced_audio_path: str | None,
        routing: RoutingTelemetry | None,
    ) -> dict[Any, Any]:
        """Map named language updates to their Gradio components."""

        updates = handle_language_change(
            language=language,
            audio_path=audio_path,
            enhanced_audio_path=enhanced_audio_path,
            routing=routing,
        )

        component_updates: dict[Any, Any] = {
            language_selector.flag: updates.flag,
            app_content.title: updates.title,
            app_content.status: updates.status,
            app_content.audio_input: updates.audio_input,
            app_content.examples_title: updates.examples_title,
            app_content.examples_placeholder: updates.examples_placeholder,
            app_content.run_button: updates.run_button,
            app_content.clear_button: updates.clear_button,
            app_content.audio_info_button: updates.audio_info_button,
            app_content.audio_info_modal_title: updates.audio_info_modal_title,
            app_content.audio_info_modal_content: updates.audio_info_modal_content,
            app_content.enhanced_audio: updates.enhanced_audio,
            app_content.spectrogram_plot: updates.spectrogram_plot,
            settings_content.title: updates.settings_title,
            settings_content.description: updates.settings_description,
            settings_content.placeholder: updates.settings_placeholder,
            about_app_content.title: updates.about_app_title,
            about_app_content.description: updates.about_app_description,
            about_app_content.placeholder: updates.about_app_placeholder,
            requirements_content.title: updates.requirements_title,
            requirements_content.placeholder: updates.requirements_placeholder,
        }

        component_updates.update(map_routing_updates(updates.routing_plots))

        for tab_name, tab_update in updates.tabs.items():
            tab_component = app_tabs.tab_components.get(tab_name)

            if tab_component is not None:
                component_updates[tab_component] = tab_update

        return component_updates

    def handle_audio_change_event(
        audio_path: str | None,
        enhanced_audio_path: str | None,
        language: str,
    ) -> dict[Any, Any]:
        """Map named input-audio updates to their Gradio components."""

        updates = handle_audio_change(
            audio_path=audio_path,
            enhanced_audio_path=enhanced_audio_path,
            language=language,
        )

        component_updates: dict[Any, Any] = {
            app_content.status: updates.status,
            app_content.run_button: updates.run_button,
            app_content.clear_button: updates.clear_button,
            app_content.audio_info_button_column: updates.audio_info_button_column,
            app_content.audio_info_button: updates.audio_info_button,
            app_content.audio_info_modal: updates.audio_info_modal,
            app_content.audio_info_modal_content: updates.audio_info_modal_content,
            app_content.enhanced_audio: updates.enhanced_audio,
            app_content.spectrogram_plot: updates.spectrogram_plot,
            app_content.routing_state: updates.routing_state,
        }

        component_updates.update(map_routing_updates(updates.routing_plots))

        return component_updates

    def handle_run_enhancement_event(
        audio_path: str | None,
        enhanced_audio_path: str | None,
        language: str,
    ) -> Iterator[dict[Any, Any]]:
        """Map streamed enhancement updates to their Gradio components."""

        for updates in handle_run_enhancement(
            audio_path=audio_path,
            enhanced_audio_path=enhanced_audio_path,
            language=language,
        ):
            component_updates: dict[Any, Any] = {
                app_content.audio_input: updates.audio_input,
                app_content.status: updates.status,
                app_content.run_button: updates.run_button,
                app_content.clear_button: updates.clear_button,
                app_content.enhanced_audio: updates.enhanced_audio,
                app_content.spectrogram_plot: updates.spectrogram_plot,
                app_content.routing_state: updates.routing_state,
            }

            component_updates.update(map_routing_updates(updates.routing_plots))

            yield component_updates

    def handle_clear_application_event(
        language: str,
        enhanced_audio_path: str | None,
    ) -> dict[Any, Any]:
        """Map named clear-state updates to their Gradio components."""

        updates = handle_clear_application(
            language=language,
            enhanced_audio_path=enhanced_audio_path,
        )

        component_updates: dict[Any, Any] = {
            app_content.audio_input: updates.audio_input,
            app_content.status: updates.status,
            app_content.run_button: updates.run_button,
            app_content.clear_button: updates.clear_button,
            app_content.audio_info_button_column: updates.audio_info_button_column,
            app_content.audio_info_button: updates.audio_info_button,
            app_content.audio_info_modal: updates.audio_info_modal,
            app_content.audio_info_modal_content: updates.audio_info_modal_content,
            app_content.enhanced_audio: updates.enhanced_audio,
            app_content.spectrogram_plot: updates.spectrogram_plot,
            app_content.routing_state: updates.routing_state,
        }

        component_updates.update(map_routing_updates(updates.routing_plots))

        return component_updates

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
        app_content.status,
        app_content.run_button,
        app_content.clear_button,
        app_content.audio_info_button_column,
        app_content.audio_info_button,
        app_content.audio_info_modal,
        app_content.audio_info_modal_content,
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
        app_content.enhanced_audio,
        app_content.spectrogram_plot,
        app_content.routing_state,
        *routing_output_components,
    ]

    clear_outputs = [
        app_content.audio_input,
        app_content.status,
        app_content.run_button,
        app_content.clear_button,
        app_content.audio_info_button_column,
        app_content.audio_info_button,
        app_content.audio_info_modal,
        app_content.audio_info_modal_content,
        app_content.enhanced_audio,
        app_content.spectrogram_plot,
        app_content.routing_state,
        *routing_output_components,
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

    language_dropdown.change(
        fn=handle_language_change_event,
        inputs=[
            language_selector.dropdown,
            app_content.audio_input,
            app_content.enhanced_audio,
            app_content.routing_state,
        ],
        outputs=language_outputs,
        queue=False,
        show_progress="hidden",
    )

    language_dropdown.change(
        fn=None,
        inputs=[
            language_selector.dropdown,
        ],
        outputs=[],
        js=LANGUAGE_PAGINATION_SYNC_JS,
        queue=False,
        show_progress="hidden",
    )

    audio_input.change(
        fn=handle_audio_change_event,
        inputs=[
            app_content.audio_input,
            app_content.enhanced_audio,
            language_selector.dropdown,
        ],
        outputs=audio_change_outputs,
        queue=False,
        show_progress="hidden",
    )

    run_button.click(
        fn=handle_run_enhancement_event,
        inputs=[
            app_content.audio_input,
            app_content.enhanced_audio,
            language_selector.dropdown,
        ],
        outputs=enhancement_outputs,
        queue=True,
        concurrency_limit=1,
        show_progress="minimal",
    )

    clear_button.click(
        fn=handle_clear_application_event,
        inputs=[
            language_selector.dropdown,
            app_content.enhanced_audio,
        ],
        outputs=clear_outputs,
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
        queue=False,
        show_progress="hidden",
    )
