"""
File: settings.py

Author: Ksenia Zolina, Dmitry Ryumin, Denis Ivanko, and Alexey Karpov

Description: Settings tab UI components for the WAVES Gradio application.

License: MIT License
"""

from dataclasses import dataclass

import gradio as gr

from waves.localization import get_localized_text


@dataclass(frozen=True, slots=True)
class SettingsTabComponents:
    """Components created inside the settings tab."""

    title: gr.Markdown
    description: gr.Markdown
    placeholder: gr.Markdown


def create_settings_tab(
    language_index: int = 0,
) -> SettingsTabComponents:
    """Create the settings tab."""

    title = gr.Markdown(
        f"### {get_localized_text('Texts_SETTINGS_TITLE', language_index)}",
        elem_classes="settings-title",
    )

    description = gr.Markdown(
        get_localized_text(
            "Texts_SETTINGS_DESCRIPTION",
            language_index,
        ),
        elem_classes="settings-description",
    )

    with gr.Column(elem_classes="settings-card"):
        placeholder = gr.Markdown(
            get_localized_text(
                "Texts_SETTINGS_PLACEHOLDER",
                language_index,
            ),
            elem_classes="settings-placeholder",
        )

    return SettingsTabComponents(
        title=title,
        description=description,
        placeholder=placeholder,
    )
