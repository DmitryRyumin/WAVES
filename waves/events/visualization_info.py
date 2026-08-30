"""
File: visualization_info.py

Author: Ksenia Zolina, Dmitry Ryumin, Denis Ivanko, and Alexey Karpov

Description: Event helpers for WAVES visualization information modals.

License: MIT License
"""

from dataclasses import dataclass
from typing import Any

import gradio as gr

from waves.localization import get_language_index
from waves.ui.visualization_info import (
    VisualizationInfoKey,
    create_visualization_info_html,
)


@dataclass(frozen=True, slots=True)
class VisualizationInfoModalUpdates:
    """Named updates for the shared visualization information modal."""

    modal: Any
    content: Any
    close_button: Any
    key_state: str | None


def _render_visualization_info(
    visualization_key: str | VisualizationInfoKey,
    language: str,
) -> tuple[
    VisualizationInfoKey,
    str,
]:
    """Render localized content for one visualization."""

    key = VisualizationInfoKey(visualization_key)

    language_index = get_language_index(language)

    content = create_visualization_info_html(
        key,
        language_index,
    )

    return (
        key,
        content,
    )


def handle_show_visualization_info(
    visualization_key: str | VisualizationInfoKey,
    language: str,
) -> VisualizationInfoModalUpdates:
    """Open the shared visualization information modal."""

    key, content = _render_visualization_info(
        visualization_key,
        language,
    )

    return VisualizationInfoModalUpdates(
        modal=gr.update(
            visible=True,
        ),
        content=gr.update(
            value=content,
        ),
        close_button=gr.update(
            visible=True,
        ),
        key_state=key.value,
    )


def handle_hide_visualization_info() -> VisualizationInfoModalUpdates:
    """Hide the shared visualization information modal."""

    return VisualizationInfoModalUpdates(
        modal=gr.update(
            visible=False,
        ),
        content=gr.update(
            value="",
        ),
        close_button=gr.update(
            visible=False,
        ),
        key_state=None,
    )


def handle_refresh_visualization_info(
    visualization_key: str | None,
    language: str,
) -> Any:
    """Refresh open modal content after a language change."""

    if visualization_key is None:
        return gr.update()

    _, content = _render_visualization_info(
        visualization_key,
        language,
    )

    return gr.update(
        value=content,
    )
