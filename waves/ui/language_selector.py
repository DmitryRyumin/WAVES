"""
File: language_selector.py

Author: Ksenia Zolina, Dmitry Ryumin, Denis Ivanko, and Alexey Karpov

Description: Language selector UI components for the WAVES Gradio application.

License: MIT License
"""

from dataclasses import dataclass
from pathlib import Path

from gradio.components import HTML, Dropdown
from gradio.layouts import Column, Row

from waves.config import get_config_str
from waves.localization import (
    DEFAULT_LANGUAGE_INDEX,
    get_language_flag_path,
    get_language_index,
    get_localized_values,
)


@dataclass(frozen=True, slots=True)
class LanguageSelectorComponents:
    """Components created for the WAVES language selector."""

    flag: HTML
    dropdown: Dropdown


def create_language_flag_html(
    language_index: int,
) -> str:
    """Create inline SVG markup for the selected language flag."""

    flag_path = Path(get_language_flag_path(language_index))

    if not flag_path.is_file():
        return ""

    return flag_path.read_text(encoding="utf-8")


def create_language_selector() -> LanguageSelectorComponents:
    """Create the WAVES language selector."""

    language_choices = get_localized_values("Languages_CHOICES")

    configured_default_language = get_config_str(
        "Languages_DEFAULT",
        language_choices[DEFAULT_LANGUAGE_INDEX],
    )

    default_language_index = get_language_index(configured_default_language)

    default_language = language_choices[default_language_index]

    with (
        Column(
            visible=True,
            render=True,
            variant="default",
            min_width=0,
            elem_classes="language-selector-wrapper",
        ),
        Row(
            visible=True,
            render=True,
            variant="default",
            elem_classes="language-selector",
        ),
    ):
        flag = HTML(
            value=create_language_flag_html(default_language_index),
            visible=True,
            elem_classes="language-selector-flag",
        )

        dropdown = Dropdown(
            label=None,
            info=None,
            choices=language_choices,
            value=default_language,
            visible=True,
            show_label=False,
            elem_classes="language-selector-dropdown",
            interactive=True,
            filterable=False,
            allow_custom_value=False,
            min_width=140,
        )

    return LanguageSelectorComponents(
        flag=flag,
        dropdown=dropdown,
    )
