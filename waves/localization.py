"""
File: localization.py

Author: Ksenia Zolina, Dmitry Ryumin, Denis Ivanko, and Alexey Karpov

Description: Localization helpers for the WAVES Gradio application.

License: MIT License
"""

from typing import Final

from waves.config import (
    PROJECT_ROOT,
    get_config_str,
    get_config_str_list,
)

DEFAULT_LANGUAGE_INDEX: Final = 0


def get_localized_values(
    config_field: str,
) -> list[str]:
    """Return validated localized values from the WAVES configuration."""

    values = get_config_str_list(
        config_field,
        [],
    )

    if not values:
        msg = f"Localized configuration field '{config_field}' must be a non-empty list of strings."
        raise TypeError(msg)

    return values


def get_localized_text(
    config_field: str,
    language_index: int,
) -> str:
    """Return a localized string by configuration field and language index."""

    values = get_localized_values(config_field)

    if language_index < 0 or language_index >= len(values):
        language_index = DEFAULT_LANGUAGE_INDEX

    return values[language_index]


def get_language_index(
    language: str,
) -> int:
    """Return the index of the selected language."""

    languages = get_localized_values("Languages_CHOICES")

    try:
        return languages.index(language)
    except ValueError:
        return DEFAULT_LANGUAGE_INDEX


def get_language_flag_path(
    language_index: int,
) -> str:
    """Return the flag image path for the selected language."""

    images_dir = get_config_str(
        "StaticPaths_IMAGES",
        "static/images",
    )

    language_images = get_localized_values("Images_LANGUAGES")

    if language_index < 0 or language_index >= len(language_images):
        language_index = DEFAULT_LANGUAGE_INDEX

    return str(PROJECT_ROOT / images_dir / language_images[language_index])
