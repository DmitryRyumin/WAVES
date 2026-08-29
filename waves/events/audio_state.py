"""
File: audio_state.py

Author: Ksenia Zolina, Dmitry Ryumin, Denis Ivanko, and Alexey Karpov

Description: Shared audio state builders for WAVES application events.

License: MIT License
"""

from dataclasses import dataclass
from pathlib import Path

from waves.audio.formatting import (
    create_unavailable_audio_metadata_html,
    format_audio_metadata_html,
    format_audio_validation_error_markdown,
)
from waves.audio.validation import validate_audio_file
from waves.localization import get_localized_text


@dataclass(frozen=True, slots=True)
class AudioStateContent:
    """Localized UI content describing the current input-audio state."""

    status_text: str
    audio_info_html: str
    is_valid: bool


def get_audio_display_filename(
    audio_path: str | None,
) -> str | None:
    """Return a safe display filename for an audio path."""

    if not audio_path:
        return None

    filename = Path(audio_path).name.strip()

    return filename or None


def create_audio_component_label(
    config_field: str,
    language_index: int,
    audio_filename: str | None,
) -> str:
    """Create a localized audio-component label with an optional filename."""

    label = get_localized_text(
        config_field,
        language_index,
    ).strip()

    if not audio_filename:
        return label

    return f"{label} ({audio_filename})"


def create_audio_state_content(
    audio_path: str | None,
    language_index: int,
    enhanced_audio_path: str | None = None,
) -> AudioStateContent:
    """Create localized status and metadata content for the current audio state."""

    if not audio_path:
        return AudioStateContent(
            status_text=get_localized_text(
                "Texts_STATUS_READY",
                language_index,
            ),
            audio_info_html="",
            is_valid=False,
        )

    validation_result = validate_audio_file(audio_path)

    if validation_result.metadata is not None:
        audio_info_html = format_audio_metadata_html(
            metadata=validation_result.metadata,
            language_index=language_index,
            validation_result=validation_result,
        )
    else:
        unavailable_html = create_unavailable_audio_metadata_html(
            language_index=language_index,
        )

        audio_info_html = f'<div class="audio-info-unavailable-error">{unavailable_html}</div>'

    if not validation_result.is_valid:
        status_text = format_audio_validation_error_markdown(
            validation_result=validation_result,
            language_index=language_index,
        )
    elif enhanced_audio_path:
        status_text = get_localized_text(
            "Texts_STATUS_COMPLETED",
            language_index,
        )
    else:
        status_text = get_localized_text(
            "Texts_STATUS_AUDIO_READY",
            language_index,
        )

    return AudioStateContent(
        status_text=status_text,
        audio_info_html=audio_info_html,
        is_valid=validation_result.is_valid,
    )
