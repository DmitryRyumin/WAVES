"""
File: visualization_info.py

Author: Ksenia Zolina, Dmitry Ryumin, Denis Ivanko, and Alexey Karpov

Description: HTML rendering for WAVES scientific visualization information.

License: MIT License
"""

from html import escape

from waves.config import (
    get_config_float,
    get_config_int,
)
from waves.localization import get_localized_text
from waves.ui.visualization_info_specs import (
    SPECS,
    VisualizationInfoKey,
    VisualizationInfoSpec,
)

SECTION_LABEL_KEYS = {
    "badge": "VisualizationInfoLabels_BADGE",
    "shows": "VisualizationInfoLabels_SHOWS",
    "read": "VisualizationInfoLabels_READ",
    "calculation": "VisualizationInfoLabels_CALCULATION",
    "notation": "VisualizationInfoLabels_NOTATION",
    "interpretation": "VisualizationInfoLabels_INTERPRETATION",
    "settings": "VisualizationInfoLabels_SETTINGS",
    "important": "VisualizationInfoLabels_IMPORTANT",
}


def _text(
    key: str,
    language_index: int,
) -> str:
    """Return one localized visualization information string."""

    return get_localized_text(
        key,
        language_index,
    )


def _create_read_rows_html(
    rows: tuple[tuple[str, str], ...],
    language_index: int,
) -> str:
    """Create compact reading-guide rows."""

    return "".join(
        (
            '<div class="visualization-info-read-row">'
            '<div class="visualization-info-read-label">'
            f"{escape(_text(label_key, language_index))}"
            "</div>"
            '<div class="visualization-info-read-value">'
            f"{escape(_text(value_key, language_index))}"
            "</div>"
            "</div>"
        )
        for label_key, value_key in rows
    )


def _create_formula_html(
    formulas: tuple[str, ...],
) -> str:
    """Create MathML formula blocks."""

    return "".join((f'<div class="visualization-info-formula">{formula}</div>') for formula in formulas)


def _create_notation_html(
    notation: tuple[tuple[str, str], ...],
    language_index: int,
) -> str:
    """Create the mathematical notation glossary."""

    return "".join(
        (
            '<div class="visualization-info-notation-row">'
            '<div class="visualization-info-notation-symbol">'
            f"{symbol}"
            "</div>"
            '<div class="visualization-info-notation-text">'
            f"{escape(_text(description_key, language_index))}"
            "</div>"
            "</div>"
        )
        for symbol, description_key in notation
    )


def _create_interpretation_html(
    keys: tuple[str, ...],
    language_index: int,
) -> str:
    """Create concise interpretation points."""

    return "".join(
        (
            '<div class="visualization-info-point">'
            '<span class="visualization-info-point-marker">'
            "</span>"
            "<span>"
            f"{escape(_text(key, language_index))}"
            "</span>"
            "</div>"
        )
        for key in keys
    )


def _create_setting_chip(
    label: str,
    value: str,
) -> str:
    """Create one current-setting chip."""

    return (
        '<span class="visualization-info-setting">'
        '<span class="visualization-info-setting-label">'
        f"{escape(label)}"
        "</span>"
        '<span class="visualization-info-setting-value">'
        f"{escape(value)}"
        "</span>"
        "</span>"
    )


def _create_spectrogram_settings_html(
    language_index: int,
) -> str:
    """Create the current spectrogram settings."""

    n_fft = get_config_int(
        "Visualization_SPECTROGRAM_N_FFT",
        2048,
    )

    hop_length = get_config_int(
        "Visualization_SPECTROGRAM_HOP_LENGTH",
        512,
    )

    num_mels = get_config_int(
        "Visualization_SPECTROGRAM_NUM_MELS",
        80,
    )

    max_frequency = get_config_float(
        "Visualization_SPECTROGRAM_MAX_FREQUENCY",
        8000.0,
    )

    top_db = get_config_float(
        "Visualization_SPECTROGRAM_TOP_DB",
        80.0,
    )

    max_delta_db = get_config_float(
        "Visualization_SPECTROGRAM_DELTA_MAX_DB",
        20.0,
    )

    settings = (
        (
            "VisualizationInfoSpectrogram_SETTING_FFT",
            str(n_fft),
        ),
        (
            "VisualizationInfoSpectrogram_SETTING_HOP",
            str(hop_length),
        ),
        (
            "VisualizationInfoSpectrogram_SETTING_MELS",
            str(num_mels),
        ),
        (
            "VisualizationInfoSpectrogram_SETTING_MAX_FREQUENCY",
            f"{max_frequency / 1000.0:g} kHz",
        ),
        (
            "VisualizationInfoSpectrogram_SETTING_RANGE",
            f"{top_db:g} dB",
        ),
        (
            "VisualizationInfoSpectrogram_SETTING_MAX_DELTA",
            f"{max_delta_db:g} dB",
        ),
    )

    chips_html = "".join(
        _create_setting_chip(
            _text(
                label_key,
                language_index,
            ),
            value,
        )
        for label_key, value in settings
    )

    title = _text(
        SECTION_LABEL_KEYS["settings"],
        language_index,
    )

    return (
        '<section class="visualization-info-section '
        'visualization-info-settings-section">'
        '<div class="visualization-info-section-title">'
        f"{escape(title)}"
        "</div>"
        '<div class="visualization-info-settings">'
        f"{chips_html}"
        "</div>"
        "</section>"
    )


def create_visualization_info_html(
    visualization_key: str | VisualizationInfoKey,
    language_index: int,
) -> str:
    """Create the information modal content for one visualization."""

    key = VisualizationInfoKey(visualization_key)

    spec = SPECS[key]

    labels = {
        name: _text(
            config_key,
            language_index,
        )
        for name, config_key in SECTION_LABEL_KEYS.items()
    }

    settings_html = _create_spectrogram_settings_html(language_index) if spec.show_settings else ""

    read_rows_html = _create_read_rows_html(
        spec.read_rows,
        language_index,
    )

    notation_html = _create_notation_html(
        spec.notation,
        language_index,
    )

    interpretation_html = _create_interpretation_html(
        spec.interpretation_keys,
        language_index,
    )

    formulas_html = _create_formula_html(spec.formulas)

    return f"""
<div
    class="visualization-info-body"
    data-visualization-key="{escape(key.value)}"
>
    <div class="visualization-info-heading">
        <div class="visualization-info-badge">
            {escape(labels["badge"])}
        </div>

        <div class="visualization-info-title">
            {escape(_text(spec.title_key, language_index))}
        </div>

        <div class="visualization-info-subtitle">
            {escape(_text(spec.subtitle_key, language_index))}
        </div>
    </div>

    <div class="visualization-info-top-grid">
        <section class="visualization-info-section">
            <div class="visualization-info-section-title">
                {escape(labels["shows"])}
            </div>

            <div class="visualization-info-section-copy">
                {escape(_text(spec.shows_key, language_index))}
            </div>
        </section>

        <section class="visualization-info-section">
            <div class="visualization-info-section-title">
                {escape(labels["read"])}
            </div>

            <div class="visualization-info-read-grid">
                {read_rows_html}
            </div>
        </section>
    </div>

    <section class="visualization-info-section">
        <div class="visualization-info-section-title">
            {escape(labels["calculation"])}
        </div>

        <div class="visualization-info-formulas">
            {formulas_html}
        </div>
    </section>

    <section class="visualization-info-section">
        <div class="visualization-info-section-title">
            {escape(labels["notation"])}
        </div>

        <div class="visualization-info-notation">
            {notation_html}
        </div>
    </section>

    {settings_html}

    <section class="visualization-info-section">
        <div class="visualization-info-section-title">
            {escape(labels["interpretation"])}
        </div>

        <div class="visualization-info-points">
            {interpretation_html}
        </div>
    </section>

    <div class="visualization-info-note">
        <div class="visualization-info-note-title">
            {escape(labels["important"])}
        </div>

        <div class="visualization-info-note-text">
            {escape(_text(spec.note_key, language_index))}
        </div>
    </div>
</div>
""".strip()


__all__ = [
    "VisualizationInfoKey",
    "VisualizationInfoSpec",
    "create_visualization_info_html",
]
