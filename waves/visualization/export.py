"""
File: export.py

Author: Ksenia Zolina, Dmitry Ryumin, Denis Ivanko, and Alexey Karpov

Description: White-theme PDF export utilities for WAVES Plotly visualizations.

License: MIT License
"""

from collections.abc import Iterable
from contextlib import suppress
from enum import StrEnum
from pathlib import Path
import shutil
import tempfile
from typing import cast

import plotly.graph_objects as go
import plotly.io as pio

from waves.config import (
    get_config_bool,
    get_config_str_list,
)


class VisualizationExportKey(StrEnum):
    """Stable identifiers for exportable WAVES visualizations."""

    SPECTROGRAM = "spectrogram"
    EXPERT_OCCUPANCY = "expert_occupancy"
    LAYER_ROUTING = "layer_routing"
    FREQUENCY_ROUTING = "frequency_routing"
    LOAD_OVER_TIME = "load_over_time"


EXPORT_DIRECTORY_PREFIX = "waves-visualizations-"

EXPORT_FILENAME_STEMS: dict[
    VisualizationExportKey,
    str,
] = {
    VisualizationExportKey.SPECTROGRAM: ("WAVES_Spectrogram_Comparison"),
    VisualizationExportKey.EXPERT_OCCUPANCY: ("WAVES_Expert_Occupancy"),
    VisualizationExportKey.LAYER_ROUTING: ("WAVES_Layer_Routing"),
    VisualizationExportKey.FREQUENCY_ROUTING: ("WAVES_Expert_Load_by_Frequency"),
    VisualizationExportKey.LOAD_OVER_TIME: ("WAVES_Expert_Load_over_Time"),
}

EXPORT_WIDTHS: dict[
    VisualizationExportKey,
    int,
] = {
    VisualizationExportKey.SPECTROGRAM: 1280,
    VisualizationExportKey.EXPERT_OCCUPANCY: 840,
    VisualizationExportKey.LAYER_ROUTING: 960,
    VisualizationExportKey.FREQUENCY_ROUTING: 900,
    VisualizationExportKey.LOAD_OVER_TIME: 900,
}

TITLELESS_TOP_MARGINS: dict[
    VisualizationExportKey,
    int,
] = {
    VisualizationExportKey.SPECTROGRAM: 44,
    VisualizationExportKey.EXPERT_OCCUPANCY: 32,
    VisualizationExportKey.LAYER_ROUTING: 28,
    VisualizationExportKey.FREQUENCY_ROUTING: 32,
    VisualizationExportKey.LOAD_OVER_TIME: 32,
}

DEFAULT_EXPORT_HEIGHT = 500

EXPORT_BACKGROUND_COLOR = "#FFFFFF"
EXPORT_TEXT_COLOR = "#111827"
EXPORT_AXIS_COLOR = "#6B7280"
EXPORT_GRID_COLOR = "rgba(107, 114, 128, 0.16)"

DEFAULT_LANGUAGE_CHOICES = [
    "English",
    "Русский",
]

DEFAULT_LANGUAGE_CODES = [
    "EN",
    "RU",
]


def create_plotly_figure_from_json(
    plot_json: str,
) -> go.Figure:
    """Create a Plotly figure from serialized Gradio Plot data."""

    return go.Figure(pio.from_json(plot_json))


def _get_layout_dimension(
    figure: go.Figure,
    dimension_name: str,
    default: int,
) -> int:
    """Read a positive numeric dimension from a Plotly figure layout."""

    figure_json = cast(
        dict[str, object],
        figure.to_plotly_json(),
    )

    layout_value = figure_json.get("layout")

    layout = (
        cast(
            dict[str, object],
            layout_value,
        )
        if isinstance(
            layout_value,
            dict,
        )
        else {}
    )

    dimension = layout.get(dimension_name)

    if isinstance(
        dimension,
        (int, float),
    ) and not isinstance(
        dimension,
        bool,
    ):
        resolved_dimension = int(dimension)

        if resolved_dimension > 0:
            return resolved_dimension

    return default


def _resolve_export_dimensions(
    figure: go.Figure,
    export_key: VisualizationExportKey,
) -> tuple[
    int,
    int,
]:
    """Resolve deterministic canvas dimensions for one PDF export."""

    width = _get_layout_dimension(
        figure=figure,
        dimension_name="width",
        default=EXPORT_WIDTHS[export_key],
    )

    height = _get_layout_dimension(
        figure=figure,
        dimension_name="height",
        default=DEFAULT_EXPORT_HEIGHT,
    )

    return (
        width,
        height,
    )


def _resolve_language_code(
    language: str,
) -> str:
    """Resolve the configured short code for the selected UI language."""

    language_choices = get_config_str_list(
        "Languages_CHOICES",
        DEFAULT_LANGUAGE_CHOICES,
    )

    language_codes = get_config_str_list(
        "Languages_CODES",
        DEFAULT_LANGUAGE_CODES,
    )

    try:
        language_index = language_choices.index(language)
    except ValueError:
        language_index = 0

    if language_index < len(language_codes):
        language_code = language_codes[language_index].strip().upper()

        if language_code:
            return language_code

    return f"LANG{language_index + 1}"


def _create_export_filename(
    export_key: VisualizationExportKey,
    language: str,
) -> str:
    """Create the configured PDF filename for one visualization."""

    filename_stem = EXPORT_FILENAME_STEMS[export_key]

    if get_config_bool(
        "VisualizationExport_SHOW_LANGUAGE_SUFFIX",
        True,
    ):
        language_code = _resolve_language_code(language)

        filename_stem = f"{filename_stem}_{language_code}"

    return f"{filename_stem}.pdf"


def _is_visualization_export_filename(
    filename: str,
) -> bool:
    """Return whether a filename belongs to a WAVES visualization export."""

    if not filename.endswith(".pdf"):
        return False

    filename_stem = filename.removesuffix(".pdf")

    return any(
        filename_stem == allowed_stem or filename_stem.startswith(f"{allowed_stem}_")
        for allowed_stem in EXPORT_FILENAME_STEMS.values()
    )


def _create_white_export_figure(
    figure: go.Figure,
    export_key: VisualizationExportKey,
) -> tuple[
    go.Figure,
    int,
    int,
]:
    """Create a white publication-style copy without changing the UI figure."""

    export_figure = go.Figure(figure)

    (
        width,
        height,
    ) = _resolve_export_dimensions(
        export_figure,
        export_key,
    )

    export_figure.update_layout(
        autosize=False,
        width=width,
        height=height,
        paper_bgcolor=(EXPORT_BACKGROUND_COLOR),
        plot_bgcolor=(EXPORT_BACKGROUND_COLOR),
        font={
            "color": EXPORT_TEXT_COLOR,
        },
        legend={
            "font": {
                "color": (EXPORT_TEXT_COLOR),
            },
        },
    )

    if not get_config_bool(
        "VisualizationExport_SHOW_TITLE",
        True,
    ):
        export_figure.update_layout(
            title=None,
            margin={
                "t": TITLELESS_TOP_MARGINS[export_key],
            },
        )

    export_figure.update_xaxes(
        color=EXPORT_TEXT_COLOR,
        linecolor=EXPORT_AXIS_COLOR,
        tickcolor=EXPORT_AXIS_COLOR,
        gridcolor=EXPORT_GRID_COLOR,
        title_font={
            "color": EXPORT_TEXT_COLOR,
        },
        tickfont={
            "color": EXPORT_TEXT_COLOR,
        },
    )

    export_figure.update_yaxes(
        color=EXPORT_TEXT_COLOR,
        linecolor=EXPORT_AXIS_COLOR,
        tickcolor=EXPORT_AXIS_COLOR,
        gridcolor=EXPORT_GRID_COLOR,
        title_font={
            "color": EXPORT_TEXT_COLOR,
        },
        tickfont={
            "color": EXPORT_TEXT_COLOR,
        },
    )

    return (
        export_figure,
        width,
        height,
    )


def create_visualization_pdf_export_directory() -> Path:
    """Create a temporary directory for one set of visualization PDF files."""

    return Path(
        tempfile.mkdtemp(
            prefix=EXPORT_DIRECTORY_PREFIX,
        )
    )


def create_visualization_pdf_export_from_plot_json(
    export_key: VisualizationExportKey,
    plot_json: str,
    export_directory: Path,
    language: str,
) -> str:
    """Export one serialized Plotly visualization to a white-theme PDF."""

    figure = create_plotly_figure_from_json(plot_json)

    return create_visualization_pdf_export(
        export_key=export_key,
        figure=figure,
        export_directory=export_directory,
        language=language,
    )


def create_visualization_pdf_export(
    export_key: VisualizationExportKey,
    figure: go.Figure,
    export_directory: Path,
    language: str,
) -> str:
    """Export one Plotly visualization to a tightly sized white-theme PDF."""

    (
        export_figure,
        width,
        height,
    ) = _create_white_export_figure(
        figure,
        export_key,
    )

    export_path = export_directory / _create_export_filename(
        export_key,
        language,
    )

    try:
        pio.write_image(
            fig=export_figure,
            file=export_path,
            format="pdf",
            width=width,
            height=height,
            scale=1,
        )
    except Exception:
        with suppress(OSError):
            export_path.unlink(missing_ok=True)

        raise

    return str(export_path)


def remove_visualization_pdf_export_directory(
    export_directory: str | Path | None,
) -> None:
    """Safely remove one WAVES visualization export directory."""

    if export_directory is None:
        return

    temporary_root = Path(tempfile.gettempdir()).resolve()

    directory = Path(export_directory).expanduser().resolve()

    if not directory.name.startswith(EXPORT_DIRECTORY_PREFIX) or not directory.is_relative_to(temporary_root):
        return

    shutil.rmtree(
        directory,
        ignore_errors=True,
    )


def remove_visualization_pdf_exports(
    paths: Iterable[str | None],
) -> None:
    """Remove WAVES-generated visualization PDF files and empty directories."""

    temporary_root = Path(tempfile.gettempdir()).resolve()

    parent_directories: set[Path] = set()

    for raw_path in paths:
        if not raw_path:
            continue

        path = Path(raw_path).expanduser().resolve()

        if (
            not _is_visualization_export_filename(path.name)
            or not path.parent.name.startswith(EXPORT_DIRECTORY_PREFIX)
            or not path.is_relative_to(temporary_root)
        ):
            continue

        try:
            path.unlink(missing_ok=True)
        except OSError:
            continue

        parent_directories.add(path.parent)

    for directory in parent_directories:
        try:
            directory.rmdir()
        except OSError:
            continue
