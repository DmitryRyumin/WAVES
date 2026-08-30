"""
File: export.py

Author: Ksenia Zolina, Dmitry Ryumin, Denis Ivanko, and Alexey Karpov

Description: White-theme PDF export utilities for WAVES Plotly visualizations.

License: MIT License
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
import shutil
import tempfile
from typing import cast

import plotly.graph_objects as go
import plotly.io as pio


class VisualizationExportKey(StrEnum):
    """Stable identifiers for exportable WAVES visualizations."""

    SPECTROGRAM = "spectrogram"
    EXPERT_OCCUPANCY = "expert_occupancy"
    LAYER_ROUTING = "layer_routing"
    FREQUENCY_ROUTING = "frequency_routing"
    LOAD_OVER_TIME = "load_over_time"


@dataclass(frozen=True, slots=True)
class VisualizationPdfExports:
    """Filesystem paths for generated visualization PDF files."""

    spectrogram: str | None = None
    expert_occupancy: str | None = None
    layer_routing: str | None = None
    frequency_routing: str | None = None
    load_over_time: str | None = None

    def as_tuple(
        self,
    ) -> tuple[
        str | None,
        str | None,
        str | None,
        str | None,
        str | None,
    ]:
        """Return export paths in application UI order."""

        return (
            self.spectrogram,
            self.expert_occupancy,
            self.layer_routing,
            self.frequency_routing,
            self.load_over_time,
        )


EXPORT_DIRECTORY_PREFIX = "waves-visualizations-"

EXPORT_FILENAMES: dict[
    VisualizationExportKey,
    str,
] = {
    VisualizationExportKey.SPECTROGRAM: ("WAVES_Spectrogram_Comparison.pdf"),
    VisualizationExportKey.EXPERT_OCCUPANCY: ("WAVES_Expert_Occupancy.pdf"),
    VisualizationExportKey.LAYER_ROUTING: ("WAVES_Layer_Routing.pdf"),
    VisualizationExportKey.FREQUENCY_ROUTING: ("WAVES_Expert_Load_by_Frequency.pdf"),
    VisualizationExportKey.LOAD_OVER_TIME: ("WAVES_Expert_Load_over_Time.pdf"),
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

DEFAULT_EXPORT_HEIGHT = 500

EXPORT_BACKGROUND_COLOR = "#FFFFFF"
EXPORT_TEXT_COLOR = "#111827"
EXPORT_AXIS_COLOR = "#6B7280"
EXPORT_GRID_COLOR = "rgba(107, 114, 128, 0.16)"


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


def create_visualization_pdf_exports_from_plot_json(
    plots: Mapping[
        VisualizationExportKey,
        str | None,
    ],
) -> VisualizationPdfExports:
    """Export serialized Plotly figures to white-theme PDF files."""

    figures: dict[
        VisualizationExportKey,
        go.Figure | None,
    ] = {
        export_key: (create_plotly_figure_from_json(plot_json) if plot_json is not None else None)
        for (
            export_key,
            plot_json,
        ) in plots.items()
    }

    return create_visualization_pdf_exports(figures)


def create_visualization_pdf_exports(
    figures: Mapping[
        VisualizationExportKey,
        go.Figure | None,
    ],
) -> VisualizationPdfExports:
    """Export available Plotly figures to tightly sized white-theme PDF files."""

    export_directory = Path(
        tempfile.mkdtemp(
            prefix=EXPORT_DIRECTORY_PREFIX,
        )
    )

    export_figures: list[go.Figure] = []

    export_paths: list[str | Path] = []

    export_widths: list[int | None] = []

    export_heights: list[int | None] = []

    generated_paths: dict[
        VisualizationExportKey,
        str,
    ] = {}

    try:
        for export_key in VisualizationExportKey:
            figure = figures.get(export_key)

            if figure is None:
                continue

            (
                export_figure,
                width,
                height,
            ) = _create_white_export_figure(
                figure,
                export_key,
            )

            export_path = export_directory / EXPORT_FILENAMES[export_key]

            export_figures.append(export_figure)

            export_paths.append(export_path)

            export_widths.append(width)

            export_heights.append(height)

            generated_paths[export_key] = str(export_path)

        if not export_figures:
            export_directory.rmdir()

            return VisualizationPdfExports()

        pio.write_images(
            fig=export_figures,
            file=export_paths,
            format="pdf",
            width=export_widths,
            height=export_heights,
            scale=1,
        )

    except Exception:
        shutil.rmtree(
            export_directory,
            ignore_errors=True,
        )

        raise

    return VisualizationPdfExports(
        spectrogram=(generated_paths.get(VisualizationExportKey.SPECTROGRAM)),
        expert_occupancy=(generated_paths.get(VisualizationExportKey.EXPERT_OCCUPANCY)),
        layer_routing=(generated_paths.get(VisualizationExportKey.LAYER_ROUTING)),
        frequency_routing=(generated_paths.get(VisualizationExportKey.FREQUENCY_ROUTING)),
        load_over_time=(generated_paths.get(VisualizationExportKey.LOAD_OVER_TIME)),
    )


def remove_visualization_pdf_exports(
    paths: Iterable[str | None],
) -> None:
    """Remove WAVES-generated visualization PDF files and empty export directories."""

    temporary_root = Path(tempfile.gettempdir()).resolve()

    allowed_filenames = set(EXPORT_FILENAMES.values())

    parent_directories: set[Path] = set()

    for raw_path in paths:
        if not raw_path:
            continue

        path = Path(raw_path).expanduser().resolve()

        if (
            path.name not in allowed_filenames
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
