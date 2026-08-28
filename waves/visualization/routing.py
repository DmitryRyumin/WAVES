"""
File: routing.py

Author: Ksenia Zolina, Dmitry Ryumin, Denis Ivanko, and Alexey Karpov

Description: Plotly visualizations for Mixture-of-Experts routing telemetry.

License: MIT License
"""

import math

import plotly.graph_objects as go
from torch import Tensor

from waves.config import get_config_int
from waves.localization import get_localized_text
from waves.routing import (
    RoutingAxis,
    RoutingTelemetry,
)

EXPERT_COLORS: tuple[str, ...] = (
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
)


def _get_expert_color(
    expert_index: int,
) -> str:
    """Return the stable categorical color assigned to an expert."""

    return EXPERT_COLORS[expert_index % len(EXPERT_COLORS)]


def _tensor_to_float_list(
    values: Tensor,
) -> list[float]:
    """Convert a tensor to a flat CPU float list."""

    return [
        float(value)
        for value in (
            values.detach()
            .to(
                device="cpu",
            )
            .flatten()
            .tolist()
        )
    ]


def _normalize_distribution(
    values: list[float],
) -> list[float]:
    """Normalize non-negative values to sum to one."""

    if not values:
        return []

    if any(value < 0.0 for value in values):
        msg = "Expert occupancy must contain non-negative values."
        raise ValueError(msg)

    total = math.fsum(values)

    if total <= 0.0:
        msg = "Expert occupancy must contain at least one positive value."
        raise ValueError(msg)

    return [value / total for value in values]


def _population_standard_deviation(
    values: list[float],
) -> float:
    """Return population standard deviation."""

    if len(values) <= 1:
        return 0.0

    mean = math.fsum(values) / len(values)

    variance = math.fsum((value - mean) ** 2 for value in values) / len(values)

    return math.sqrt(variance)


def _normalized_entropy(
    probabilities: list[float],
) -> float:
    """Return Shannon entropy normalized by the maximum entropy."""

    if not probabilities:
        return 0.0

    if len(probabilities) == 1:
        return 1.0

    entropy = -math.fsum(probability * math.log(probability) for probability in probabilities if probability > 0.0)

    maximum_entropy = math.log(len(probabilities))

    if maximum_entropy <= 0.0:
        return 1.0

    normalized_entropy = entropy / maximum_entropy

    return min(
        max(
            normalized_entropy,
            0.0,
        ),
        1.0,
    )


def _get_window_standard_deviations(
    telemetry: RoutingTelemetry,
) -> list[float]:
    """Return per-expert occupancy SD across inference windows."""

    num_experts = telemetry.num_experts

    if num_experts <= 0:
        return []

    window_loads = telemetry.window_expert_load()

    if len(window_loads) <= 1:
        return [0.0 for _ in range(num_experts)]

    expert_values: list[list[float]] = [[] for _ in range(num_experts)]

    for window_index in sorted(window_loads):
        load = _tensor_to_float_list(window_loads[window_index])

        if len(load) != num_experts:
            msg = "Window expert load size does not match the number of experts."
            raise ValueError(msg)

        for (
            expert_index,
            value,
        ) in enumerate(load):
            expert_values[expert_index].append(value * 100.0)

    return [_population_standard_deviation(values) for values in expert_values]


def create_expert_occupancy_figure(
    telemetry: RoutingTelemetry,
    language_index: int = 0,
    *,
    sort_by_load: bool = False,
) -> go.Figure:
    """Create a compact expert-occupancy lollipop visualization."""

    if telemetry.is_empty:
        msg = "Cannot create an expert occupancy figure from empty routing telemetry."
        raise ValueError(msg)

    occupancy = _normalize_distribution(
        _tensor_to_float_list(
            telemetry.expert_occupancy(),
        )
    )

    num_experts = len(occupancy)

    if num_experts != telemetry.num_experts:
        msg = "Expert occupancy size does not match the number of experts."
        raise ValueError(msg)

    if num_experts <= 0:
        msg = "Expert occupancy visualization requires at least one expert."
        raise ValueError(msg)

    uniform_share = 1.0 / num_experts
    uniform_percent = uniform_share * 100.0

    occupancy_percent = [value * 100.0 for value in occupancy]

    deviation_pp = [value - uniform_percent for value in occupancy_percent]

    relative_deviation_percent = [
        (deviation / uniform_percent * 100.0) if uniform_percent > 0.0 else 0.0 for deviation in deviation_pp
    ]

    window_standard_deviation = _get_window_standard_deviations(telemetry)

    global_standard_deviation = _population_standard_deviation(occupancy_percent)

    maximum_deviation = max(
        (abs(deviation) for deviation in deviation_pp),
        default=0.0,
    )

    coefficient_of_variation = global_standard_deviation / uniform_percent * 100.0 if uniform_percent > 0.0 else 0.0

    occupancy_entropy = _normalized_entropy(occupancy)

    window_count = len(telemetry.window_indices)

    expert_label = get_localized_text(
        "Labels_ROUTING_EXPERT",
        language_index,
    )

    occupancy_title = get_localized_text(
        "Labels_ROUTING_OCCUPANCY",
        language_index,
    )

    occupancy_label = get_localized_text(
        "Labels_ROUTING_OCCUPANCY_SHARE",
        language_index,
    )

    uniform_label = get_localized_text(
        "Labels_ROUTING_UNIFORM_SHORT",
        language_index,
    )

    deviation_label = get_localized_text(
        "Labels_ROUTING_DEVIATION_FROM_UNIFORM",
        language_index,
    )

    relative_deviation_label = get_localized_text(
        "Labels_ROUTING_RELATIVE_DEVIATION",
        language_index,
    )

    temporal_sd_label = get_localized_text(
        "Labels_ROUTING_WINDOW_SD",
        language_index,
    )

    maximum_delta_label = get_localized_text(
        "Labels_ROUTING_MAX_DELTA",
        language_index,
    )

    windows_label = get_localized_text(
        "Labels_ROUTING_WINDOWS_SHORT",
        language_index,
    )

    percentage_points_unit = get_localized_text(
        "Units_PERCENTAGE_POINTS",
        language_index,
    )

    expert_indices = list(range(num_experts))

    if sort_by_load:
        expert_indices.sort(
            key=lambda index: occupancy_percent[index],
            reverse=True,
        )

    expert_names = [f"{expert_label} {expert_index + 1}" for expert_index in expert_indices]

    displayed_occupancy = [occupancy_percent[expert_index] for expert_index in expert_indices]

    displayed_deviation = [deviation_pp[expert_index] for expert_index in expert_indices]

    displayed_relative_deviation = [relative_deviation_percent[expert_index] for expert_index in expert_indices]

    displayed_window_sd = [window_standard_deviation[expert_index] for expert_index in expert_indices]

    colors = [_get_expert_color(expert_index) for expert_index in expert_indices]

    row_positions = [float(index) for index in range(num_experts)]

    whisker_offset = 0.35

    whisker_positions = [position - whisker_offset for position in row_positions]

    hover_text = [
        (
            f"<b>{expert_name}</b><br>"
            f"{occupancy_label}: "
            f"{occupancy_value:.2f}%<br>"
            f"{uniform_label}: "
            f"{uniform_percent:.2f}%<br>"
            f"{deviation_label}: "
            f"{deviation:+.2f} "
            f"{percentage_points_unit}<br>"
            f"{relative_deviation_label}: "
            f"{relative_deviation:+.2f}%<br>"
            f"{temporal_sd_label}: "
            f"{temporal_sd:.2f} "
            f"{percentage_points_unit}"
        )
        for (
            expert_name,
            occupancy_value,
            deviation,
            relative_deviation,
            temporal_sd,
        ) in zip(
            expert_names,
            displayed_occupancy,
            displayed_deviation,
            displayed_relative_deviation,
            displayed_window_sd,
            strict=True,
        )
    ]

    figure = go.Figure()

    for (
        row_position,
        occupancy_value,
        color,
    ) in zip(
        row_positions,
        displayed_occupancy,
        colors,
        strict=True,
    ):
        figure.add_trace(
            go.Scatter(
                x=[
                    uniform_percent,
                    occupancy_value,
                ],
                y=[
                    row_position,
                    row_position,
                ],
                mode="lines",
                line={
                    "color": color,
                    "width": 7,
                },
                opacity=0.42,
                hoverinfo="skip",
                showlegend=False,
            )
        )

    if window_count > 1:
        figure.add_trace(
            go.Scatter(
                x=displayed_occupancy,
                y=whisker_positions,
                mode="markers",
                marker={
                    "size": 1,
                    "color": "rgba(0,0,0,0)",
                },
                error_x={
                    "type": "data",
                    "array": displayed_window_sd,
                    "visible": True,
                    "thickness": 1.5,
                    "width": 4,
                    "color": "#A3A3A3",
                },
                hoverinfo="skip",
                showlegend=False,
            )
        )

    text_positions = [
        ("middle right" if occupancy_value >= uniform_percent else "middle left")
        for occupancy_value in displayed_occupancy
    ]

    figure.add_trace(
        go.Scatter(
            x=displayed_occupancy,
            y=row_positions,
            mode="markers+text",
            marker={
                "color": colors,
                "size": 16,
                "line": {
                    "color": ("rgba(255,255,255,0.70)"),
                    "width": 1.5,
                },
            },
            text=[f"{value:.2f}%" for value in displayed_occupancy],
            textposition=text_positions,
            textfont={
                "size": 12,
            },
            hovertext=hover_text,
            hovertemplate=("%{hovertext}<extra></extra>"),
            showlegend=False,
        )
    )

    figure.add_shape(
        type="line",
        xref="x",
        x0=uniform_percent,
        x1=uniform_percent,
        yref="paper",
        y0=0.0,
        y1=1.0,
        layer="below",
        line={
            "color": "#F59E0B",
            "width": 2.5,
            "dash": "dash",
        },
    )

    figure.add_annotation(
        x=uniform_percent,
        y=1.015,
        xref="x",
        yref="paper",
        text=f"{uniform_label} {uniform_percent:.2f}%",
        showarrow=False,
        xanchor="center",
        yanchor="bottom",
        font={
            "size": 11,
            "color": "#F59E0B",
        },
    )

    maximum_extent = max(
        (
            abs(deviation) + temporal_sd
            for (
                deviation,
                temporal_sd,
            ) in zip(
                displayed_deviation,
                displayed_window_sd,
                strict=True,
            )
        ),
        default=1.0,
    )

    axis_padding = max(
        0.35,
        maximum_extent * 0.45,
    )

    x_min = max(
        0.0,
        (uniform_percent - maximum_extent - axis_padding),
    )

    x_max = min(
        100.0,
        (uniform_percent + maximum_extent + axis_padding),
    )

    title_text = (
        f"{occupancy_title}"
        "<br>"
        "<span style='font-size:12px'>"
        f"Hₙ {occupancy_entropy:.4f}"
        f" · CV {coefficient_of_variation:.2f}%"
        "<br>"
        f"{maximum_delta_label} "
        f"{maximum_deviation:.2f} "
        f"{percentage_points_unit}"
        f" · {windows_label} "
        f"{window_count}"
        "</span>"
    )

    figure.update_layout(
        title={
            "text": title_text,
            "x": 0.5,
            "xanchor": "center",
            "y": 0.955,
            "yanchor": "top",
            "font": {
                "size": 20,
            },
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        hovermode="closest",
        hoverlabel={
            "bgcolor": "rgba(17,24,39,0.96)",
            "bordercolor": "rgba(255,255,255,0.18)",
            "font": {
                "color": "#F9FAFB",
                "size": 12,
            },
            "namelength": -1,
        },
        margin={
            "l": 95,
            "r": 72,
            "t": 126,
            "b": 55,
        },
        height=430,
        uirevision="routing-expert-occupancy",
    )

    figure.update_xaxes(
        title={
            "text": f"{occupancy_label}, %",
            "font": {
                "size": 13,
            },
        },
        range=[
            x_min,
            x_max,
        ],
        ticksuffix="%",
        hoverformat=".2f",
        tickfont={
            "size": 11,
        },
        nticks=6,
        showgrid=True,
        gridcolor="rgba(127,127,127,0.20)",
        gridwidth=1,
        zeroline=False,
        showline=True,
        linewidth=1,
        ticks="outside",
        ticklen=4,
    )

    figure.update_yaxes(
        title=None,
        tickmode="array",
        tickvals=row_positions,
        ticktext=expert_names,
        range=[
            num_experts - 0.55,
            -0.55,
        ],
        tickfont={
            "size": 12,
        },
        showgrid=False,
        zeroline=False,
        showline=False,
        ticks="",
    )

    return figure


def _aggregate_layer_axis_expert_load(
    telemetry: RoutingTelemetry,
) -> list[
    tuple[
        int,
        RoutingAxis,
        list[float],
        int,
    ]
]:
    """Aggregate expert occupancy separately by layer and routing axis."""

    num_experts = telemetry.num_experts

    weighted_loads: dict[
        tuple[
            int,
            RoutingAxis,
        ],
        list[float],
    ] = {}

    assignment_counts: dict[
        tuple[
            int,
            RoutingAxis,
        ],
        int,
    ] = {}

    for observation in telemetry.observations:
        key = (
            observation.layer_index,
            observation.axis,
        )

        if key not in weighted_loads:
            weighted_loads[key] = [0.0 for _ in range(num_experts)]

            assignment_counts[key] = 0

        load = _tensor_to_float_list(observation.expert_load)

        if len(load) != num_experts:
            msg = "Layer-axis expert load size does not match the number of experts."
            raise ValueError(msg)

        weight = observation.assignment_count

        for (
            expert_index,
            value,
        ) in enumerate(load):
            weighted_loads[key][expert_index] += value * weight

        assignment_counts[key] += weight

    axis_order: dict[
        RoutingAxis,
        int,
    ] = {
        "time": 0,
        "frequency": 1,
    }

    results: list[
        tuple[
            int,
            RoutingAxis,
            list[float],
            int,
        ]
    ] = []

    for key in sorted(
        weighted_loads,
        key=lambda item: (
            item[0],
            axis_order[item[1]],
        ),
    ):
        (
            layer_index,
            axis,
        ) = key

        assignment_count = assignment_counts[key]

        if assignment_count <= 0:
            continue

        normalized_load = [value / assignment_count for value in weighted_loads[key]]

        results.append(
            (
                layer_index,
                axis,
                normalized_load,
                assignment_count,
            )
        )

    return results


def _get_heatmap_badge_style(
    deviation_pp: float,
) -> tuple[
    str,
    str,
    str,
]:
    """Return readable annotation styling for a heatmap cell."""

    magnitude = abs(deviation_pp)

    if magnitude >= 1.75:
        return (
            "#F8FAFC",
            "rgba(15,23,42,0.56)",
            "rgba(255,255,255,0.28)",
        )

    if magnitude >= 0.90:
        return (
            "#F8FAFC",
            "rgba(31,41,55,0.48)",
            "rgba(255,255,255,0.24)",
        )

    return (
        "#111827",
        "rgba(255,255,255,0.74)",
        "rgba(15,23,42,0.16)",
    )


def _format_heatmap_delta(
    value: float,
    unit: str,
) -> str:
    """Format deviation from uniform routing."""

    if value > 0.0:
        return f"+{value:.2f} {unit}"

    if value < 0.0:
        return f"-{abs(value):.2f} {unit}"

    return f"0.00 {unit}"


def _get_dynamic_color_limits(
    deviation_matrix: list[list[float]],
) -> tuple[
    int,
    int,
]:
    """Return outward-rounded dynamic color limits."""

    values = [value for row in deviation_matrix for value in row if math.isfinite(value)]

    if not values:
        return (
            -1,
            1,
        )

    minimum = min(values)
    maximum = max(values)

    lower_limit = math.floor(minimum)

    upper_limit = math.ceil(maximum)

    if lower_limit >= 0:
        lower_limit = -1

    if upper_limit <= 0:
        upper_limit = 1

    return (
        lower_limit,
        upper_limit,
    )


def _get_zero_centered_colorscale(
    lower_limit: int,
    upper_limit: int,
) -> list[list[float | str]]:
    """Return a diverging colorscale whose neutral color represents zero."""

    span = upper_limit - lower_limit

    zero_position = 0.5 if span <= 0 else -lower_limit / span

    zero_position = min(
        max(
            zero_position,
            0.0,
        ),
        1.0,
    )

    negative_midpoint = zero_position * 0.5

    positive_midpoint = zero_position + (1.0 - zero_position) * 0.5

    return [
        [
            0.0,
            "#2166AC",
        ],
        [
            negative_midpoint,
            "#92C5DE",
        ],
        [
            zero_position,
            "#F7F7F7",
        ],
        [
            positive_midpoint,
            "#F4A582",
        ],
        [
            1.0,
            "#B2182B",
        ],
    ]


def _get_colorbar_ticks(
    lower_limit: int,
    upper_limit: int,
) -> tuple[
    list[int],
    list[str],
]:
    """Return readable integer ticks for a dynamic colorbar."""

    span = upper_limit - lower_limit

    tick_values = (
        list(
            range(
                lower_limit,
                upper_limit + 1,
            )
        )
        if span <= 8
        else [
            lower_limit,
            0,
            upper_limit,
        ]
    )

    tick_text = [("0" if value == 0 else f"{value:+d}") for value in tick_values]

    return (
        tick_values,
        tick_text,
    )


def create_layer_routing_figure(
    telemetry: RoutingTelemetry,
    language_index: int = 0,
) -> go.Figure:
    """Create a compact layer-and-axis expert routing heatmap."""

    if telemetry.is_empty:
        msg = "Cannot create a layer routing figure from empty routing telemetry."
        raise ValueError(msg)

    rows = _aggregate_layer_axis_expert_load(telemetry)

    if not rows:
        msg = "No layer routing observations are available."
        raise ValueError(msg)

    num_experts = telemetry.num_experts

    if num_experts <= 0:
        msg = "Layer routing visualization requires at least one expert."
        raise ValueError(msg)

    uniform_percent = 100.0 / num_experts

    title = get_localized_text(
        "Labels_ROUTING_LAYERS",
        language_index,
    )

    subtitle = get_localized_text(
        "Labels_ROUTING_LAYER_AXIS_SUBTITLE",
        language_index,
    )

    expert_label = get_localized_text(
        "Labels_ROUTING_EXPERT",
        language_index,
    )

    block_label = get_localized_text(
        "Labels_ROUTING_LAYER",
        language_index,
    )

    time_axis_label = get_localized_text(
        "Labels_ROUTING_TIME_AXIS",
        language_index,
    )

    frequency_axis_label = get_localized_text(
        "Labels_ROUTING_FREQUENCY_AXIS",
        language_index,
    )

    occupancy_label = get_localized_text(
        "Labels_ROUTING_OCCUPANCY_SHARE",
        language_index,
    )

    deviation_label = get_localized_text(
        "Labels_ROUTING_DEVIATION_FROM_UNIFORM",
        language_index,
    )

    assignments_label = get_localized_text(
        "Labels_ROUTING_ASSIGNMENTS",
        language_index,
    )

    percentage_points_unit = get_localized_text(
        "Units_PERCENTAGE_POINTS",
        language_index,
    )

    expert_names = [
        f"{expert_label} {index}"
        for index in range(
            1,
            num_experts + 1,
        )
    ]

    row_names: list[str] = []
    occupancy_matrix: list[list[float]] = []
    deviation_matrix: list[list[float]] = []
    hover_text_matrix: list[list[str]] = []

    for (
        layer_index,
        axis,
        load,
        assignment_count,
    ) in rows:
        axis_label = time_axis_label if axis == "time" else frequency_axis_label

        block_name = f"{block_label} {layer_index + 1}"

        row_names.append(f"{block_name} · {axis_label}")

        occupancy_row = [value * 100.0 for value in load]

        deviation_row = [value - uniform_percent for value in occupancy_row]

        occupancy_matrix.append(occupancy_row)

        deviation_matrix.append(deviation_row)

        hover_text_matrix.append(
            [
                (
                    f"<b>{block_name}</b><br>"
                    f"{axis_label}<br>"
                    f"{expert_label}: "
                    f"{expert_names[expert_index]}<br>"
                    f"{deviation_label}: "
                    f"{deviation_value:+.2f} "
                    f"{percentage_points_unit}<br>"
                    f"{occupancy_label}: "
                    f"{occupancy_value:.2f}%<br>"
                    f"{assignments_label}: "
                    f"{assignment_count:,}"
                )
                for (
                    expert_index,
                    (
                        occupancy_value,
                        deviation_value,
                    ),
                ) in enumerate(
                    zip(
                        occupancy_row,
                        deviation_row,
                        strict=True,
                    )
                )
            ]
        )

    (
        lower_color_limit,
        upper_color_limit,
    ) = _get_dynamic_color_limits(deviation_matrix)

    colorscale = _get_zero_centered_colorscale(
        lower_color_limit,
        upper_color_limit,
    )

    (
        colorbar_tick_values,
        colorbar_tick_text,
    ) = _get_colorbar_ticks(
        lower_color_limit,
        upper_color_limit,
    )

    title_text = f"{title}<br><span style='font-size:11px'>{subtitle} ({uniform_percent:.2f}%)</span>"

    figure = go.Figure()

    figure.add_trace(
        go.Heatmap(
            x=expert_names,
            y=row_names,
            z=deviation_matrix,
            text=hover_text_matrix,
            colorscale=colorscale,
            zmin=lower_color_limit,
            zmax=upper_color_limit,
            zhoverformat=".2f",
            xgap=5,
            ygap=5,
            hovertemplate=("%{text}<extra></extra>"),
            colorbar={
                "title": {
                    "text": (f"Δ, {percentage_points_unit}"),
                    "side": "top",
                },
                "orientation": "h",
                "thickness": 12,
                "len": 0.80,
                "x": 0.5,
                "xanchor": "center",
                "y": -0.10,
                "yanchor": "top",
                "ticks": "outside",
                "ticklen": 4,
                "tickmode": "array",
                "tickvals": (colorbar_tick_values),
                "ticktext": (colorbar_tick_text),
                "outlinewidth": 0,
            },
            showscale=True,
        )
    )

    for (
        row_index,
        row_name,
    ) in enumerate(row_names):
        for (
            expert_index,
            expert_name,
        ) in enumerate(expert_names):
            occupancy_value = occupancy_matrix[row_index][expert_index]

            deviation_value = deviation_matrix[row_index][expert_index]

            (
                font_color,
                background_color,
                border_color,
            ) = _get_heatmap_badge_style(deviation_value)

            delta_text = _format_heatmap_delta(
                deviation_value,
                percentage_points_unit,
            )

            figure.add_annotation(
                x=expert_name,
                y=row_name,
                xref="x",
                yref="y",
                text=(f"<b>{delta_text}</b><br><span style='font-size:10px'>({occupancy_value:.2f}%)</span>"),
                showarrow=False,
                align="center",
                font={
                    "size": 11,
                    "color": font_color,
                },
                bgcolor=background_color,
                bordercolor=border_color,
                borderwidth=1,
                borderpad=4,
            )

    row_count = len(row_names)

    figure_height = max(
        410,
        150 + row_count * 72,
    )

    figure.update_layout(
        title={
            "text": title_text,
            "x": 0.5,
            "xanchor": "center",
            "y": 0.94,
            "yanchor": "top",
            "font": {
                "size": 19,
            },
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={
            "l": 125,
            "r": 20,
            "t": 56,
            "b": 52,
        },
        height=figure_height,
        hovermode="closest",
        hoverlabel={
            "bgcolor": "rgba(17,24,39,0.96)",
            "bordercolor": "rgba(255,255,255,0.18)",
            "font": {
                "color": "#F9FAFB",
                "size": 12,
            },
            "namelength": -1,
        },
        uirevision="routing-layer-axis",
    )

    figure.update_xaxes(
        title=None,
        tickfont={
            "size": 11,
        },
        side="bottom",
        showgrid=False,
        zeroline=False,
        showline=False,
        ticks="",
        fixedrange=False,
    )

    figure.update_yaxes(
        title=None,
        domain=[
            0.0,
            1.0,
        ],
        autorange="reversed",
        tickfont={
            "size": 11,
        },
        showgrid=False,
        zeroline=False,
        showline=False,
        ticks="",
        fixedrange=False,
    )

    return figure


def _resolve_frequency_sample_rate(
    sample_rate: int | None,
) -> int:
    """Resolve the sample rate used for the frequency-axis mapping."""

    if sample_rate is not None:
        if sample_rate <= 0:
            msg = "Sample rate must be greater than zero."
            raise ValueError(msg)

        return sample_rate

    configured_sample_rate = get_config_int(
        "AudioDecoding_TARGET_SAMPLE_RATE",
        16000,
    )

    if configured_sample_rate > 0:
        return configured_sample_rate

    return 16000


def _get_frequency_axis_khz(
    num_positions: int,
    sample_rate: int,
) -> list[float]:
    """Return encoded frequency-position center frequencies in kHz."""

    if num_positions <= 0:
        return []

    if sample_rate <= 0:
        msg = "Sample rate must be greater than zero."
        raise ValueError(msg)

    if num_positions == 1:
        return [0.0]

    nyquist_khz = sample_rate / 2000.0

    denominator = num_positions - 1

    return [(nyquist_khz * position / denominator) for position in range(num_positions)]


def _get_frequency_ticks(
    nyquist_khz: float,
) -> list[float]:
    """Return compact major ticks for a frequency axis in kHz."""

    if nyquist_khz <= 0.0:
        return [0.0]

    step = 1.0 if nyquist_khz <= 4.0 else (2.0 if nyquist_khz <= 10.0 else 5.0)

    tick_count = math.floor(nyquist_khz / step)

    ticks = [step * index for index in range(tick_count + 1)]

    if not math.isclose(
        ticks[-1],
        nyquist_khz,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        ticks.append(nyquist_khz)

    return ticks


def _get_percentile(
    values: list[float],
    percentile: float,
) -> float:
    """Return a linearly interpolated percentile."""

    if not values:
        return 0.0

    if not 0.0 <= percentile <= 100.0:
        msg = "Percentile must be in the range [0, 100]."
        raise ValueError(msg)

    sorted_values = sorted(values)

    if len(sorted_values) == 1:
        return sorted_values[0]

    position = percentile / 100.0 * (len(sorted_values) - 1)

    lower_index = math.floor(position)

    upper_index = math.ceil(position)

    if lower_index == upper_index:
        return sorted_values[lower_index]

    interpolation_weight = position - lower_index

    return sorted_values[lower_index] + (sorted_values[upper_index] - sorted_values[lower_index]) * interpolation_weight


def _get_robust_dynamic_color_limits(
    deviation_matrix: list[list[float]],
    *,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> tuple[
    int,
    int,
]:
    """Return robust asymmetric color limits from deviation percentiles."""

    if not (0.0 <= lower_percentile <= 100.0):
        msg = "Lower percentile must be in the range [0, 100]."
        raise ValueError(msg)

    if not (0.0 <= upper_percentile <= 100.0):
        msg = "Upper percentile must be in the range [0, 100]."
        raise ValueError(msg)

    if lower_percentile > upper_percentile:
        msg = "Lower percentile must not exceed upper percentile."
        raise ValueError(msg)

    values = [value for row in deviation_matrix for value in row if math.isfinite(value)]

    if not values:
        return (
            -1,
            1,
        )

    lower_value = _get_percentile(
        values,
        lower_percentile,
    )

    upper_value = _get_percentile(
        values,
        upper_percentile,
    )

    lower_limit = math.floor(lower_value)

    upper_limit = math.ceil(upper_value)

    if lower_limit >= 0:
        lower_limit = -1

    if upper_limit <= 0:
        upper_limit = 1

    return (
        lower_limit,
        upper_limit,
    )


def create_frequency_routing_figure(
    telemetry: RoutingTelemetry,
    language_index: int = 0,
    *,
    sample_rate: int | None = None,
) -> go.Figure:
    """Create an expert-load-by-frequency heatmap."""

    if telemetry.is_empty:
        msg = "Cannot create a frequency routing figure from empty routing telemetry."
        raise ValueError(msg)

    frequency_load = telemetry.frequency_expert_load()

    if frequency_load is None:
        msg = "No frequency-routing observations are available."
        raise ValueError(msg)

    if frequency_load.ndim != 2:
        msg = "Frequency expert load must have shape [frequency_positions, experts]."
        raise ValueError(msg)

    num_positions = frequency_load.shape[0]

    num_experts = frequency_load.shape[1]

    if num_positions <= 0:
        msg = "Frequency routing visualization requires at least one position."
        raise ValueError(msg)

    if num_experts != telemetry.num_experts:
        msg = "Frequency expert load size does not match the number of experts."
        raise ValueError(msg)

    if num_experts <= 0:
        msg = "Frequency routing visualization requires at least one expert."
        raise ValueError(msg)

    resolved_sample_rate = _resolve_frequency_sample_rate(sample_rate)

    frequency_positions_khz = _get_frequency_axis_khz(
        num_positions=num_positions,
        sample_rate=(resolved_sample_rate),
    )

    load_by_position = (
        frequency_load.detach()
        .to(
            device="cpu",
        )
        .tolist()
    )

    uniform_percent = 100.0 / num_experts

    occupancy_matrix = [
        [(float(load_by_position[position_index][expert_index]) * 100.0) for position_index in range(num_positions)]
        for expert_index in range(num_experts)
    ]

    deviation_matrix = [
        [(occupancy_value - uniform_percent) for occupancy_value in expert_row] for expert_row in occupancy_matrix
    ]

    (
        lower_color_limit,
        upper_color_limit,
    ) = _get_robust_dynamic_color_limits(
        deviation_matrix,
        lower_percentile=1.0,
        upper_percentile=99.0,
    )

    colorscale = _get_zero_centered_colorscale(
        lower_color_limit,
        upper_color_limit,
    )

    (
        colorbar_tick_values,
        colorbar_tick_text,
    ) = _get_colorbar_ticks(
        lower_color_limit,
        upper_color_limit,
    )

    title = get_localized_text(
        "Labels_ROUTING_FREQUENCY",
        language_index,
    )

    subtitle = get_localized_text(
        "Labels_ROUTING_FREQUENCY_SUBTITLE",
        language_index,
    )

    expert_label = get_localized_text(
        "Labels_ROUTING_EXPERT",
        language_index,
    )

    occupancy_label = get_localized_text(
        "Labels_ROUTING_OCCUPANCY_SHARE",
        language_index,
    )

    deviation_label = get_localized_text(
        "Labels_ROUTING_DEVIATION_FROM_UNIFORM",
        language_index,
    )

    frequency_label = get_localized_text(
        "Labels_ROUTING_FREQUENCY_CENTER",
        language_index,
    )

    position_label = get_localized_text(
        "Labels_ROUTING_FREQUENCY_POSITION",
        language_index,
    )

    position_count_label = get_localized_text(
        "Labels_ROUTING_FREQUENCY_POSITIONS",
        language_index,
    )

    percentage_points_unit = get_localized_text(
        "Units_PERCENTAGE_POINTS",
        language_index,
    )

    expert_names = [
        (f"{expert_label} {expert_index}")
        for expert_index in range(
            1,
            num_experts + 1,
        )
    ]

    hover_text_matrix = [
        [
            (
                f"<b>{expert_names[expert_index]}</b><br>"
                f"{frequency_label}: "
                f"{frequency_positions_khz[position_index]:.2f}<br>"
                f"{position_label}: "
                f"{position_index}<br>"
                f"{deviation_label}: "
                f"{deviation_matrix[expert_index][position_index]:+.2f} "
                f"{percentage_points_unit}<br>"
                f"{occupancy_label}: "
                f"{occupancy_matrix[expert_index][position_index]:.2f}%"
            )
            for position_index in range(num_positions)
        ]
        for expert_index in range(num_experts)
    ]

    nyquist_khz = resolved_sample_rate / 2000.0

    frequency_ticks = _get_frequency_ticks(nyquist_khz)

    title_text = (
        f"{title}"
        "<br>"
        "<span style='font-size:11px'>"
        f"{subtitle} "
        f"({uniform_percent:.2f}%)"
        f" · {num_positions} "
        f"{position_count_label}"
        " · P1-P99 Δ"
        "</span>"
    )

    figure = go.Figure()

    figure.add_trace(
        go.Heatmap(
            x=frequency_positions_khz,
            y=expert_names,
            z=deviation_matrix,
            text=hover_text_matrix,
            colorscale=colorscale,
            zmin=lower_color_limit,
            zmax=upper_color_limit,
            zmid=0.0,
            zhoverformat=".2f",
            zsmooth=False,
            connectgaps=False,
            xgap=0,
            ygap=3,
            hovertemplate=("%{text}<extra></extra>"),
            colorbar={
                "title": {
                    "text": (f"Δ, {percentage_points_unit}"),
                    "side": "right",
                },
                "orientation": "h",
                "thickness": 11,
                "len": 0.62,
                "x": 0.5,
                "xanchor": "center",
                "y": -0.34,
                "yanchor": "top",
                "ticks": "outside",
                "ticklen": 4,
                "tickmode": "array",
                "tickvals": (colorbar_tick_values),
                "ticktext": (colorbar_tick_text),
                "outlinewidth": 0,
            },
            showscale=True,
        )
    )

    figure.update_layout(
        title={
            "text": title_text,
            "x": 0.5,
            "xanchor": "center",
            "y": 0.94,
            "yanchor": "top",
            "font": {
                "size": 19,
            },
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={
            "l": 82,
            "r": 28,
            "t": 78,
            "b": 112,
        },
        height=410,
        hovermode="closest",
        hoverlabel={
            "bgcolor": "rgba(17,24,39,0.96)",
            "bordercolor": "rgba(255,255,255,0.18)",
            "font": {
                "color": "#F9FAFB",
                "size": 12,
            },
            "namelength": -1,
        },
        uirevision="routing-frequency",
    )

    figure.update_xaxes(
        title={
            "text": frequency_label,
            "font": {
                "size": 13,
            },
        },
        range=[
            0.0,
            nyquist_khz,
        ],
        hoverformat=".2f",
        tickmode="array",
        tickvals=frequency_ticks,
        ticktext=[f"{value:g}" for value in frequency_ticks],
        tickfont={
            "size": 11,
        },
        showgrid=False,
        zeroline=False,
        showline=True,
        linewidth=1,
        ticks="outside",
        ticklen=4,
        fixedrange=False,
    )

    figure.update_yaxes(
        title=None,
        categoryorder="array",
        categoryarray=expert_names,
        autorange="reversed",
        tickfont={
            "size": 11,
        },
        showgrid=False,
        zeroline=False,
        showline=False,
        ticks="",
        fixedrange=False,
    )

    return figure


def _get_window_sample_ranges(
    telemetry: RoutingTelemetry,
) -> dict[
    int,
    tuple[
        int,
        int,
    ],
]:
    """Return validated sample ranges for all inference windows."""

    sample_ranges: dict[
        int,
        tuple[
            int,
            int,
        ],
    ] = {}

    for observation in telemetry.observations:
        sample_range = (
            observation.start_sample,
            observation.end_sample,
        )

        if sample_range[0] < 0 or sample_range[1] <= sample_range[0]:
            msg = f"Routing telemetry contains an invalid inference-window sample range: {sample_range}."
            raise ValueError(msg)

        existing_range = sample_ranges.get(observation.window_index)

        if existing_range is None:
            sample_ranges[observation.window_index] = sample_range
            continue

        if existing_range != sample_range:
            msg = (
                "Routing telemetry contains "
                "inconsistent sample ranges "
                f"for window "
                f"{observation.window_index}: "
                f"{existing_range} "
                f"!= {sample_range}."
            )
            raise ValueError(msg)

    return dict(sorted(sample_ranges.items()))


def _get_time_load_y_range(
    occupancy_matrix: list[list[float]],
    uniform_percent: float,
) -> tuple[
    float,
    float,
]:
    """Return a compact dynamic y-axis range for temporal expert load."""

    values = [value for expert_values in occupancy_matrix for value in expert_values if math.isfinite(value)]

    values.append(uniform_percent)

    minimum = min(values)
    maximum = max(values)

    span = maximum - minimum

    padding = max(
        0.35,
        span * 0.18,
    )

    lower = max(
        0.0,
        minimum - padding,
    )

    upper = min(
        100.0,
        maximum + padding,
    )

    if upper - lower < 1.0:
        midpoint = (upper + lower) * 0.5

        lower = max(
            0.0,
            midpoint - 0.5,
        )

        upper = min(
            100.0,
            midpoint + 0.5,
        )

    if upper <= lower:
        upper = min(
            100.0,
            lower + 1.0,
        )

    return (
        lower,
        upper,
    )


def create_load_over_time_figure(
    telemetry: RoutingTelemetry,
    language_index: int = 0,
    *,
    sample_rate: int | None = None,
) -> go.Figure:
    """Create expert-load trajectories across overlapping inference windows."""

    if telemetry.is_empty:
        msg = "Cannot create a temporal routing figure from empty routing telemetry."
        raise ValueError(msg)

    num_experts = telemetry.num_experts

    if num_experts <= 0:
        msg = "Temporal routing visualization requires at least one expert."
        raise ValueError(msg)

    resolved_sample_rate = _resolve_frequency_sample_rate(sample_rate)

    window_loads = telemetry.window_expert_load()

    if not window_loads:
        msg = "No inference-window expert loads are available."
        raise ValueError(msg)

    sample_ranges = _get_window_sample_ranges(telemetry)

    if not sample_ranges:
        msg = "No routing-window sample ranges are available."
        raise ValueError(msg)

    window_indices = list(sample_ranges)

    if set(window_indices) != set(window_loads):
        msg = "Routing window ranges and expert loads contain different window indices."
        raise ValueError(msg)

    window_start_seconds = [(sample_ranges[window_index][0] / resolved_sample_rate) for window_index in window_indices]

    window_end_seconds = [(sample_ranges[window_index][1] / resolved_sample_rate) for window_index in window_indices]

    center_times = [
        (start_seconds + end_seconds) * 0.5
        for (
            start_seconds,
            end_seconds,
        ) in zip(
            window_start_seconds,
            window_end_seconds,
            strict=True,
        )
    ]

    start_seconds = min(window_start_seconds)

    end_seconds = max(window_end_seconds)

    occupancy_matrix: list[list[float]] = [[] for _ in range(num_experts)]

    for window_index in window_indices:
        load = _tensor_to_float_list(window_loads[window_index])

        if len(load) != num_experts:
            msg = f"Window expert load size does not match the number of experts: {len(load)} != {num_experts}."
            raise ValueError(msg)

        normalized_load = _normalize_distribution(load)

        for (
            expert_index,
            occupancy,
        ) in enumerate(normalized_load):
            occupancy_matrix[expert_index].append(occupancy * 100.0)

    uniform_percent = 100.0 / num_experts

    window_count = len(window_indices)

    title = get_localized_text(
        "Labels_ROUTING_LOAD",
        language_index,
    )

    subtitle = get_localized_text(
        "Labels_ROUTING_TIME_SUBTITLE",
        language_index,
    )

    expert_label = get_localized_text(
        "Labels_ROUTING_EXPERT",
        language_index,
    )

    occupancy_label = get_localized_text(
        "Labels_ROUTING_OCCUPANCY_SHARE",
        language_index,
    )

    deviation_label = get_localized_text(
        "Labels_ROUTING_DEVIATION_FROM_UNIFORM",
        language_index,
    )

    uniform_label = get_localized_text(
        "Labels_ROUTING_UNIFORM_SHORT",
        language_index,
    )

    window_label = get_localized_text(
        "Labels_ROUTING_WINDOW",
        language_index,
    )

    windows_label = get_localized_text(
        "Labels_ROUTING_WINDOWS_SHORT",
        language_index,
    )

    window_range_label = get_localized_text(
        "Labels_ROUTING_WINDOW_RANGE",
        language_index,
    )

    window_center_label = get_localized_text(
        "Labels_ROUTING_WINDOW_CENTER",
        language_index,
    )

    time_label = get_localized_text(
        "Labels_ROUTING_TIME_AXIS",
        language_index,
    )

    seconds_unit = get_localized_text(
        "Units_SECONDS",
        language_index,
    )

    percentage_points_unit = get_localized_text(
        "Units_PERCENTAGE_POINTS",
        language_index,
    )

    title_text = (
        f"{title}"
        "<br>"
        "<span style='font-size:11px'>"
        f"{subtitle}"
        f" · {windows_label} "
        f"{window_count}"
        f" · {uniform_label} "
        f"{uniform_percent:.2f}%"
        "</span>"
    )

    figure = go.Figure()

    for expert_index in range(num_experts):
        expert_name = f"{expert_label} {expert_index + 1}"

        occupancies = occupancy_matrix[expert_index]

        deviations = [(occupancy - uniform_percent) for occupancy in occupancies]

        hover_text = [
            (
                f"<b>{expert_name}</b><br>"
                f"{window_label}: "
                f"{window_index}<br>"
                f"{window_range_label}: "
                f"{start_time:.2f}-{end_time:.2f} "
                f"{seconds_unit}<br>"
                f"{window_center_label}: "
                f"{center_time:.2f} "
                f"{seconds_unit}<br>"
                f"{occupancy_label}: "
                f"{occupancy:.2f}%<br>"
                f"{deviation_label}: "
                f"{deviation:+.2f} "
                f"{percentage_points_unit}"
            )
            for (
                window_index,
                start_time,
                end_time,
                center_time,
                occupancy,
                deviation,
            ) in zip(
                window_indices,
                window_start_seconds,
                window_end_seconds,
                center_times,
                occupancies,
                deviations,
                strict=True,
            )
        ]

        color = _get_expert_color(expert_index)

        figure.add_trace(
            go.Scatter(
                x=center_times,
                y=occupancies,
                hovertext=hover_text,
                mode="lines+markers",
                name=expert_name,
                line={
                    "color": color,
                    "width": 1.8,
                },
                marker={
                    "color": color,
                    "size": 5,
                },
                hovertemplate=("%{hovertext}<extra></extra>"),
            )
        )

    figure.add_shape(
        type="line",
        xref="x",
        x0=start_seconds,
        x1=end_seconds,
        yref="y",
        y0=uniform_percent,
        y1=uniform_percent,
        layer="below",
        line={
            "color": "#F59E0B",
            "width": 2.2,
            "dash": "dash",
        },
    )

    figure.add_annotation(
        x=end_seconds,
        y=uniform_percent,
        xref="x",
        yref="y",
        text=f"{uniform_label} {uniform_percent:.2f}%",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        yshift=5,
        font={
            "size": 10,
            "color": "#F59E0B",
        },
    )

    (
        y_min,
        y_max,
    ) = _get_time_load_y_range(
        occupancy_matrix=occupancy_matrix,
        uniform_percent=uniform_percent,
    )

    figure.update_layout(
        title={
            "text": title_text,
            "x": 0.5,
            "xanchor": "center",
            "y": 0.955,
            "yanchor": "top",
            "font": {
                "size": 20,
            },
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={
            "l": 72,
            "r": 28,
            "t": 78,
            "b": 112,
        },
        height=430,
        hovermode="closest",
        hoverlabel={
            "bgcolor": "rgba(17,24,39,0.96)",
            "bordercolor": "rgba(255,255,255,0.18)",
            "font": {
                "color": "#F9FAFB",
                "size": 12,
            },
            "namelength": -1,
        },
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": -0.30,
            "yanchor": "top",
            "bgcolor": "rgba(0,0,0,0)",
            "traceorder": "normal",
            "font": {
                "size": 11,
            },
        },
        uirevision="routing-load-over-time",
    )

    figure.update_xaxes(
        title={
            "text": (f"{time_label}, {seconds_unit}"),
            "font": {
                "size": 13,
            },
        },
        range=[
            start_seconds,
            end_seconds,
        ],
        hoverformat=".2f",
        tickfont={
            "size": 11,
        },
        showgrid=True,
        gridcolor="rgba(127,127,127,0.16)",
        gridwidth=1,
        zeroline=False,
        showline=True,
        linewidth=1,
        ticks="outside",
        ticklen=4,
        fixedrange=False,
    )

    figure.update_yaxes(
        title={
            "text": (f"{occupancy_label}, %"),
            "font": {
                "size": 13,
            },
        },
        range=[
            y_min,
            y_max,
        ],
        ticksuffix="%",
        hoverformat=".2f",
        tickfont={
            "size": 11,
        },
        showgrid=True,
        gridcolor="rgba(127,127,127,0.16)",
        gridwidth=1,
        zeroline=False,
        showline=True,
        linewidth=1,
        ticks="outside",
        ticklen=4,
        fixedrange=False,
    )

    return figure
