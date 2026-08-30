"""
File: visualization_info_specs.py

Author: Ksenia Zolina, Dmitry Ryumin, Denis Ivanko, and Alexey Karpov

Description: Scientific content specifications for WAVES visualizations.

License: MIT License
"""

from dataclasses import dataclass
from enum import StrEnum

from waves.ui.mathml import MathML


class VisualizationInfoKey(StrEnum):
    """Supported WAVES visualization information views."""

    SPECTROGRAM = "spectrogram"
    EXPERT_OCCUPANCY = "expert_occupancy"
    LAYER_ROUTING = "layer_routing"
    FREQUENCY_ROUTING = "frequency_routing"
    LOAD_OVER_TIME = "load_over_time"


@dataclass(frozen=True, slots=True)
class VisualizationInfoSpec:
    """Static specification for one scientific visualization."""

    title_key: str
    subtitle_key: str
    shows_key: str
    read_rows: tuple[tuple[str, str], ...]
    formulas: tuple[str, ...]
    notation: tuple[tuple[str, str], ...]
    interpretation_keys: tuple[str, ...]
    note_key: str
    show_settings: bool = False


M = MathML


def _var(
    name: str,
    *,
    normal: bool = False,
) -> str:
    """Create one MathML identifier."""

    return M.identifier(
        name,
        normal=normal,
    )


def _idx(
    base: str,
    *indices: str,
    text_index: bool = False,
) -> str:
    """Create one indexed MathML variable."""

    return M.indexed(
        base,
        *indices,
        text_index=text_index,
    )


def _multi_idx(
    base: str,
    *indices: str,
) -> str:
    """Create one variable with a compound subscript."""

    return M.subscript(
        _var(base),
        M.comma_row(*(_var(index) for index in indices)),
    )


def _delta(
    *indices: str,
    text_index: bool = False,
) -> str:
    """Create delta with an optional subscript."""

    return M.delta(
        *indices,
        text_index=text_index,
    )


def _multi_delta(
    *indices: str,
) -> str:
    """Create delta with a compound subscript."""

    return M.subscript(
        _var("&#x0394;"),
        M.comma_row(*(_var(index) for index in indices)),
    )


def _sum(
    index: str,
) -> str:
    """Create a summation indexed by one variable."""

    return M.indexed_sum(index)


def _sum_in_window() -> str:
    """Create a summation over observations in a window."""

    return M.summation(
        M.row(
            _var("o"),
            M.operator("&#x2208;"),
            _var("w"),
        )
    )


def _uniform() -> str:
    """Create uniform expert occupancy."""

    return M.fraction(
        M.number(1),
        _var("E"),
    )


def _equation(
    lhs: str,
    rhs: str,
) -> str:
    """Create one display-style equation."""

    return M.display(
        M.equation(
            lhs,
            rhs,
        )
    )


def _notation(
    body: str,
    key: str,
) -> tuple[str, str]:
    """Create one notation glossary entry."""

    return (
        M.inline(body),
        key,
    )


def _spectrogram_formulas() -> tuple[str, ...]:
    """Create spectrogram formulas."""

    power = M.call(
        _var("P"),
        _var("m"),
        _var("t"),
    )

    log10 = M.subscript(
        _var(
            "log",
            normal=True,
        ),
        M.number(10),
    )

    return (
        _equation(
            power,
            M.call(
                M.text("MelPower"),
                _var("x"),
            ),
        ),
        _equation(
            M.call(
                _idx(
                    "S",
                    "dB",
                    text_index=True,
                ),
                _var("m"),
                _var("t"),
            ),
            M.row(
                M.number(10),
                log10,
                M.parentheses(
                    M.fraction(
                        power,
                        _idx(
                            "P",
                            "ref",
                            text_index=True,
                        ),
                    )
                ),
            ),
        ),
        _equation(
            M.call(
                _delta(
                    "dB",
                    text_index=True,
                ),
                _var("m"),
                _var("t"),
            ),
            M.row(
                M.number(10),
                log10,
                M.parentheses(
                    M.fraction(
                        M.call(
                            _idx(
                                "P",
                                "enh",
                                text_index=True,
                            ),
                            _var("m"),
                            _var("t"),
                        ),
                        M.call(
                            _idx(
                                "P",
                                "in",
                                text_index=True,
                            ),
                            _var("m"),
                            _var("t"),
                        ),
                    )
                ),
            ),
        ),
    )


def _expert_occupancy_formulas() -> tuple[str, ...]:
    """Create global expert-occupancy formulas."""

    p_e = _idx(
        "p",
        "e",
    )

    u = _var("u")

    return (
        _equation(
            p_e,
            M.fraction(
                _idx(
                    "A",
                    "e",
                ),
                M.row(
                    _sum("j"),
                    _idx(
                        "A",
                        "j",
                    ),
                ),
            ),
        ),
        _equation(
            u,
            _uniform(),
        ),
        _equation(
            _delta("e"),
            M.row(
                M.number(100),
                M.parentheses(
                    M.row(
                        p_e,
                        M.operator("&#x2212;"),
                        u,
                    )
                ),
                M.text("pp"),
            ),
        ),
        _equation(
            _idx(
                "H",
                "n",
            ),
            M.fraction(
                M.row(
                    M.operator("&#x2212;"),
                    _sum("e"),
                    p_e,
                    _var(
                        "ln",
                        normal=True,
                    ),
                    p_e,
                ),
                M.row(
                    _var(
                        "ln",
                        normal=True,
                    ),
                    _var("E"),
                ),
            ),
        ),
        _equation(
            _var(
                "CV",
                normal=True,
            ),
            M.fraction(
                M.row(
                    M.number(100),
                    _var("&#x03C3;"),
                ),
                M.subscript(
                    u,
                    M.operator("%"),
                ),
            ),
        ),
    )


def _layer_routing_formulas() -> tuple[str, ...]:
    """Create layer-routing formulas."""

    layer = "&#x2113;"

    occupancy = _multi_idx(
        "p",
        layer,
        "a",
        "e",
    )

    u = _var("u")

    return (
        _equation(
            occupancy,
            M.fraction(
                M.row(
                    _sum("o"),
                    _idx(
                        "n",
                        "o",
                    ),
                    _multi_idx(
                        "L",
                        "o",
                        "e",
                    ),
                ),
                M.row(
                    _sum("o"),
                    _idx(
                        "n",
                        "o",
                    ),
                ),
            ),
        ),
        _equation(
            u,
            _uniform(),
        ),
        _equation(
            _multi_delta(
                layer,
                "a",
                "e",
            ),
            M.row(
                M.number(100),
                M.parentheses(
                    M.row(
                        occupancy,
                        M.operator("&#x2212;"),
                        u,
                    )
                ),
                M.text("pp"),
            ),
        ),
    )


def _frequency_routing_formulas() -> tuple[str, ...]:
    """Create frequency-routing formulas."""

    occupancy = _multi_idx(
        "p",
        "j",
        "e",
    )

    return (
        _equation(
            _idx(
                "w",
                "o",
            ),
            M.row(
                _idx(
                    "S",
                    "o",
                ),
                _idx(
                    "K",
                    "o",
                ),
            ),
        ),
        _equation(
            occupancy,
            M.fraction(
                M.row(
                    _sum("o"),
                    _idx(
                        "w",
                        "o",
                    ),
                    _multi_idx(
                        "L",
                        "o",
                        "j",
                        "e",
                    ),
                ),
                M.row(
                    _sum("o"),
                    _idx(
                        "w",
                        "o",
                    ),
                ),
            ),
        ),
        _equation(
            _multi_delta(
                "j",
                "e",
            ),
            M.row(
                M.number(100),
                M.parentheses(
                    M.row(
                        occupancy,
                        M.operator("&#x2212;"),
                        _uniform(),
                    )
                ),
                M.text("pp"),
            ),
        ),
        _equation(
            _idx(
                "f",
                "j",
            ),
            M.row(
                M.fraction(
                    _idx(
                        "f",
                        "s",
                    ),
                    M.number(2),
                ),
                M.operator("&#x00B7;"),
                M.fraction(
                    _var("j"),
                    M.row(
                        _var("J"),
                        M.operator("&#x2212;"),
                        M.number(1),
                    ),
                ),
            ),
        ),
    )


def _load_over_time_formulas() -> tuple[str, ...]:
    """Create temporal expert-load formulas."""

    weighted_load = _multi_idx(
        "a",
        "w",
        "e",
    )

    occupancy = _multi_idx(
        "p",
        "w",
        "e",
    )

    return (
        _equation(
            weighted_load,
            M.fraction(
                M.row(
                    _sum_in_window(),
                    _idx(
                        "n",
                        "o",
                    ),
                    _multi_idx(
                        "L",
                        "o",
                        "e",
                    ),
                ),
                M.row(
                    _sum_in_window(),
                    _idx(
                        "n",
                        "o",
                    ),
                ),
            ),
        ),
        _equation(
            occupancy,
            M.fraction(
                weighted_load,
                M.row(
                    _sum("j"),
                    _multi_idx(
                        "a",
                        "w",
                        "j",
                    ),
                ),
            ),
        ),
        _equation(
            _idx(
                "t",
                "w",
            ),
            M.fraction(
                M.row(
                    _idx(
                        "s",
                        "w",
                    ),
                    M.operator("+"),
                    _idx(
                        "e",
                        "w",
                    ),
                ),
                M.row(
                    M.number(2),
                    _idx(
                        "f",
                        "s",
                    ),
                ),
            ),
        ),
        _equation(
            _multi_delta(
                "w",
                "e",
            ),
            M.row(
                M.number(100),
                M.parentheses(
                    M.row(
                        occupancy,
                        M.operator("&#x2212;"),
                        _uniform(),
                    )
                ),
                M.text("pp"),
            ),
        ),
    )


SPECTROGRAM_FORMULAS = _spectrogram_formulas()

EXPERT_OCCUPANCY_FORMULAS = _expert_occupancy_formulas()

LAYER_ROUTING_FORMULAS = _layer_routing_formulas()

FREQUENCY_ROUTING_FORMULAS = _frequency_routing_formulas()

LOAD_OVER_TIME_FORMULAS = _load_over_time_formulas()


SPECS: dict[
    VisualizationInfoKey,
    VisualizationInfoSpec,
] = {
    VisualizationInfoKey.SPECTROGRAM: VisualizationInfoSpec(
        title_key="Labels_SPECTROGRAM",
        subtitle_key=("VisualizationInfoSpectrogram_SUBTITLE"),
        shows_key=("VisualizationInfoSpectrogram_SHOWS"),
        read_rows=(
            (
                "VisualizationInfoLabels_X_AXIS",
                "VisualizationInfoSpectrogram_READ_X",
            ),
            (
                "VisualizationInfoLabels_Y_AXIS",
                "VisualizationInfoSpectrogram_READ_Y",
            ),
            (
                "VisualizationInfoSpectrogram_READ_SIGNAL_COLOR_LABEL",
                "VisualizationInfoSpectrogram_READ_SIGNAL_COLOR",
            ),
            (
                "VisualizationInfoSpectrogram_READ_CHANGE_COLOR_LABEL",
                "VisualizationInfoSpectrogram_READ_CHANGE_COLOR",
            ),
        ),
        formulas=SPECTROGRAM_FORMULAS,
        notation=(
            _notation(
                _var("x"),
                "VisualizationInfoSpectrogram_NOTATION_X",
            ),
            _notation(
                _var("t"),
                "VisualizationInfoSpectrogram_NOTATION_T",
            ),
            _notation(
                _var("m"),
                "VisualizationInfoSpectrogram_NOTATION_M",
            ),
            _notation(
                _var("P"),
                "VisualizationInfoSpectrogram_NOTATION_P",
            ),
            _notation(
                _idx(
                    "P",
                    "ref",
                    text_index=True,
                ),
                "VisualizationInfoSpectrogram_NOTATION_P_REF",
            ),
            _notation(
                _idx(
                    "P",
                    "in",
                    text_index=True,
                ),
                "VisualizationInfoSpectrogram_NOTATION_P_IN",
            ),
            _notation(
                _idx(
                    "P",
                    "enh",
                    text_index=True,
                ),
                "VisualizationInfoSpectrogram_NOTATION_P_ENH",
            ),
            _notation(
                _idx(
                    "S",
                    "dB",
                    text_index=True,
                ),
                "VisualizationInfoSpectrogram_NOTATION_S_DB",
            ),
            _notation(
                _delta(
                    "dB",
                    text_index=True,
                ),
                "VisualizationInfoSpectrogram_NOTATION_DELTA_DB",
            ),
            _notation(
                M.text("MelPower"),
                "VisualizationInfoSpectrogram_NOTATION_MEL_POWER",
            ),
            _notation(
                M.subscript(
                    _var(
                        "log",
                        normal=True,
                    ),
                    M.number(10),
                ),
                "VisualizationInfoSpectrogram_NOTATION_LOG10",
            ),
        ),
        interpretation_keys=(
            "VisualizationInfoSpectrogram_INTERPRETATION_1",
            "VisualizationInfoSpectrogram_INTERPRETATION_2",
        ),
        note_key=("VisualizationInfoSpectrogram_NOTE"),
        show_settings=True,
    ),
    VisualizationInfoKey.EXPERT_OCCUPANCY: VisualizationInfoSpec(
        title_key="Labels_ROUTING_OCCUPANCY",
        subtitle_key=("VisualizationInfoExpertOccupancy_SUBTITLE"),
        shows_key=("VisualizationInfoExpertOccupancy_SHOWS"),
        read_rows=(
            (
                "Labels_ROUTING_EXPERT",
                "VisualizationInfoExpertOccupancy_READ_EXPERT",
            ),
            (
                "VisualizationInfoExpertOccupancy_READ_MARKER_LABEL",
                "VisualizationInfoExpertOccupancy_READ_MARKER",
            ),
            (
                "VisualizationInfoLabels_DASHED_LINE",
                "VisualizationInfoExpertOccupancy_READ_DASHED",
            ),
            (
                "VisualizationInfoExpertOccupancy_READ_WHISKER_LABEL",
                "VisualizationInfoExpertOccupancy_READ_WHISKER",
            ),
        ),
        formulas=EXPERT_OCCUPANCY_FORMULAS,
        notation=(
            _notation(
                _var("E"),
                "VisualizationInfoExpertOccupancy_NOTATION_E_COUNT",
            ),
            _notation(
                _var("e"),
                "VisualizationInfoExpertOccupancy_NOTATION_E_INDEX",
            ),
            _notation(
                _var("j"),
                "VisualizationInfoExpertOccupancy_NOTATION_J_INDEX",
            ),
            _notation(
                _idx(
                    "A",
                    "e",
                ),
                "VisualizationInfoExpertOccupancy_NOTATION_A_E",
            ),
            _notation(
                _idx(
                    "p",
                    "e",
                ),
                "VisualizationInfoExpertOccupancy_NOTATION_P_E",
            ),
            _notation(
                _var("u"),
                "VisualizationInfoExpertOccupancy_NOTATION_U",
            ),
            _notation(
                M.subscript(
                    _var("u"),
                    M.operator("%"),
                ),
                "VisualizationInfoExpertOccupancy_NOTATION_U_PERCENT",
            ),
            _notation(
                _delta("e"),
                "VisualizationInfoExpertOccupancy_NOTATION_DELTA_E",
            ),
            _notation(
                _idx(
                    "H",
                    "n",
                ),
                "VisualizationInfoExpertOccupancy_NOTATION_H_N",
            ),
            _notation(
                _var("&#x03C3;"),
                "VisualizationInfoExpertOccupancy_NOTATION_SIGMA",
            ),
            _notation(
                _var(
                    "CV",
                    normal=True,
                ),
                "VisualizationInfoExpertOccupancy_NOTATION_CV",
            ),
            _notation(
                _var(
                    "ln",
                    normal=True,
                ),
                "VisualizationInfoExpertOccupancy_NOTATION_LN",
            ),
            _notation(
                M.text("pp"),
                "VisualizationInfoExpertOccupancy_NOTATION_PP",
            ),
        ),
        interpretation_keys=(
            "VisualizationInfoExpertOccupancy_INTERPRETATION_1",
            "VisualizationInfoExpertOccupancy_INTERPRETATION_2",
        ),
        note_key=("VisualizationInfoExpertOccupancy_NOTE"),
    ),
    VisualizationInfoKey.LAYER_ROUTING: VisualizationInfoSpec(
        title_key="Labels_ROUTING_LAYERS",
        subtitle_key=("VisualizationInfoLayerRouting_SUBTITLE"),
        shows_key=("VisualizationInfoLayerRouting_SHOWS"),
        read_rows=(
            (
                "VisualizationInfoLayerRouting_READ_ROWS_LABEL",
                "VisualizationInfoLayerRouting_READ_ROWS",
            ),
            (
                "VisualizationInfoLayerRouting_READ_COLUMNS_LABEL",
                "VisualizationInfoLayerRouting_READ_COLUMNS",
            ),
            (
                "VisualizationInfoLayerRouting_READ_CELL_LABEL",
                "VisualizationInfoLayerRouting_READ_CELL",
            ),
            (
                "VisualizationInfoLabels_COLOR",
                "VisualizationInfoLayerRouting_READ_COLOR",
            ),
        ),
        formulas=LAYER_ROUTING_FORMULAS,
        notation=(
            _notation(
                _var("E"),
                "VisualizationInfoLayerRouting_NOTATION_E_COUNT",
            ),
            _notation(
                _var("&#x2113;"),
                "VisualizationInfoLayerRouting_NOTATION_LAYER",
            ),
            _notation(
                _var("a"),
                "VisualizationInfoLayerRouting_NOTATION_AXIS",
            ),
            _notation(
                _var("e"),
                "VisualizationInfoLayerRouting_NOTATION_E_INDEX",
            ),
            _notation(
                _var("o"),
                "VisualizationInfoLayerRouting_NOTATION_OBSERVATION",
            ),
            _notation(
                _idx(
                    "n",
                    "o",
                ),
                "VisualizationInfoLayerRouting_NOTATION_N_O",
            ),
            _notation(
                _multi_idx(
                    "L",
                    "o",
                    "e",
                ),
                "VisualizationInfoLayerRouting_NOTATION_L_O_E",
            ),
            _notation(
                _multi_idx(
                    "p",
                    "&#x2113;",
                    "a",
                    "e",
                ),
                "VisualizationInfoLayerRouting_NOTATION_P",
            ),
            _notation(
                _var("u"),
                "VisualizationInfoLayerRouting_NOTATION_U",
            ),
            _notation(
                _multi_delta(
                    "&#x2113;",
                    "a",
                    "e",
                ),
                "VisualizationInfoLayerRouting_NOTATION_DELTA",
            ),
            _notation(
                M.text("pp"),
                "VisualizationInfoLayerRouting_NOTATION_PP",
            ),
        ),
        interpretation_keys=(
            "VisualizationInfoLayerRouting_INTERPRETATION_1",
            "VisualizationInfoLayerRouting_INTERPRETATION_2",
        ),
        note_key=("VisualizationInfoLayerRouting_NOTE"),
    ),
    VisualizationInfoKey.FREQUENCY_ROUTING: VisualizationInfoSpec(
        title_key="Labels_ROUTING_FREQUENCY",
        subtitle_key=("VisualizationInfoFrequencyRouting_SUBTITLE"),
        shows_key=("VisualizationInfoFrequencyRouting_SHOWS"),
        read_rows=(
            (
                "VisualizationInfoLabels_X_AXIS",
                "VisualizationInfoFrequencyRouting_READ_X",
            ),
            (
                "VisualizationInfoLabels_Y_AXIS",
                "VisualizationInfoFrequencyRouting_READ_Y",
            ),
            (
                "VisualizationInfoLabels_COLOR",
                "VisualizationInfoFrequencyRouting_READ_COLOR",
            ),
            (
                "VisualizationInfoFrequencyRouting_READ_PERCENTILES_LABEL",
                "VisualizationInfoFrequencyRouting_READ_PERCENTILES",
            ),
        ),
        formulas=FREQUENCY_ROUTING_FORMULAS,
        notation=(
            _notation(
                _var("E"),
                "VisualizationInfoFrequencyRouting_NOTATION_E_COUNT",
            ),
            _notation(
                _var("o"),
                "VisualizationInfoFrequencyRouting_NOTATION_OBSERVATION",
            ),
            _notation(
                _var("j"),
                "VisualizationInfoFrequencyRouting_NOTATION_J",
            ),
            _notation(
                _var("J"),
                "VisualizationInfoFrequencyRouting_NOTATION_J_COUNT",
            ),
            _notation(
                _var("e"),
                "VisualizationInfoFrequencyRouting_NOTATION_E_INDEX",
            ),
            _notation(
                _idx(
                    "w",
                    "o",
                ),
                "VisualizationInfoFrequencyRouting_NOTATION_W_O",
            ),
            _notation(
                _idx(
                    "S",
                    "o",
                ),
                "VisualizationInfoFrequencyRouting_NOTATION_S_O",
            ),
            _notation(
                _idx(
                    "K",
                    "o",
                ),
                "VisualizationInfoFrequencyRouting_NOTATION_K_O",
            ),
            _notation(
                _multi_idx(
                    "L",
                    "o",
                    "j",
                    "e",
                ),
                "VisualizationInfoFrequencyRouting_NOTATION_L",
            ),
            _notation(
                _multi_idx(
                    "p",
                    "j",
                    "e",
                ),
                "VisualizationInfoFrequencyRouting_NOTATION_P",
            ),
            _notation(
                _idx(
                    "f",
                    "s",
                ),
                "VisualizationInfoFrequencyRouting_NOTATION_SAMPLE_RATE",
            ),
            _notation(
                _idx(
                    "f",
                    "j",
                ),
                "VisualizationInfoFrequencyRouting_NOTATION_F_J",
            ),
            _notation(
                _multi_delta(
                    "j",
                    "e",
                ),
                "VisualizationInfoFrequencyRouting_NOTATION_DELTA",
            ),
            _notation(
                M.text("pp"),
                "VisualizationInfoFrequencyRouting_NOTATION_PP",
            ),
        ),
        interpretation_keys=(
            "VisualizationInfoFrequencyRouting_INTERPRETATION_1",
            "VisualizationInfoFrequencyRouting_INTERPRETATION_2",
        ),
        note_key=("VisualizationInfoFrequencyRouting_NOTE"),
    ),
    VisualizationInfoKey.LOAD_OVER_TIME: VisualizationInfoSpec(
        title_key="Labels_ROUTING_LOAD",
        subtitle_key=("VisualizationInfoLoadOverTime_SUBTITLE"),
        shows_key=("VisualizationInfoLoadOverTime_SHOWS"),
        read_rows=(
            (
                "VisualizationInfoLabels_X_AXIS",
                "VisualizationInfoLoadOverTime_READ_X",
            ),
            (
                "VisualizationInfoLabels_Y_AXIS",
                "VisualizationInfoLoadOverTime_READ_Y",
            ),
            (
                "VisualizationInfoLoadOverTime_READ_LINE_LABEL",
                "VisualizationInfoLoadOverTime_READ_LINE",
            ),
            (
                "VisualizationInfoLabels_DASHED_LINE",
                "VisualizationInfoLoadOverTime_READ_DASHED",
            ),
        ),
        formulas=LOAD_OVER_TIME_FORMULAS,
        notation=(
            _notation(
                _var("E"),
                "VisualizationInfoLoadOverTime_NOTATION_E_COUNT",
            ),
            _notation(
                _var("w"),
                "VisualizationInfoLoadOverTime_NOTATION_WINDOW",
            ),
            _notation(
                _var("o"),
                "VisualizationInfoLoadOverTime_NOTATION_OBSERVATION",
            ),
            _notation(
                _var("e"),
                "VisualizationInfoLoadOverTime_NOTATION_E_INDEX",
            ),
            _notation(
                _var("j"),
                "VisualizationInfoLoadOverTime_NOTATION_J",
            ),
            _notation(
                _idx(
                    "n",
                    "o",
                ),
                "VisualizationInfoLoadOverTime_NOTATION_N_O",
            ),
            _notation(
                _multi_idx(
                    "L",
                    "o",
                    "e",
                ),
                "VisualizationInfoLoadOverTime_NOTATION_L",
            ),
            _notation(
                _multi_idx(
                    "a",
                    "w",
                    "e",
                ),
                "VisualizationInfoLoadOverTime_NOTATION_A",
            ),
            _notation(
                _multi_idx(
                    "p",
                    "w",
                    "e",
                ),
                "VisualizationInfoLoadOverTime_NOTATION_P",
            ),
            _notation(
                _idx(
                    "s",
                    "w",
                ),
                "VisualizationInfoLoadOverTime_NOTATION_START",
            ),
            _notation(
                _idx(
                    "e",
                    "w",
                ),
                "VisualizationInfoLoadOverTime_NOTATION_END",
            ),
            _notation(
                _idx(
                    "f",
                    "s",
                ),
                "VisualizationInfoLoadOverTime_NOTATION_SAMPLE_RATE",
            ),
            _notation(
                _idx(
                    "t",
                    "w",
                ),
                "VisualizationInfoLoadOverTime_NOTATION_TIME",
            ),
            _notation(
                _multi_delta(
                    "w",
                    "e",
                ),
                "VisualizationInfoLoadOverTime_NOTATION_DELTA",
            ),
            _notation(
                M.text("pp"),
                "VisualizationInfoLoadOverTime_NOTATION_PP",
            ),
        ),
        interpretation_keys=(
            "VisualizationInfoLoadOverTime_INTERPRETATION_1",
            "VisualizationInfoLoadOverTime_INTERPRETATION_2",
        ),
        note_key=("VisualizationInfoLoadOverTime_NOTE"),
    ),
}
