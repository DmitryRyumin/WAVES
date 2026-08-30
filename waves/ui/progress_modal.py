"""
File: progress_modal.py

Author: Ksenia Zolina, Dmitry Ryumin, Denis Ivanko, and Alexey Karpov

Description: Processing progress modal rendering for the WAVES application.

License: MIT License
"""

from dataclasses import dataclass
from html import escape

from waves.inference.progress import (
    EnhancementProgressEvent,
    EnhancementStage,
    EnhancementStageTiming,
    get_enhancement_stage_definitions,
)
from waves.localization import get_localized_text


@dataclass(frozen=True, slots=True)
class ProcessingSummary:
    """Completed processing information retained by the application."""

    event: EnhancementProgressEvent
    device: str


_STAGE_LABEL_FIELDS: dict[
    EnhancementStage,
    str,
] = {
    EnhancementStage.VALIDATION: "Labels_PROGRESS_STAGE_VALIDATION",
    EnhancementStage.AUDIO_DECODING: "Labels_PROGRESS_STAGE_AUDIO_DECODING",
    EnhancementStage.PREPROCESSING: "Labels_PROGRESS_STAGE_PREPROCESSING",
    EnhancementStage.MODEL_LOADING: "Labels_PROGRESS_STAGE_MODEL_LOADING",
    EnhancementStage.ENHANCEMENT: "Labels_PROGRESS_STAGE_ENHANCEMENT",
    EnhancementStage.POSTPROCESSING: "Labels_PROGRESS_STAGE_POSTPROCESSING",
    EnhancementStage.WAV_ENCODING: "Labels_PROGRESS_STAGE_WAV_ENCODING",
    EnhancementStage.SPECTROGRAM_RENDERING: ("Labels_PROGRESS_STAGE_SPECTROGRAM_RENDERING"),
    EnhancementStage.ROUTING_VISUALIZATION: ("Labels_PROGRESS_STAGE_ROUTING_VISUALIZATION"),
}


_STAGE_DETAIL_FIELDS: dict[
    EnhancementStage,
    str,
] = {
    EnhancementStage.VALIDATION: "Texts_PROGRESS_DETAIL_VALIDATION",
    EnhancementStage.AUDIO_DECODING: "Texts_PROGRESS_DETAIL_AUDIO_DECODING",
    EnhancementStage.PREPROCESSING: "Texts_PROGRESS_DETAIL_PREPROCESSING",
    EnhancementStage.MODEL_LOADING: "Texts_PROGRESS_DETAIL_MODEL_LOADING",
    EnhancementStage.ENHANCEMENT: "Texts_PROGRESS_DETAIL_ENHANCEMENT",
    EnhancementStage.POSTPROCESSING: "Texts_PROGRESS_DETAIL_POSTPROCESSING",
    EnhancementStage.WAV_ENCODING: "Texts_PROGRESS_DETAIL_WAV_ENCODING",
    EnhancementStage.SPECTROGRAM_RENDERING: ("Texts_PROGRESS_DETAIL_SPECTROGRAM_RENDERING"),
    EnhancementStage.ROUTING_VISUALIZATION: ("Texts_PROGRESS_DETAIL_ROUTING_VISUALIZATION"),
}


def get_progress_stage_label(
    stage: EnhancementStage,
    language_index: int,
) -> str:
    """Return the localized display label for one processing stage."""

    return get_localized_text(
        _STAGE_LABEL_FIELDS[stage],
        language_index,
    )


def get_progress_stage_detail(
    stage: EnhancementStage,
    language_index: int,
) -> str:
    """Return the localized running detail for one processing stage."""

    return get_localized_text(
        _STAGE_DETAIL_FIELDS[stage],
        language_index,
    )


def _format_elapsed_seconds(
    seconds: float,
    language_index: int,
) -> str:
    """Format elapsed processing time."""

    unit = get_localized_text(
        "Units_SECONDS",
        language_index,
    )

    return f"{seconds:.2f} {unit}"


def create_processing_time_button_label(
    summary: ProcessingSummary,
    language_index: int,
) -> str:
    """Create the localized processing-time button label."""

    processing_time_label = get_localized_text(
        "Labels_PROCESSING_TIME",
        language_index,
    )

    device_label = get_localized_text(
        "Labels_PROGRESS_DEVICE",
        language_index,
    )

    return (
        f"{processing_time_label}: "
        f"{_format_elapsed_seconds(summary.event.elapsed_seconds, language_index)}"
        f" · {device_label}: {summary.device}"
    )


def _create_metric_card(
    label: str,
    value: str,
) -> str:
    """Create one progress summary metric card."""

    return (
        '<div class="processing-modal-metric">'
        '<div class="processing-modal-metric-label">'
        f"{escape(label)}"
        "</div>"
        '<div class="processing-modal-metric-value">'
        f"{escape(value)}"
        "</div>"
        "</div>"
    )


def _create_stage_detail(
    *,
    stage: EnhancementStage,
    timing: EnhancementStageTiming | None,
    event: EnhancementProgressEvent,
    language_index: int,
    completed: bool,
    total_stage_seconds: float,
) -> str:
    """Create stage detail text for the running or completed modal."""

    if timing is not None:
        elapsed = _format_elapsed_seconds(
            timing.elapsed_seconds,
            language_index,
        )

        if completed:
            percentage = timing.elapsed_seconds / total_stage_seconds * 100.0 if total_stage_seconds > 0.0 else 0.0

            return f"{elapsed} · {percentage:.1f}%"

        return elapsed

    stage_active = stage == event.stage and stage not in event.completed_stages

    if stage_active:
        return get_progress_stage_detail(
            stage,
            language_index,
        )

    return ""


def _create_stage_badge(
    *,
    timing: EnhancementStageTiming | None,
    language_index: int,
) -> str:
    """Create an optional stage-state badge."""

    if timing is None or not timing.cached:
        return ""

    cached_label = get_localized_text(
        "Labels_PROGRESS_CACHED",
        language_index,
    )

    return f'<span class="processing-modal-stage-badge">{escape(cached_label)}</span>'


def _create_stage_item(
    *,
    index: int,
    stage: EnhancementStage,
    event: EnhancementProgressEvent,
    timings: dict[
        EnhancementStage,
        EnhancementStageTiming,
    ],
    language_index: int,
    completed: bool,
    total_stage_seconds: float,
) -> str:
    """Create one stage row for the processing modal."""

    stage_completed = stage in event.completed_stages

    stage_active = stage == event.stage and not stage_completed

    if stage_completed:
        state_name = "completed"
    elif stage_active:
        state_name = "active"
    else:
        state_name = "pending"

    state_class = f"is-{state_name}"

    layout_class = "is-single-line" if state_name == "pending" else "is-two-line"

    stage_label = get_progress_stage_label(
        stage,
        language_index,
    )

    timing = timings.get(stage)

    detail = _create_stage_detail(
        stage=stage,
        timing=timing,
        event=event,
        language_index=language_index,
        completed=completed,
        total_stage_seconds=total_stage_seconds,
    )

    badge_html = _create_stage_badge(
        timing=timing,
        language_index=language_index,
    )

    detail_html = f'<div class="processing-modal-stage-detail">{escape(detail) if detail else "&nbsp;"}</div>'

    marker = "✓" if stage_completed else str(index)

    return (
        f'<div class="processing-modal-stage '
        f'{state_class} {layout_class}" '
        f'data-stage-key="{escape(stage.value)}" '
        f'data-stage-state="{state_name}">'
        '<div class="processing-modal-stage-marker">'
        f"{escape(marker)}"
        "</div>"
        '<div class="processing-modal-stage-text">'
        '<div class="processing-modal-stage-header">'
        '<div class="processing-modal-stage-name">'
        f"{escape(stage_label)}"
        "</div>"
        f"{badge_html}"
        "</div>"
        f"{detail_html}"
        "</div>"
        "</div>"
    )


def _create_progress_bar_text(
    event: EnhancementProgressEvent,
    language_index: int,
    *,
    completed: bool,
) -> str:
    """Create text displayed inside the processing progress bar."""

    if completed:
        return get_localized_text(
            "Labels_PROGRESS_COMPLETED",
            language_index,
        )

    if event.stage != EnhancementStage.ENHANCEMENT:
        return ""

    if event.completed_units is None or event.total_units is None or event.total_units <= 0:
        return ""

    window_label = get_localized_text(
        "Labels_PROGRESS_WINDOW",
        language_index,
    )

    return f"{window_label} {event.completed_units} / {event.total_units}"


def create_processing_modal_html(
    event: EnhancementProgressEvent,
    language_index: int,
    *,
    completed: bool = False,
    device: str | None = None,
    auto_close: bool = False,
) -> str:
    """Create running or completed processing modal HTML."""

    if completed:
        title = get_localized_text(
            "Texts_PROGRESS_COMPLETED_TITLE",
            language_index,
        )

        subtitle = get_localized_text(
            "Texts_PROGRESS_COMPLETED_SUBTITLE",
            language_index,
        )
    else:
        title = get_localized_text(
            "Texts_PROGRESS_RUNNING_TITLE",
            language_index,
        )

        subtitle = get_localized_text(
            "Texts_PROGRESS_RUNNING_SUBTITLE",
            language_index,
        )

    elapsed_label = get_localized_text(
        "Labels_PROGRESS_ELAPSED",
        language_index,
    )

    progress_label = get_localized_text(
        "Labels_PROGRESS_PROGRESS",
        language_index,
    )

    progress_percentage = min(
        100.0,
        max(
            0.0,
            event.overall_progress * 100.0,
        ),
    )

    if completed:
        first_metric_label = elapsed_label

        first_metric_value = _format_elapsed_seconds(
            event.elapsed_seconds,
            language_index,
        )

        second_metric_label = get_localized_text(
            "Labels_PROGRESS_DEVICE",
            language_index,
        )

        second_metric_value = device if device else "—"
    else:
        first_metric_label = get_localized_text(
            "Labels_PROGRESS_CURRENT_STAGE",
            language_index,
        )

        first_metric_value = get_progress_stage_label(
            event.stage,
            language_index,
        )

        second_metric_label = elapsed_label

        second_metric_value = _format_elapsed_seconds(
            event.elapsed_seconds,
            language_index,
        )

    metrics_html = "".join(
        (
            _create_metric_card(
                first_metric_label,
                first_metric_value,
            ),
            _create_metric_card(
                second_metric_label,
                second_metric_value,
            ),
            _create_metric_card(
                progress_label,
                f"{progress_percentage:.0f}%",
            ),
        )
    )

    timings = {timing.stage: timing for timing in event.stage_timings}

    total_stage_seconds = sum(timing.elapsed_seconds for timing in event.stage_timings)

    stages_html = "".join(
        _create_stage_item(
            index=index,
            stage=definition.stage,
            event=event,
            timings=timings,
            language_index=language_index,
            completed=completed,
            total_stage_seconds=total_stage_seconds,
        )
        for index, definition in enumerate(
            get_enhancement_stage_definitions(),
            start=1,
        )
    )

    body_classes = [
        "processing-modal-body",
        ("is-completed" if completed else "is-running"),
    ]

    if auto_close:
        body_classes.append("is-auto-closing")

    body_class = " ".join(body_classes)

    processing_state = "completed" if completed else "running"

    progress_bar_text = _create_progress_bar_text(
        event,
        language_index,
        completed=completed,
    )

    progress_bar_text_html = ""

    if progress_bar_text:
        progress_bar_text_html = f'<div class="processing-modal-progress-label">{escape(progress_bar_text)}</div>'

    footer = get_localized_text(
        "Texts_PROGRESS_FOOTER",
        language_index,
    )

    aria_busy = "false" if completed else "true"

    return f"""
<div
    class="{body_class}"
    data-processing-state="{processing_state}"
    aria-busy="{aria_busy}"
>
    <div class="processing-modal-heading">
        <div class="processing-modal-title">{escape(title)}</div>
        <div class="processing-modal-subtitle">{escape(subtitle)}</div>
    </div>

    <div class="processing-modal-metrics">
        {metrics_html}
    </div>

    <div class="processing-modal-progress-row">
        <div class="processing-modal-progress-track">
            <div
                class="processing-modal-progress-fill"
                style="width: {progress_percentage:.3f}%"
            >
                <div class="processing-modal-progress-bar"></div>
            </div>
            {progress_bar_text_html}
        </div>
    </div>

    <div class="processing-modal-stages">
        {stages_html}
    </div>

    <div class="processing-modal-footer">
        {escape(footer)}
    </div>
</div>
""".strip()


def create_completed_processing_modal_html(
    summary: ProcessingSummary,
    language_index: int,
    *,
    auto_close: bool = False,
) -> str:
    """Create completed processing modal HTML."""

    return create_processing_modal_html(
        event=summary.event,
        language_index=language_index,
        completed=True,
        device=summary.device,
        auto_close=auto_close,
    )
