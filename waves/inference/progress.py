"""
File: progress.py

Author: Ksenia Zolina, Dmitry Ryumin, Denis Ivanko, and Alexey Karpov

Description: Progress tracking utilities for the WAVES enhancement pipeline.

License: MIT License
"""

from dataclasses import dataclass
from enum import StrEnum
from time import perf_counter

from waves.config import get_config_float_mapping


class EnhancementStage(StrEnum):
    """One top-level WAVES speech-enhancement processing stage."""

    VALIDATION = "validation"
    AUDIO_DECODING = "audio_decoding"
    PREPROCESSING = "preprocessing"
    MODEL_LOADING = "model_loading"
    ENHANCEMENT = "enhancement"
    POSTPROCESSING = "postprocessing"
    WAV_ENCODING = "wav_encoding"
    SPECTROGRAM_RENDERING = "spectrogram_rendering"
    ROUTING_VISUALIZATION = "routing_visualization"


@dataclass(frozen=True, slots=True)
class EnhancementStageDefinition:
    """Configured WAVES processing stage and its progress weight."""

    stage: EnhancementStage
    weight: float


@dataclass(frozen=True, slots=True)
class EnhancementStageTiming:
    """Measured runtime statistics for one completed stage."""

    stage: EnhancementStage
    elapsed_seconds: float
    cached: bool | None = None


@dataclass(frozen=True, slots=True)
class EnhancementProgressEvent:
    """Immutable progress snapshot emitted by the enhancement pipeline."""

    stage: EnhancementStage

    stage_progress: float
    overall_progress: float
    elapsed_seconds: float

    completed_stages: tuple[
        EnhancementStage,
        ...,
    ]

    stage_timings: tuple[
        EnhancementStageTiming,
        ...,
    ]

    completed_units: int | None = None
    total_units: int | None = None
    cached: bool | None = None


def get_enhancement_stage_definitions() -> tuple[
    EnhancementStageDefinition,
    ...,
]:
    """Load and validate ordered progress stages from config.toml."""

    configured_weights = get_config_float_mapping("ProgressStages")

    if not configured_weights:
        msg = "ProgressStages must contain at least one processing stage."
        raise ValueError(msg)

    definitions: list[EnhancementStageDefinition] = []

    configured_stages: set[EnhancementStage] = set()

    for stage_name, weight in configured_weights.items():
        try:
            stage = EnhancementStage(stage_name)
        except ValueError as error:
            msg = f"Unsupported progress stage in config.toml: {stage_name!r}."
            raise ValueError(msg) from error

        if stage in configured_stages:
            msg = f"Duplicate progress stage in config.toml: {stage.value!r}."
            raise ValueError(msg)

        if weight <= 0.0:
            msg = f"Progress-stage weight must be greater than zero: {stage.value}={weight}."
            raise ValueError(msg)

        configured_stages.add(stage)

        definitions.append(
            EnhancementStageDefinition(
                stage=stage,
                weight=weight,
            )
        )

    missing_stages: list[EnhancementStage] = [stage for stage in EnhancementStage if stage not in configured_stages]

    if missing_stages:
        missing = ", ".join(sorted(stage.value for stage in missing_stages))

        msg = f"ProgressStages is missing required stages: {missing}."
        raise ValueError(msg)

    total_weight = sum(definition.weight for definition in definitions)

    if total_weight <= 0.0:
        msg = "The total progress-stage weight must be greater than zero."
        raise ValueError(msg)

    return tuple(
        EnhancementStageDefinition(
            stage=definition.stage,
            weight=(definition.weight / total_weight),
        )
        for definition in definitions
    )


class EnhancementProgressTracker:
    """Track weighted progress and stage timings for one enhancement run."""

    def __init__(
        self,
        stage_definitions: (
            tuple[
                EnhancementStageDefinition,
                ...,
            ]
            | None
        ) = None,
    ) -> None:
        definitions = stage_definitions if stage_definitions is not None else get_enhancement_stage_definitions()

        if not definitions:
            msg = "At least one progress-stage definition is required."
            raise ValueError(msg)

        self._definitions = definitions

        self._stage_order = tuple(definition.stage for definition in definitions)

        self._weights = {definition.stage: (definition.weight) for definition in definitions}

        self._started_at = perf_counter()

        self._active_stage: EnhancementStage | None = None

        self._active_stage_started_at: float | None = None

        self._active_stage_progress = 0.0

        self._stage_timings: dict[
            EnhancementStage,
            EnhancementStageTiming,
        ] = {}

    @property
    def stage_order(
        self,
    ) -> tuple[
        EnhancementStage,
        ...,
    ]:
        """Return configured processing stages in display order."""

        return self._stage_order

    @property
    def active_stage(
        self,
    ) -> EnhancementStage | None:
        """Return the currently active processing stage."""

        return self._active_stage

    @property
    def elapsed_seconds(
        self,
    ) -> float:
        """Return total elapsed runtime."""

        return perf_counter() - self._started_at

    @property
    def completed_stages(
        self,
    ) -> tuple[
        EnhancementStage,
        ...,
    ]:
        """Return completed stages in configured order."""

        return tuple(stage for stage in self._stage_order if stage in self._stage_timings)

    @property
    def stage_timings(
        self,
    ) -> tuple[
        EnhancementStageTiming,
        ...,
    ]:
        """Return measured stage timings in configured order."""

        return tuple(self._stage_timings[stage] for stage in self._stage_order if stage in self._stage_timings)

    @property
    def overall_progress(
        self,
    ) -> float:
        """Return weighted overall progress in the range [0, 1]."""

        completed_progress = sum(self._weights[stage] for stage in self._stage_timings)

        active_progress = 0.0

        if self._active_stage is not None:
            active_progress = self._weights[self._active_stage] * self._active_stage_progress

        return min(
            1.0,
            max(
                0.0,
                completed_progress + active_progress,
            ),
        )

    def _validate_stage(
        self,
        stage: EnhancementStage,
    ) -> None:
        """Validate that a stage belongs to the configured pipeline."""

        if stage not in self._weights:
            msg = f"Processing stage is not configured: {stage.value}."
            raise ValueError(msg)

    def _create_event(
        self,
        *,
        stage: EnhancementStage,
        stage_progress: float,
        completed_units: int | None = None,
        total_units: int | None = None,
        cached: bool | None = None,
    ) -> EnhancementProgressEvent:
        """Create an immutable progress snapshot."""

        return EnhancementProgressEvent(
            stage=stage,
            stage_progress=stage_progress,
            overall_progress=self.overall_progress,
            elapsed_seconds=self.elapsed_seconds,
            completed_stages=(self.completed_stages),
            stage_timings=(self.stage_timings),
            completed_units=completed_units,
            total_units=total_units,
            cached=cached,
        )

    def begin_stage(
        self,
        stage: EnhancementStage,
        *,
        cached: bool | None = None,
    ) -> EnhancementProgressEvent:
        """Start timing one processing stage."""

        self._validate_stage(stage)

        if self._active_stage is not None:
            msg = f"Cannot start processing stage '{stage.value}' while '{self._active_stage.value}' is still active."
            raise RuntimeError(msg)

        if stage in self._stage_timings:
            msg = f"Processing stage has already completed: {stage.value}."
            raise RuntimeError(msg)

        self._active_stage = stage
        self._active_stage_started_at = perf_counter()
        self._active_stage_progress = 0.0

        return self._create_event(
            stage=stage,
            stage_progress=0.0,
            cached=cached,
        )

    def update_stage(
        self,
        stage: EnhancementStage,
        stage_progress: float,
        *,
        completed_units: int | None = None,
        total_units: int | None = None,
        cached: bool | None = None,
    ) -> EnhancementProgressEvent:
        """Update progress inside the currently active stage."""

        self._validate_stage(stage)

        if self._active_stage != stage:
            active_stage = self._active_stage.value if self._active_stage is not None else "none"

            msg = f"Cannot update processing stage '{stage.value}'; active stage is '{active_stage}'."
            raise RuntimeError(msg)

        if not 0.0 <= stage_progress <= 1.0:
            msg = "Stage progress must be in the range [0, 1]."
            raise ValueError(msg)

        if completed_units is not None and completed_units < 0:
            msg = "Completed work units must not be negative."
            raise ValueError(msg)

        if total_units is not None and total_units <= 0:
            msg = "Total work units must be greater than zero."
            raise ValueError(msg)

        if completed_units is not None and total_units is not None and completed_units > total_units:
            msg = "Completed work units cannot exceed total work units."
            raise ValueError(msg)

        self._active_stage_progress = stage_progress

        return self._create_event(
            stage=stage,
            stage_progress=stage_progress,
            completed_units=completed_units,
            total_units=total_units,
            cached=cached,
        )

    def complete_stage(
        self,
        stage: EnhancementStage,
        *,
        completed_units: int | None = None,
        total_units: int | None = None,
        cached: bool | None = None,
    ) -> EnhancementProgressEvent:
        """Complete and record timing for the active processing stage."""

        self._validate_stage(stage)

        if self._active_stage != stage:
            active_stage = self._active_stage.value if self._active_stage is not None else "none"

            msg = f"Cannot complete processing stage '{stage.value}'; active stage is '{active_stage}'."
            raise RuntimeError(msg)

        if self._active_stage_started_at is None:
            msg = "Active processing stage does not have a start time."
            raise RuntimeError(msg)

        elapsed_seconds = perf_counter() - self._active_stage_started_at

        self._active_stage_progress = 1.0

        self._stage_timings[stage] = EnhancementStageTiming(
            stage=stage,
            elapsed_seconds=elapsed_seconds,
            cached=cached,
        )

        self._active_stage = None
        self._active_stage_started_at = None
        self._active_stage_progress = 0.0

        return self._create_event(
            stage=stage,
            stage_progress=1.0,
            completed_units=completed_units,
            total_units=total_units,
            cached=cached,
        )
