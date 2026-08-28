"""
File: validation.py

Author: Ksenia Zolina, Dmitry Ryumin, Denis Ivanko, and Alexey Karpov

Description: Audio validation utilities for the WAVES Gradio application.

License: MIT License
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from waves.audio.metadata import (
    AudioMetadata,
    read_audio_metadata,
)
from waves.config import (
    get_config_float,
    get_config_int,
)

ERROR_SEVERITY: Final = "error"
WARNING_SEVERITY: Final = "warning"


@dataclass(frozen=True, slots=True)
class AudioValidationIssue:
    """One audio validation issue."""

    code: str
    severity: str
    actual: float | int | str | None = None
    expected: float | int | str | None = None


@dataclass(frozen=True, slots=True)
class AudioValidationResult:
    """Audio validation result."""

    metadata: AudioMetadata | None
    issues: list[AudioValidationIssue]

    @property
    def errors(self) -> list[AudioValidationIssue]:
        """Return validation errors."""

        return [issue for issue in self.issues if issue.severity == ERROR_SEVERITY]

    @property
    def warnings(self) -> list[AudioValidationIssue]:
        """Return validation warnings."""

        return [issue for issue in self.issues if issue.severity == WARNING_SEVERITY]

    @property
    def is_valid(self) -> bool:
        """Return whether the audio is valid for speech enhancement."""

        return not self.errors


def create_issue(
    code: str,
    severity: str,
    actual: float | int | str | None = None,
    expected: float | int | str | None = None,
) -> AudioValidationIssue:
    """Create one validation issue."""

    return AudioValidationIssue(
        code=code,
        severity=severity,
        actual=actual,
        expected=expected,
    )


def validate_audio_metadata(
    metadata: AudioMetadata,
) -> AudioValidationResult:
    """Validate audio metadata."""

    issues: list[AudioValidationIssue] = []

    min_duration_seconds = get_config_float(
        "AudioValidation_MIN_DURATION_SECONDS",
        0.1,
    )
    max_duration_seconds = get_config_float(
        "AudioValidation_MAX_DURATION_SECONDS",
        300.0,
    )
    target_sample_rate = get_config_int(
        "AudioDecoding_TARGET_SAMPLE_RATE",
        16000,
    )
    target_num_channels = get_config_int(
        "AudioDecoding_TARGET_NUM_CHANNELS",
        1,
    )

    if metadata.size_bytes <= 0:
        issues.append(
            create_issue(
                "EMPTY_FILE",
                ERROR_SEVERITY,
            )
        )

    if metadata.duration_seconds is None or metadata.duration_seconds <= 0:
        issues.append(
            create_issue(
                "DURATION_UNAVAILABLE",
                ERROR_SEVERITY,
            )
        )
    elif metadata.duration_seconds < min_duration_seconds:
        issues.append(
            create_issue(
                code="TOO_SHORT",
                severity=ERROR_SEVERITY,
                actual=metadata.duration_seconds,
                expected=min_duration_seconds,
            )
        )
    elif metadata.duration_seconds > max_duration_seconds:
        issues.append(
            create_issue(
                code="TOO_LONG",
                severity=ERROR_SEVERITY,
                actual=metadata.duration_seconds,
                expected=max_duration_seconds,
            )
        )

    if metadata.sample_rate is None or metadata.sample_rate <= 0:
        issues.append(
            create_issue(
                "SAMPLE_RATE_UNAVAILABLE",
                ERROR_SEVERITY,
            )
        )
    elif metadata.sample_rate != target_sample_rate:
        issues.append(
            create_issue(
                code="SAMPLE_RATE_WARNING",
                severity=WARNING_SEVERITY,
                actual=metadata.sample_rate,
                expected=target_sample_rate,
            )
        )

    if metadata.num_channels is None or metadata.num_channels <= 0:
        issues.append(
            create_issue(
                "CHANNELS_UNAVAILABLE",
                ERROR_SEVERITY,
            )
        )
    elif metadata.num_channels != target_num_channels:
        issues.append(
            create_issue(
                code="CHANNELS_WARNING",
                severity=WARNING_SEVERITY,
                actual=metadata.num_channels,
                expected=target_num_channels,
            )
        )

    return AudioValidationResult(
        metadata=metadata,
        issues=issues,
    )


def validate_audio_file(
    audio_path: str,
) -> AudioValidationResult:
    """Validate an audio file by path."""

    path = Path(audio_path)

    if not path.is_file():
        return AudioValidationResult(
            metadata=None,
            issues=[
                create_issue(
                    code="FILE_NOT_FOUND",
                    severity=ERROR_SEVERITY,
                    actual=str(path),
                )
            ],
        )

    try:
        metadata = read_audio_metadata(str(path))
    except Exception as error:
        return AudioValidationResult(
            metadata=None,
            issues=[
                create_issue(
                    code="READ_FAILED",
                    severity=ERROR_SEVERITY,
                    actual=str(error),
                )
            ],
        )

    return validate_audio_metadata(metadata)
