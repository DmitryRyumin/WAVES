"""
File: registry.py

Author: Ksenia Zolina, Dmitry Ryumin, Denis Ivanko, and Alexey Karpov

Description: Model discovery utilities for the WAVES application.

License: MIT License
"""

from dataclasses import dataclass
from pathlib import Path

from waves.config import (
    PROJECT_ROOT,
    get_config_str,
)


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """One discovered WAVES model and its runtime configuration."""

    name: str
    weights_path: Path
    config_path: Path


def get_model_root() -> Path:
    """Return the configured production model root directory."""

    configured_root = Path(
        get_config_str(
            "Model_ROOT",
            "models",
        )
    ).expanduser()

    if configured_root.is_absolute():
        return configured_root.resolve()

    return (PROJECT_ROOT / configured_root).resolve()


def discover_models() -> list[ModelInfo]:
    """Discover production models with colocated weights and configuration."""

    root = get_model_root()

    if not root.is_dir():
        return []

    weights_filename = get_config_str(
        "Model_WEIGHTS_FILENAME",
        "model.safetensors",
    )

    config_filename = get_config_str(
        "Model_CONFIG_FILENAME",
        "config.yaml",
    )

    models: list[ModelInfo] = []

    for weights_path in sorted(root.rglob(weights_filename)):
        if not weights_path.is_file():
            continue

        config_path = weights_path.parent / config_filename

        if not config_path.is_file():
            continue

        relative_parent = weights_path.parent.relative_to(root)

        models.append(
            ModelInfo(
                name=relative_parent.as_posix(),
                weights_path=weights_path.resolve(),
                config_path=config_path.resolve(),
            )
        )

    return models


def get_model_info(
    model_name: str,
) -> ModelInfo:
    """Return metadata for one model registered under the model root."""

    normalized_name = model_name.strip()

    if not normalized_name:
        msg = "Model name must not be empty."
        raise ValueError(msg)

    root = get_model_root()
    model_directory = (root / normalized_name).resolve()

    try:
        relative_directory = model_directory.relative_to(root)
    except ValueError as error:
        msg = f"Model must be located inside the configured model root: {root}"
        raise ValueError(msg) from error

    if not model_directory.is_dir():
        msg = f"Model directory was not found: {model_directory}"
        raise FileNotFoundError(msg)

    weights_filename = get_config_str(
        "Model_WEIGHTS_FILENAME",
        "model.safetensors",
    )

    config_filename = get_config_str(
        "Model_CONFIG_FILENAME",
        "config.yaml",
    )

    weights_path = model_directory / weights_filename
    config_path = model_directory / config_filename

    if not weights_path.is_file():
        msg = f"Model weights were not found: {weights_path}"
        raise FileNotFoundError(msg)

    if not config_path.is_file():
        msg = f"Model config was not found: {config_path}"
        raise FileNotFoundError(msg)

    return ModelInfo(
        name=relative_directory.as_posix(),
        weights_path=weights_path.resolve(),
        config_path=config_path.resolve(),
    )
