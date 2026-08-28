"""
File: config.py

Author: Ksenia Zolina, Dmitry Ryumin, Denis Ivanko, and Alexey Karpov

Description: Model configuration utilities for WAVES.

License: MIT License
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml


@dataclass(frozen=True, slots=True)
class WAVESConfig:
    """Runtime configuration required to construct and run WAVESModel."""

    sampling_rate: int
    n_fft: int
    hop_size: int
    win_size: int

    dense_channel: int
    dense_depth: int
    beta: float
    compress_factor: float
    n_heads: int
    num_tsblocks: int

    moe: dict[str, Any] | None


def get_config_section(
    config: dict[str, Any],
    section_name: str,
) -> dict[str, Any]:
    """Return a configuration section as a dictionary."""

    section = config.get(section_name)

    if not isinstance(section, dict):
        msg = f"Model config section '{section_name}' is missing or invalid."
        raise ValueError(msg)

    return cast(dict[str, Any], section)


def get_config_int(
    section: dict[str, Any],
    key: str,
) -> int:
    """Return a required integer configuration value."""

    value = section.get(key)

    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"Model config value '{key}' must be an integer."
        raise ValueError(msg)

    return value


def get_config_float(
    section: dict[str, Any],
    key: str,
) -> float:
    """Return a required numeric configuration value as float."""

    value = section.get(key)

    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"Model config value '{key}' must be numeric."
        raise ValueError(msg)

    return float(value)


def validate_waves_config(config: WAVESConfig) -> None:
    """Validate runtime WAVES configuration values."""

    if config.sampling_rate <= 0:
        msg = "Model sampling rate must be greater than zero."
        raise ValueError(msg)

    if config.n_fft <= 0 or config.hop_size <= 0 or config.win_size <= 0:
        msg = "STFT parameters must be greater than zero."
        raise ValueError(msg)

    if config.win_size > config.n_fft:
        msg = "Model win_size cannot exceed n_fft."
        raise ValueError(msg)

    if config.dense_channel <= 0:
        msg = "Model dense_channel must be greater than zero."
        raise ValueError(msg)

    if config.dense_depth <= 0:
        msg = "Model dense_depth must be greater than zero."
        raise ValueError(msg)

    if config.n_heads <= 0:
        msg = "Model n_heads must be greater than zero."
        raise ValueError(msg)

    if config.num_tsblocks <= 0:
        msg = "Model num_tsblocks must be greater than zero."
        raise ValueError(msg)

    if config.dense_channel % config.n_heads != 0:
        msg = "dense_channel must be divisible by n_heads."
        raise ValueError(msg)


def load_waves_config(
    config_path: str | Path,
) -> WAVESConfig:
    """Load the runtime WAVES configuration from YAML."""

    path = Path(config_path)

    if not path.is_file():
        msg = f"Model config file was not found: {path}"
        raise FileNotFoundError(msg)

    with path.open(encoding="utf-8") as file:
        raw_data: Any = yaml.safe_load(file)

    if not isinstance(raw_data, dict):
        msg = f"Model config must contain a YAML mapping: {path}"
        raise ValueError(msg)

    raw_config = cast(dict[str, Any], raw_data)

    data = get_config_section(
        raw_config,
        "data",
    )

    model = get_config_section(
        raw_config,
        "model",
    )

    raw_moe = model.get("moe")

    if raw_moe is not None and not isinstance(raw_moe, dict):
        msg = "Model config value 'moe' must be a mapping or null."
        raise ValueError(msg)

    moe = (
        dict(
            cast(
                dict[str, Any],
                raw_moe,
            )
        )
        if isinstance(raw_moe, dict)
        else None
    )

    config = WAVESConfig(
        sampling_rate=get_config_int(
            data,
            "sampling_rate",
        ),
        n_fft=get_config_int(
            data,
            "n_fft",
        ),
        hop_size=get_config_int(
            data,
            "hop_size",
        ),
        win_size=get_config_int(
            data,
            "win_size",
        ),
        dense_channel=get_config_int(
            model,
            "dense_channel",
        ),
        dense_depth=get_config_int(
            model,
            "dense_depth",
        ),
        beta=get_config_float(
            model,
            "beta",
        ),
        compress_factor=get_config_float(
            model,
            "compress_factor",
        ),
        n_heads=get_config_int(
            model,
            "n_heads",
        ),
        num_tsblocks=get_config_int(
            model,
            "num_tsblocks",
        ),
        moe=moe,
    )

    validate_waves_config(config)

    return config
