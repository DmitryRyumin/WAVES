"""
File: config.py

Author: Ksenia Zolina, Dmitry Ryumin, Denis Ivanko, and Alexey Karpov

Description: Configuration utilities for loading WAVES application settings.

License: MIT License
"""

from collections.abc import Mapping
import os
from pathlib import Path
import tomllib
from types import SimpleNamespace
from typing import Any, Final, Protocol

CONFIG_NAME: Final = "config.toml"

PROJECT_ROOT: Final = Path(__file__).resolve().parent.parent

CONFIG_PATH: Final = PROJECT_ROOT / CONFIG_NAME


def is_hugging_face_space() -> bool:
    """Return whether the app is running inside a Hugging Face Space."""

    return bool(
        os.getenv("SPACE_ID")
        or os.getenv("SPACE_HOST")
        or os.getenv("SPACE_REPO_NAME")
        or os.getenv("SPACE_AUTHOR_NAME")
    )


class TabCreator(Protocol):
    """Callable protocol for Gradio tab factory functions."""

    def __call__(
        self,
        language_index: int = 0,
    ) -> Any: ...


def flatten_dict(
    data: Mapping[str, Any],
    prefix: str = "",
) -> dict[str, Any]:
    """Flatten a nested mapping using underscore-separated keys."""

    flattened: dict[str, Any] = {}

    for key, value in data.items():
        flattened_key = f"{prefix}{key}"

        if isinstance(
            value,
            Mapping,
        ):
            flattened.update(
                flatten_dict(
                    value,
                    prefix=f"{flattened_key}_",
                )
            )
        else:
            flattened[flattened_key] = value

    return flattened


def load_toml(
    file_path: str | Path,
) -> dict[str, Any]:
    """Load a TOML file."""

    path = Path(file_path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        msg = f"Configuration file not found: {path}"
        raise FileNotFoundError(msg)

    with path.open("rb") as file:
        return tomllib.load(file)


_RAW_CONFIG: Final[dict[str, Any]] = load_toml(CONFIG_PATH)

_CONFIG_DATA: Final[SimpleNamespace] = SimpleNamespace(**flatten_dict(_RAW_CONFIG))


def get_config_str_mapping(
    section_name: str,
) -> dict[str, str]:
    """Return a configuration section containing string key-value pairs."""

    section = _RAW_CONFIG.get(section_name)

    if not isinstance(
        section,
        Mapping,
    ):
        msg = f"Configuration section '{section_name}' must be a TOML table."
        raise TypeError(msg)

    values: dict[str, str] = {}

    for key, value in section.items():
        if not isinstance(
            key,
            str,
        ):
            msg = f"Configuration section '{section_name}' contains a non-string key."
            raise TypeError(msg)

        if not isinstance(
            value,
            str,
        ):
            msg = f"Configuration value '{section_name}.{key}' must be a string."
            raise TypeError(msg)

        values[key] = value

    return values


def load_tab_creators(
    available_functions: Mapping[
        str,
        TabCreator,
    ],
) -> dict[str, TabCreator]:
    """Resolve tab creator functions from the loaded WAVES configuration."""

    tab_creators_data = get_config_str_mapping("TabCreators")

    tab_creators: dict[
        str,
        TabCreator,
    ] = {}

    for (
        tab_name,
        function_name,
    ) in tab_creators_data.items():
        tab_creator = available_functions.get(function_name)

        if tab_creator is None:
            available = ", ".join(sorted(available_functions)) or "none"

            msg = (
                f"Tab creator function "
                f"'{function_name}' "
                f"for tab '{tab_name}' "
                "was not found. "
                f"Available functions: "
                f"{available}."
            )
            raise KeyError(msg)

        tab_creators[tab_name] = tab_creator

    return tab_creators


def get_config_bool(
    field_name: str,
    default_value: bool,
) -> bool:
    """Return a boolean WAVES configuration value."""

    value = getattr(
        _CONFIG_DATA,
        field_name,
        default_value,
    )

    if isinstance(
        value,
        bool,
    ):
        return value

    return default_value


def get_config_int(
    field_name: str,
    default_value: int,
) -> int:
    """Return an integer WAVES configuration value."""

    value = getattr(
        _CONFIG_DATA,
        field_name,
        default_value,
    )

    if isinstance(
        value,
        bool,
    ):
        return default_value

    if isinstance(
        value,
        int,
    ):
        return value

    return default_value


def get_config_float(
    field_name: str,
    default_value: float,
) -> float:
    """Return a floating-point WAVES configuration value."""

    value = getattr(
        _CONFIG_DATA,
        field_name,
        default_value,
    )

    if isinstance(
        value,
        bool,
    ):
        return default_value

    if isinstance(
        value,
        int | float,
    ):
        return float(value)

    return default_value


def get_config_str(
    field_name: str,
    default_value: str,
) -> str:
    """Return a string WAVES configuration value."""

    value = getattr(
        _CONFIG_DATA,
        field_name,
        default_value,
    )

    if isinstance(
        value,
        str,
    ):
        return value

    return default_value


def get_config_str_list(
    field_name: str,
    default_value: list[str],
) -> list[str]:
    """Return a list of string WAVES configuration values."""

    value = getattr(
        _CONFIG_DATA,
        field_name,
        default_value,
    )

    if isinstance(
        value,
        list,
    ) and all(
        isinstance(
            item,
            str,
        )
        for item in value
    ):
        return list(value)

    return list(default_value)
