"""
File: loader.py

Author: Ksenia Zolina, Dmitry Ryumin, Denis Ivanko, and Alexey Karpov

Description: Model loading and caching utilities for WAVES.

License: MIT License
"""

from dataclasses import dataclass
from threading import RLock

from safetensors.torch import load_model as load_safetensors_model
import torch

from waves.config import (
    get_config_bool,
    get_config_str,
)
from waves.models.config import (
    WAVESConfig,
    load_waves_config,
)
from waves.models.model import WAVESModel
from waves.models.registry import (
    ModelInfo,
    get_model_info,
)


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """Loaded WAVES model and associated runtime metadata."""

    model: WAVESModel
    config: WAVESConfig
    info: ModelInfo
    device: torch.device


_MODEL_CACHE: dict[
    tuple[str, str],
    LoadedModel,
] = {}

_MODEL_CACHE_LOCK = RLock()


def get_inference_device(
    configured_device: str | None = None,
) -> torch.device:
    """Resolve the configured inference device."""

    device_name = (
        (
            configured_device
            if configured_device is not None
            else get_config_str(
                "Inference_DEVICE",
                "auto",
            )
        )
        .strip()
        .lower()
    )

    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    if device_name.startswith("cuda"):
        if not torch.cuda.is_available():
            msg = "CUDA was requested but is not available."
            raise RuntimeError(msg)

        return torch.device(device_name)

    if device_name == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            msg = "MPS was requested but is not available."
            raise RuntimeError(msg)

        return torch.device("mps")

    if device_name == "cpu":
        return torch.device("cpu")

    msg = f"Unsupported inference device: {device_name}"
    raise ValueError(msg)


def load_model(
    model_name: str,
    *,
    device_name: str | None = None,
    use_cache: bool | None = None,
) -> LoadedModel:
    """Load a WAVES SafeTensors model strictly and optionally cache it."""

    model_info = get_model_info(model_name)
    device = get_inference_device(device_name)

    cache_models = get_config_bool(
        "Model_CACHE_MODELS",
        True,
    )

    if use_cache is not None:
        cache_models = use_cache

    cache_key = (
        str(model_info.weights_path),
        str(device),
    )

    with _MODEL_CACHE_LOCK:
        if cache_models and cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key]

        model_config = load_waves_config(model_info.config_path)
        model = WAVESModel(model_config)

        missing_keys, unexpected_keys = load_safetensors_model(
            model,
            model_info.weights_path,
            strict=True,
            device="cpu",
            backend="mmap",
        )

        if missing_keys or unexpected_keys:
            msg = (
                "SafeTensors model loading produced an inconsistent state: "
                f"missing={missing_keys}, unexpected={unexpected_keys}."
            )
            raise RuntimeError(msg)

        model.to(device)
        model.eval()

        loaded_model = LoadedModel(
            model=model,
            config=model_config,
            info=model_info,
            device=device,
        )

        if cache_models:
            _MODEL_CACHE[cache_key] = loaded_model

        return loaded_model


def clear_model_cache() -> None:
    """Clear cached models and release accelerator memory."""

    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE.clear()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
