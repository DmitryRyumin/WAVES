"""
File: __init__.py

Author: Ksenia Zolina, Dmitry Ryumin, Denis Ivanko, and Alexey Karpov

Description: Model utilities for the WAVES application.

License: MIT License
"""

from waves.models.registry import (
    ModelInfo,
    discover_models,
    get_model_info,
)

__all__ = [
    "ModelInfo",
    "discover_models",
    "get_model_info",
]
