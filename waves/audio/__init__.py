"""
File: __init__.py

Author: Ksenia Zolina, Dmitry Ryumin, Denis Ivanko, and Alexey Karpov

Description: Audio utilities for the WAVES Gradio application.

License: MIT License
"""

from waves.audio.decoder import (
    decode_audio_for_enhancement,
)

__all__ = [
    "decode_audio_for_enhancement",
]
