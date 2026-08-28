"""
File: torchcodec_compat.py

Author: Ksenia Zolina, Dmitry Ryumin, Denis Ivanko, and Alexey Karpov

Description: TorchCodec compatibility imports for the WAVES Gradio application.

License: MIT License
"""

from typing import TYPE_CHECKING

import torchcodec.decoders as torchcodec_decoders
import torchcodec.encoders as torchcodec_encoders

if TYPE_CHECKING:
    from torchcodec.decoders._audio_decoder import AudioDecoder as AudioDecoder
    from torchcodec.encoders._audio_encoder import AudioEncoder as AudioEncoder
else:
    AudioDecoder = torchcodec_decoders.AudioDecoder
    AudioEncoder = torchcodec_encoders.AudioEncoder

__all__ = [
    "AudioDecoder",
    "AudioEncoder",
]
