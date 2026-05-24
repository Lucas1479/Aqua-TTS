"""Spectralis-TTS — GPU-optimized inference layer for GPT-SoVITS.

Provides static KV cache, CUDA Graph acceleration, and pre-compiled BigVGAN
CUDA kernels for 3.5-5x faster text-to-speech inference.
"""

__version__ = "1.0.0"

from spectralis.inference.params import get_sovits_params
from spectralis.inference.streaming import apply_fade_in, apply_fade_out, finalize_stream_chunk

__all__ = [
    "__version__",
    "apply_fade_in",
    "apply_fade_out",
    "finalize_stream_chunk",
    "get_sovits_params",
]
