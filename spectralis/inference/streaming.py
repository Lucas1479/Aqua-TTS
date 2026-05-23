"""Audio post-processing utilities and BigVGAN model loader.

Lazy imports avoid requiring torch/numpy at package import time, so the
spectralis package can be structurally inspected without a GPU.
"""


def apply_fade_in(audio, sr: int, duration_ms: int = 15):
    """Linear fade-in at the start of the audio chunk.

    Prevents click/pop artifacts on chunk boundaries during streaming playback.
    duration_ms: fade duration in milliseconds, clamped to at most 1/4 of audio.
    """
    import numpy as np
    fade_samples = min(int(sr * duration_ms / 1000), len(audio) // 4)
    if fade_samples > 0:
        audio = audio.copy()
        audio[:fade_samples] *= np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    return audio


def apply_fade_out(audio, sr: int, duration_ms: int = 15):
    """Linear fade-out at the end of the audio chunk.

    Prevents click/pop when audio cuts off mid-sample.
    duration_ms: fade duration in milliseconds, clamped to at most 1/4 of audio.
    """
    import numpy as np
    fade_samples = min(int(sr * duration_ms / 1000), len(audio) // 4)
    if fade_samples > 0:
        audio = audio.copy()
        audio[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    return audio


def finalize_stream_chunk(audio, sr: int, fade_out_ms: int = 15):
    """Apply fade-out to a stream chunk and normalize to [-1, 1] range."""
    import numpy as np
    audio = apply_fade_out(audio, sr, fade_out_ms)
    peak = np.abs(audio).max()
    if peak > 1.0:
        audio = audio / peak * 0.95
    return audio


def load_bigvgan(model_dir: str, use_cuda_kernel: bool = False):
    """Load a BigVGAN model from a pretrained directory.

    Args:
        model_dir: Path to the BigVGAN pretrained model directory.
        use_cuda_kernel: If True, attempt to use the fused CUDA kernel
            (requires Ninja + MSVC on Windows).

    Returns:
        BigVGAN model (eval mode, weight norm removed).
    """
    import torch
    from GPT_SoVITS.BigVGAN import bigvgan

    model = bigvgan.BigVGAN.from_pretrained(model_dir, use_cuda_kernel=use_cuda_kernel)
    model.remove_weight_norm()
    model = model.eval()

    if torch.cuda.is_available():
        model = model.half().cuda()
    else:
        model = model.cpu()

    return model
