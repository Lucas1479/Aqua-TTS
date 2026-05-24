"""Audio post-processing utilities and BigVGAN model loader. / 音频后处理工具和 BigVGAN 模型加载器。

Lazy imports avoid requiring torch/numpy at package import time, so the
spectralis package can be structurally inspected without a GPU. / 延迟导入可避免在导入包时要求 torch/numpy，因此可以在没有 GPU 的情况下对 spectralis 包进行结构检查。
"""


def apply_fade_in(audio, sr: int, duration_ms: int = 15):
    """Linear fade-in at the start of the audio chunk. / 音频块开头的线性淡入。

    Prevents click/pop artifacts on chunk boundaries during streaming playback. / 防止流式播放时块边界处出现咔嗒/爆音伪影。
    duration_ms: fade duration in milliseconds, clamped to at most 1/4 of audio. / 淡入持续时间（毫秒），最多限制为音频长度的四分之一。
    """
    import numpy as np
    fade_samples = min(int(sr * duration_ms / 1000), len(audio) // 4)
    if fade_samples > 0:
        audio = audio.copy()
        audio[:fade_samples] *= np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    return audio


def apply_fade_out(audio, sr: int, duration_ms: int = 15):
    """Linear fade-out at the end of the audio chunk. / 音频块末尾的线性淡出。

    Prevents click/pop when audio cuts off mid-sample. / 防止音频在样本中间截断时出现咔嗒/爆音。
    duration_ms: fade duration in milliseconds, clamped to at most 1/4 of audio. / 淡出持续时间（毫秒），最多限制为音频长度的四分之一。
    """
    import numpy as np
    fade_samples = min(int(sr * duration_ms / 1000), len(audio) // 4)
    if fade_samples > 0:
        audio = audio.copy()
        audio[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    return audio


def finalize_stream_chunk(audio, sr: int, fade_out_ms: int = 15):
    """Apply fade-out to a stream chunk and normalize to [-1, 1] range. / 对流式音频块应用淡出并归一化到 [-1, 1] 范围。"""
    import numpy as np
    audio = apply_fade_out(audio, sr, fade_out_ms)
    peak = np.abs(audio).max()
    if peak > 1.0:
        audio = audio / peak * 0.95
    return audio


def load_bigvgan(model_dir: str, use_cuda_kernel: bool = False):
    """Load a BigVGAN model from a pretrained directory. / 从预训练目录加载 BigVGAN 模型。

    Args: / 参数：
        model_dir: Path to the BigVGAN pretrained model directory. / BigVGAN 预训练模型目录的路径。
        use_cuda_kernel: If True, attempt to use the fused CUDA kernel
            (requires Ninja + MSVC on Windows). / 如果为 True，尝试使用融合 CUDA kernel（在 Windows 上需要 Ninja + MSVC）。

    Returns: / 返回值：
        BigVGAN model (eval mode, weight norm removed). / BigVGAN 模型（评估模式，已移除 weight norm）。
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
