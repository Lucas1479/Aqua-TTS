"""Explicit-path adapter for the upstream GPT-SoVITS Pro speaker encoder.

Uses upstream ERes2NetV2 and Kaldi features; model sources and weights remain
caller-owned. The feature contract follows GPT-SoVITS's MIT-licensed sv.py.
"""

from pathlib import Path

from aquatts.upstream import configure_speaker_encoder


class SpeakerEncoder:
    def __init__(self, model_path, device, is_half):
        import torch

        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Pro speaker encoder weight is missing: {path}")
        configure_speaker_encoder()
        from GPT_SoVITS.eres2net.ERes2NetV2 import ERes2NetV2
        from GPT_SoVITS.eres2net.kaldi import fbank

        self._fbank = fbank
        self.device = device
        self.dtype = torch.float16 if is_half else torch.float32
        self.model = ERes2NetV2(baseWidth=24, scale=4, expansion=4)
        self.model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.model = self.model.eval().to(device=device, dtype=self.dtype)

    def encode(self, filename):
        """Return the speaker condition for a mono reference resampled to 16 kHz."""
        import librosa
        import torch

        waveform, _ = librosa.load(filename, sr=16000, mono=True)
        audio = torch.as_tensor(waveform, device=self.device, dtype=self.dtype).unsqueeze(0)
        with torch.inference_mode():
            features = self._fbank(audio, num_mel_bins=80, sample_frequency=16000, dither=0)
            return self.model.forward3(features.unsqueeze(0))
