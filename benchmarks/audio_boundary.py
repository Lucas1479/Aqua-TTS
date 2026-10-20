"""Benchmark-only output boundaries; production onset behavior is unchanged."""
from contextlib import contextmanager
from importlib import import_module
from unittest.mock import patch


class _RawChunks:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    def feed(self, audio, *, final=False):
        return audio

    def finish(self):
        return None


@contextmanager
def audio_boundary(inferencer, boundary):
    """Expose pre-trim PCM for historical TTFP, or retain current API output.

    Use only in isolated benchmark processes. The raw mode bypasses onset
    buffering/cropping after PCM finalization; model generation, normalization,
    CPU transfer, fades and chunk sizes stay intact. This also lets a first-only
    probe stop at the original first chunk, preserving its RNG consumption.
    """
    if boundary == "api-pcm":
        yield
        return
    if boundary != "raw-pcm":
        raise ValueError(f"Unknown audio boundary: {boundary}")
    module = import_module("aquatts.inferencer")
    with patch.object(module, "LeadingSilence", _RawChunks), patch.object(
        inferencer, "_trim_generated_lead", lambda audio, sample_rate: audio
    ):
        yield
