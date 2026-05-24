import numpy as np
import pytest

from aqua.inference.streaming import apply_fade_in, apply_fade_out, finalize_stream_chunk


class TestFadeIn:
    def test_basic_fade(self):
        audio = np.ones(48000, dtype=np.float32)  # 1 second at 48kHz
        result = apply_fade_in(audio, 48000, duration_ms=15)
        # Fade duration: 15ms at 48kHz = 720 samples, clamped to len/4 = 12000
        assert result[0] < 0.1
        assert result[720] == pytest.approx(1.0, abs=1e-4)

    def test_short_audio(self):
        audio = np.ones(10, dtype=np.float32)  # very short
        result = apply_fade_in(audio, 48000, duration_ms=15)
        assert result[0] < 0.5

    def test_no_copy_if_no_fade(self):
        audio = np.ones(10, dtype=np.float32)
        result = apply_fade_in(audio, 100, duration_ms=0)
        assert result is audio  # no copy when fade_samples == 0


class TestFadeOut:
    def test_basic_fade(self):
        audio = np.ones(48000, dtype=np.float32)
        result = apply_fade_out(audio, 48000, duration_ms=15)
        assert result[-1] < 0.1
        assert result[-721] == pytest.approx(1.0, abs=1e-4)

    def test_short_audio(self):
        audio = np.ones(10, dtype=np.float32)
        result = apply_fade_out(audio, 48000, duration_ms=15)
        assert result[-1] < 0.5


class TestFinalizeStreamChunk:
    def test_normalize_peak(self):
        audio = np.array([0.0, 2.0, -2.0, 0.0], dtype=np.float32)
        result = finalize_stream_chunk(audio, 48000, fade_out_ms=0)
        assert np.abs(result).max() <= 1.0

    def test_no_normalize_if_quiet(self):
        audio = np.array([0.1, -0.1, 0.05], dtype=np.float32)
        result = finalize_stream_chunk(audio, 48000, fade_out_ms=0)
        np.testing.assert_array_almost_equal(np.abs(result[1:-1]), np.abs(audio[1:-1]))
