"""Demo measurements separate PCM arrival from the first active PCM window."""
from pathlib import Path
import runpy
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


@pytest.mark.parametrize("has_voice", [False, True])
def test_demo_preserves_first_packet_metric_and_reports_pcm_lead(monkeypatch, has_voice):
    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")
    monkeypatch.setenv("PYTHONUTF8", "1")
    monkeypatch.setenv("ENABLE_CUDA_GRAPH", "0")
    demo = runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples/play_ete.py"))
    rate = 24000
    audio = np.zeros(28800, dtype=np.float32)
    if has_voice:
        audio[21600:] = .2

    def generate(**kwargs):
        for start in range(0, len(audio), 8448):
            yield rate, audio[start:start + 8448], ""

    args = SimpleNamespace(seed=-1, ref_text="Reference.", text_lang="英文", ref_lang="英文",
                           how_to_cut="不切", top_k=5, top_p=1., temperature=.6, speed=1.,
                           sample_steps=4, no_cuda_graph=True, chunk_size_seconds=.35,
                           verbose=True, show_total=False)
    tts = SimpleNamespace(infer_stream=generate, t2s_stats=[])
    result = demo["play_utterance"](tts, None, None, args, "ref.wav", "test", "Text.",
                                    output_stream=Mock())
    assert result["first_audio_ms"] >= 0
    if has_voice:
        assert result["lead_ms"] == 900
        assert result["first_voiced_chunk_ms"] >= result["first_audio_ms"]
    else:
        assert result["lead_ms"] is None
        assert result["first_voiced_chunk_ms"] is None
