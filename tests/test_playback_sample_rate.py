"""The public streaming example must play Pro PCM at its returned rate."""
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


@pytest.mark.parametrize("rate", [24000, 32000])
def test_streaming_example_uses_model_rate_and_speaker_path(monkeypatch, rate):
    def infer_stream(**kwargs):
        yield rate, None, ""
        yield rate, np.ones(32, dtype=np.float32), "hello"

    factory = Mock(return_value=SimpleNamespace(infer_stream=infer_stream))
    player = Mock()
    stream = player.open.return_value
    monkeypatch.setitem(sys.modules, "aquatts", SimpleNamespace(TTSInferencer=factory))
    monkeypatch.setitem(sys.modules, "pyaudio", SimpleNamespace(PyAudio=lambda: player, paFloat32=1))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "argv", [
        "streaming_inference.py", "--gpt-model", "gpt.ckpt", "--sovits-model", "sovits.pth",
        "--sv-model", "speaker.ckpt", "--ref-audio", "reference.wav", "--ref-text", "hello", "--text", "hello",
    ])
    # The script sets defaults, so isolate the environment as well.
    for name in ("PYTHONIOENCODING", "PYTHONUTF8", "ENABLE_CUDA_GRAPH"):
        monkeypatch.setenv(name, "1")
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples/streaming_inference.py"), run_name="__main__")
    assert factory.call_args.kwargs["sv_model_path"] == "speaker.ckpt"
    assert player.open.call_args.kwargs["rate"] == rate
    stream.write.assert_called_once()
    stream.close.assert_called_once()
    player.terminate.assert_called_once()
