# -*- coding: utf-8 -*-
"""Tests for HTTP server endpoints using FastAPI TestClient with a fake inferencer."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from spectralis.server import _create_app
from spectralis.voice_registry import Voice, VoiceRegistry


class FakeInferencer:
    """Minimal fake TTSInferencer for server endpoint tests."""

    def __init__(self):
        self.device = "cpu"
        self.t2s_model = True
        self.sovits_model = True
        self._cuda_graph_preset = "off"

    def infer_stream(self, text, ref_audio_path, prompt_text=None,
                     text_language="日文", prompt_language="日文",
                     preset=None, how_to_cut="不切", **kwargs):
        """Yield one chunk of fake audio per call."""
        sr = 24000
        # Generate 0.1s of silence (float32)
        chunk = np.zeros(int(sr * 0.1), dtype=np.float32)
        yield sr, chunk, text


@pytest.fixture
def registry():
    """In-memory VoiceRegistry with one pre-registered voice."""
    reg = VoiceRegistry()
    reg.add(Voice(name="alice", ref_audio_path="/tmp/alice.wav",
                  prompt_text="Hello world", prompt_language="英文"))
    return reg


@pytest.fixture
def client(registry):
    """FastAPI TestClient wired to a fake inferencer and in-memory registry."""
    app = _create_app(FakeInferencer(), voice_registry=registry)
    return TestClient(app)


class TestHealth:
    def test_health_returns_status(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["device"] == "cpu"
        assert data["gpt_loaded"] is True
        assert data["sovits_loaded"] is True


class TestVoices:
    def test_list_voices(self, client, registry):
        resp = client.get("/voices")
        assert resp.status_code == 200
        data = resp.json()
        assert "voices" in data
        assert len(data["voices"]) == 1
        assert data["voices"][0]["name"] == "alice"

    def test_add_voice(self, client):
        resp = client.post("/voices/add", json={
            "name": "bob",
            "ref_audio_path": "/tmp/bob.wav",
            "prompt_text": "Hi there",
            "prompt_language": "ja",
        })
        assert resp.status_code == 200
        assert resp.json()["voice"]["name"] == "bob"

    def test_remove_voice(self, client, registry):
        resp = client.delete("/voices/alice")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        # Verify it's gone
        resp2 = client.get("/voices")
        assert len(resp2.json()["voices"]) == 0

    def test_remove_unknown_voice(self, client):
        resp = client.delete("/voices/nobody")
        assert resp.status_code == 404


class TestTTSFile:
    def test_generates_wav_with_valid_text(self, client):
        resp = client.post("/tts/file?text=Hello&voice=alice")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/wav"
        # WAV header starts with "RIFF"
        assert resp.content[:4] == b"RIFF"

    def test_rejects_empty_text(self, client):
        resp = client.post("/tts/file?text=  &voice=alice")
        assert resp.status_code == 400

    def test_rejects_missing_ref_and_voice(self, client):
        resp = client.post("/tts/file?text=Hello")
        assert resp.status_code == 400

    def test_unknown_voice_returns_404(self, client):
        resp = client.post("/tts/file?text=Hello&voice=nobody")
        assert resp.status_code == 404
