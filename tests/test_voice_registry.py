# -*- coding: utf-8 -*-
"""Tests for VoiceRegistry and Voice dataclass."""

import json
import os
import tempfile

import pytest
from spectralis.voice_registry import Voice, VoiceRegistry, registry_from_env


class TestVoice:
    def test_defaults(self):
        v = Voice(name="alice", ref_audio_path="/path/to/audio.wav")
        assert v.name == "alice"
        assert v.ref_audio_path == "/path/to/audio.wav"
        assert v.prompt_text == ""
        assert v.prompt_language == "日文"

    def test_full(self):
        v = Voice(name="bob", ref_audio_path="/b.wav", prompt_text="Hello", prompt_language="英文")
        assert v.prompt_text == "Hello"
        assert v.prompt_language == "英文"

    def test_to_dict(self):
        v = Voice(name="alice", ref_audio_path="/a.wav", prompt_text="你好")
        d = v.to_dict()
        assert d["name"] == "alice"
        assert d["prompt_text"] == "你好"
        assert d["prompt_language"] == "日文"

    def test_from_dict(self):
        d = {"name": "alice", "ref_audio_path": "/a.wav", "prompt_text": "hi", "prompt_language": "英文"}
        v = Voice.from_dict(d)
        assert v.name == "alice"
        assert v.prompt_text == "hi"
        assert v.prompt_language == "英文"

    def test_from_dict_defaults(self):
        d = {"name": "alice", "ref_audio_path": "/a.wav"}
        v = Voice.from_dict(d)
        assert v.prompt_text == ""
        assert v.prompt_language == "日文"

    def test_immutable(self):
        v = Voice(name="alice", ref_audio_path="/a.wav")
        with pytest.raises(Exception):
            v.name = "bob"


class TestVoiceRegistry:
    def test_add_and_get(self):
        reg = VoiceRegistry()
        reg.add(Voice(name="alice", ref_audio_path="/a.wav"))
        v = reg.get("alice")
        assert v is not None
        assert v.ref_audio_path == "/a.wav"

    def test_get_missing(self):
        reg = VoiceRegistry()
        assert reg.get("nobody") is None

    def test_remove(self):
        reg = VoiceRegistry()
        reg.add(Voice(name="alice", ref_audio_path="/a.wav"))
        assert reg.remove("alice") is True
        assert reg.get("alice") is None

    def test_remove_missing(self):
        reg = VoiceRegistry()
        assert reg.remove("nobody") is False

    def test_contains(self):
        reg = VoiceRegistry()
        reg.add(Voice(name="alice", ref_audio_path="/a.wav"))
        assert "alice" in reg
        assert "bob" not in reg

    def test_list(self):
        reg = VoiceRegistry()
        reg.add(Voice(name="a", ref_audio_path="/a.wav"))
        reg.add(Voice(name="b", ref_audio_path="/b.wav"))
        names = [v.name for v in reg.list()]
        assert sorted(names) == ["a", "b"]

    def test_names(self):
        reg = VoiceRegistry()
        reg.add(Voice(name="a", ref_audio_path="/a.wav"))
        assert set(reg.names()) == {"a"}

    def test_len(self):
        reg = VoiceRegistry()
        assert len(reg) == 0
        reg.add(Voice(name="a", ref_audio_path="/a.wav"))
        assert len(reg) == 1

    def test_update_existing(self):
        reg = VoiceRegistry()
        reg.add(Voice(name="alice", ref_audio_path="/old.wav"))
        reg.add(Voice(name="alice", ref_audio_path="/new.wav"))
        assert len(reg) == 1
        assert reg.get("alice").ref_audio_path == "/new.wav"


class TestVoiceRegistryPersistence:
    def test_save_and_load(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            path = f.name

        try:
            reg = VoiceRegistry(json_path=path)
            reg.add(Voice(name="alice", ref_audio_path="/a.wav", prompt_text="hello"))
            reg.add(Voice(name="bob", ref_audio_path="/b.wav"))

            reg2 = VoiceRegistry(json_path=path)
            assert len(reg2) == 2
            assert reg2.get("alice").prompt_text == "hello"
            assert reg2.get("bob").ref_audio_path == "/b.wav"
        finally:
            os.unlink(path)

    def test_load_corrupted_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            f.write("not valid json {{{")
            path = f.name

        try:
            reg = VoiceRegistry(json_path=path)
            assert len(reg) == 0  # silently handles corruption
        finally:
            os.unlink(path)

    def test_save_creates_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "subdir", "voices.json")
            reg = VoiceRegistry(json_path=json_path)
            reg.add(Voice(name="alice", ref_audio_path="/a.wav"))
            assert os.path.isfile(json_path)

            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            assert len(data) == 1


class TestRegistryFromEnv:
    def test_env_var(self, monkeypatch):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            path = f.name

        try:
            reg = VoiceRegistry(json_path=path)
            reg.add(Voice(name="env_voice", ref_audio_path="/e.wav"))
            monkeypatch.setenv("SPECTRALIS_VOICE_JSON", path)

            reg2 = registry_from_env()
            v = reg2.get("env_voice")
            assert v is not None
            assert v.ref_audio_path == "/e.wav"
        finally:
            os.unlink(path)

    def test_no_env_falls_back_to_cwd(self, monkeypatch):
        monkeypatch.delenv("SPECTRALIS_VOICE_JSON", raising=False)
        reg = registry_from_env()
        assert len(reg) == 0
