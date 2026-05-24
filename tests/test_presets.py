# -*- coding: utf-8 -*-
"""Tests for generation and CUDA Graph presets."""

import pytest
from spectralis.inference.presets import (
    GENERATION_PRESETS,
    CUDA_GRAPH_PRESETS,
    apply_preset,
    list_presets,
    apply_cuda_graph_preset,
    list_cuda_graph_presets,
)


class TestGenerationPresets:
    def test_list_presets(self):
        names = list_presets()
        assert names == ["balanced", "fast", "quality"]

    def test_apply_fast(self):
        p = apply_preset("fast")
        assert p["speed"] == 1.3
        assert p["sample_steps"] == 4
        assert p["top_k"] == 3
        assert p["top_p"] == 1.0
        assert p["temperature"] == 0.6

    def test_apply_quality(self):
        p = apply_preset("quality")
        assert p["speed"] == 1.0
        assert p["sample_steps"] == 16
        assert p["top_k"] == 8

    def test_apply_with_overrides(self):
        p = apply_preset("balanced", overrides={"top_k": 15})
        assert p["top_k"] == 15  # overridden
        assert p["speed"] == 1.1  # from preset

    def test_apply_unknown_preset(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            apply_preset("nonexistent")

    def test_preset_returns_copy(self):
        p1 = apply_preset("fast")
        p2 = apply_preset("fast")
        assert p1 == p2
        assert p1 is not p2  # independent copies

    def test_all_presets_have_required_keys(self):
        required = {"speed", "sample_steps", "top_k", "top_p", "temperature"}
        for name, params in GENERATION_PRESETS.items():
            missing = required - set(params)
            assert not missing, f"Preset {name!r} missing keys: {missing}"


class TestCudaGraphPresets:
    def test_list_presets(self):
        names = list_cuda_graph_presets()
        assert names == ["full", "lazy", "minimal", "off"]

    def test_apply_full(self):
        cg = apply_cuda_graph_preset("full")
        assert cg["enable"] is True
        assert cg["precapture"] is True
        assert cg["buckets"] == [128, 256, 448, 512, 768, 1024]

    def test_apply_minimal(self):
        cg = apply_cuda_graph_preset("minimal")
        assert cg["enable"] is True
        assert cg["precapture"] is True
        assert cg["buckets"] == [256, 512, 1024]

    def test_apply_lazy(self):
        cg = apply_cuda_graph_preset("lazy")
        assert cg["enable"] is True
        assert cg["precapture"] is False
        assert cg["buckets"] is None

    def test_apply_off(self):
        cg = apply_cuda_graph_preset("off")
        assert cg["enable"] is False

    def test_apply_unknown_preset(self):
        with pytest.raises(ValueError, match="Unknown CUDA Graph preset"):
            apply_cuda_graph_preset("nonexistent")
