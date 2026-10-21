import importlib.util
from pathlib import Path

import pytest


def _benchmark(filename="model_latency.py"):
    path = Path(__file__).resolve().parents[1] / "benchmarks" / filename
    spec = importlib.util.spec_from_file_location("aqua_benchmark_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_latency_summary_retains_outlier_and_uses_median():
    rows = [dict(first_audio_ms=value, total_ms=value + 100, audio_seconds=2,
                 rtf=(value + 100) / 2000, semantic_ms=value / 2)
            for value in [100, 110, 5000, 90, 120]]
    result = _benchmark().summarize(rows)
    assert result["median_first_audio_ms"] == 110
    assert result["median_total_ms"] == 210
    assert result["median_rtf"] == .105
    assert result["max_first_audio_ms"] == 5000


def test_empty_latency_summary_is_not_a_successful_measurement():
    with pytest.raises(ValueError, match="empty"):
        _benchmark().summarize([])


def test_steady_summary_p95_and_uninstrumented_semantic_time():
    rows = [dict(first_audio_ms=value, total_ms=value + 1, audio_seconds=2,
                 rtf=(value + 1) / 2000, semantic_ms=None) for value in range(20)]
    result = _benchmark().summarize(rows)
    assert result["median_first_audio_ms"] == 9.5
    assert result["p95_first_audio_ms"] == 18
    assert result["median_semantic_ms"] is None


@pytest.mark.parametrize("raise_inside", [False, True])
def test_raw_pcm_boundary_releases_quiet_pcm_without_changing_samples(raise_inside, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    import numpy as np

    native_lead = object()

    def native_cut(audio, rate):
        return 120

    owner = SimpleNamespace(LeadingSilence=native_lead, leading_cut=native_cut)
    boundary = _benchmark("audio_boundary.py")
    load_owner = Mock(return_value=owner)
    monkeypatch.setattr(boundary, "import_module", load_owner)
    quiet = np.zeros(240, dtype=np.float32)
    voiced = np.full(240, 0.1, dtype=np.float32)
    try:
        with boundary.audio_boundary(None, "raw-pcm"):
            stream = owner.LeadingSilence(24000)
            assert stream.feed(quiet) is quiet  # Quiet first PCM is immediately available.
            assert stream.feed(voiced) is voiced
            assert stream.finish() is None  # No buffered/duplicated packet at EOF.
            assert owner.leading_cut(quiet, 24000) == 0
            if raise_inside:
                raise RuntimeError("measurement failure")
    except RuntimeError:
        assert raise_inside
    load_owner.assert_called_once_with("aquatts.inferencer")
    assert owner.LeadingSilence is native_lead
    assert owner.leading_cut is native_cut


def test_api_pcm_boundary_retains_the_production_onset_behavior(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    owner = SimpleNamespace(LeadingSilence=object(), leading_cut=object())
    native_lead, native_cut = owner.LeadingSilence, owner.leading_cut
    boundary = _benchmark("audio_boundary.py")
    load_owner = Mock(return_value=owner)
    monkeypatch.setattr(boundary, "import_module", load_owner)
    with boundary.audio_boundary(None, "api-pcm"):
        assert owner.LeadingSilence is native_lead
        assert owner.leading_cut is native_cut
    load_owner.assert_not_called()
