import importlib.util
from pathlib import Path

import pytest


def _benchmark():
    path = Path(__file__).resolve().parents[1] / "benchmarks/model_latency.py"
    spec = importlib.util.spec_from_file_location("model_latency", path)
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
