from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = ROOT / "benchmarks" / "t2s_speed_bench.py"


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location("aqua_t2s_speed_bench", BENCHMARK_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_default_cases_cover_expected_buckets_and_trace_shape():
    benchmark = _load_benchmark_module()
    buckets = (128, 256, 448, 512, 768, 1024)

    for case in benchmark.DEFAULT_CASES:
        aligned = ((case.initial_len + 31) // 32) * 32
        selected = next(bucket for bucket in buckets if aligned + 96 < bucket)
        assert selected == case.expected_bucket

    conversation = next(case for case in benchmark.DEFAULT_CASES if case.name == "conversation")
    assert conversation.initial_len == 377
    assert conversation.generated_steps == 123
    assert conversation.expected_bucket == 512


def test_summarize_reports_median_and_range():
    benchmark = _load_benchmark_module()
    result = benchmark._summarize(
        [
            {"sync_it_s": 500.0, "wall_it_s": 510.0},
            {"sync_it_s": 550.0, "wall_it_s": 560.0},
            {"sync_it_s": 525.0, "wall_it_s": 535.0},
        ]
    )

    assert result["median_sync_it_s"] == 525.0
    assert result["median_wall_it_s"] == 535.0
    assert result["min_sync_it_s"] == 500.0
    assert result["max_sync_it_s"] == 550.0


def test_summarize_rejects_empty_rows():
    benchmark = _load_benchmark_module()
    with pytest.raises(ValueError, match="empty"):
        benchmark._summarize([])


def test_resolve_file_checks_cwd_then_upstream(tmp_path, monkeypatch):
    benchmark = _load_benchmark_module()
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    checkpoint = upstream / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.chdir(tmp_path)

    assert benchmark._resolve_file("model.ckpt", upstream) == checkpoint.resolve()


def test_parse_buckets_deduplicates_and_sorts():
    benchmark = _load_benchmark_module()
    assert benchmark._parse_buckets("768,448,512,448") == [448, 512, 768]
