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


@pytest.mark.parametrize("raise_inside", [False, True])
def test_fixed_work_preserves_native_eos_and_predicate_work(raise_inside):
    from types import SimpleNamespace

    benchmark = _load_benchmark_module()
    decoder = SimpleNamespace(EOS=1024)
    calls = []

    def native_check(logits, samples, eos):
        calls.append((logits, samples, eos))
        return True

    owner = SimpleNamespace(_should_stop_on_eos=native_check)
    try:
        with benchmark._fixed_length_eos(decoder, eos_module=owner):
            assert decoder.EOS == 1024
            assert owner._should_stop_on_eos("logits", "samples", decoder.EOS) is False
            if raise_inside:
                raise RuntimeError("benchmark failure")
    except RuntimeError:
        assert raise_inside
    assert calls == [("logits", "samples", 1024)]
    assert decoder.EOS == 1024
    assert owner._should_stop_on_eos is native_check


def test_upstream_fixed_work_restores_its_eos_id():
    from types import SimpleNamespace

    benchmark = _load_benchmark_module()
    decoder = SimpleNamespace(EOS=1024)
    with benchmark._fixed_length_eos(decoder):
        assert decoder.EOS == -1
    assert decoder.EOS == 1024
