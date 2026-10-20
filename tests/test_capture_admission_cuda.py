"""Actual CUDA capture versus actual upstream sampling, randn and synchronize."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import threading

import pytest

torch = pytest.importorskip("torch")
from tests.test_graph_runtime import make_decoder, streaming, upstream  # noqa: E402


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA CUDA required")
@pytest.mark.parametrize("layout,graph", [("bucket", True), ("nested", False)])
def test_capture_is_deferred_while_real_sampler_and_cfm_rng_are_active(monkeypatch, layout, graph):
    monkeypatch.setattr(streaming, "_CUDA_GRAPH_WARMUP_STEPS", 2)
    decoder = make_decoder("cuda:0", layout=layout)
    assert decoder._warmup_and_capture_bucket(256, "cuda:0")
    x = torch.arange(1, 17, device="cuda:0").unsqueeze(0)
    prompt = x.clone()
    bert = torch.zeros(1, 1024, 16, device="cuda:0", dtype=torch.float16)
    torch.cuda.synchronize(0)
    sampling, decided, sampled = threading.Event(), threading.Event(), threading.Event()
    capture_entered = []
    original_graph, original_sample = torch.cuda.graph, upstream.sample
    @contextmanager
    def held_graph(*args, **kwargs):
        with original_graph(*args, **kwargs):
            capture_entered.append(True)
            decided.set()
            assert sampled.wait(10)
            yield
    monkeypatch.setattr(torch.cuda, "graph", held_graph)
    calls = []
    def real_sample(*args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            sampling.set()
            assert decided.wait(10)
            try:
                result = original_sample(*args, **kwargs)
                assert torch.isfinite(torch.randn(4, device="cuda:0")).all()
                torch.cuda.synchronize(0)
                return result
            finally:
                sampled.set()
        return original_sample(*args, **kwargs)
    monkeypatch.setattr(upstream, "sample", real_sample)
    def request():
        with torch.cuda.device(0), torch.cuda.stream(torch.cuda.Stream(device=0)):
            return decoder.infer_panel(x, None, prompt, bert, top_k=5, early_stop_num=8, enable_cuda_graph=graph)
    def capture():
        try:
            with torch.cuda.device(0):
                return decoder._warmup_and_capture_bucket(448, "cuda:0")
        finally:
            decided.set()
    with ThreadPoolExecutor(2) as pool:
        work = pool.submit(request)
        assert sampling.wait(10)
        attempt = pool.submit(capture)
        captured = attempt.result(timeout=20)
        tokens, _ = work.result(timeout=20)
        assert captured is False
    assert tokens.numel() and calls and not capture_entered
    # A declined capture is retryable once work has finished.
    monkeypatch.setattr(torch.cuda, "graph", original_graph)
    assert decoder._warmup_and_capture_bucket(448, "cuda:0")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA CUDA required")
def test_admitted_capture_blocks_new_cuda_work_until_exit(monkeypatch):
    from aquatts.inference.cuda_execution import cuda_work
    monkeypatch.setattr(streaming, "_CUDA_GRAPH_WARMUP_STEPS", 2)
    decoder = make_decoder("cuda:0")
    entered, release, launched, working = (threading.Event() for _ in range(4))
    original_graph = torch.cuda.graph
    @contextmanager
    def held_graph(*args, **kwargs):
        with original_graph(*args, **kwargs):
            entered.set()
            assert release.wait(10)
            yield
    monkeypatch.setattr(torch.cuda, "graph", held_graph)
    @cuda_work
    def gpu_work():
        working.set()
        result = torch.randn(16, device="cuda:0")
        torch.cuda.synchronize(0)
        return result.cpu()
    def start_work():
        launched.set()
        return gpu_work()
    with ThreadPoolExecutor(2) as pool:
        capturing = pool.submit(decoder._warmup_and_capture_bucket, 256, "cuda:0")
        assert entered.wait(10)
        work = pool.submit(start_work)
        try:
            assert launched.wait(5) and not working.wait(.05)
        finally:
            release.set()
        assert capturing.result(timeout=10)
        assert torch.isfinite(work.result(timeout=10)).all()
