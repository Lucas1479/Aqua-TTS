"""Opt-in real multi-GPU correctness checks; never probe CUDA by default.

Small models exercise real CUDA allocation, RNG, streams, capture and replay.
The loader factory replaces checkpoint IO/kernel compilation only; no CUDA
device APIs are mocked. This is not a pretrained BigVGAN/audio-quality test.
"""
import os

import pytest

if os.environ.get("AQUATTS_RUN_MULTI_GPU") != "1":
    pytest.skip("set AQUATTS_RUN_MULTI_GPU=1 only when all selected GPUs are idle", allow_module_level=True)

import gc  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
import itertools  # noqa: E402
import threading  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import torch  # noqa: E402

if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    pytest.fail("multi-GPU validation was requested but fewer than two CUDA devices are visible", pytrace=False)

from aquatts import TTSInferencer  # noqa: E402
from aquatts.inference.cuda_execution import cuda_execution, cuda_work  # noqa: E402
from aquatts.inference.cuda_runtime import resolve_cuda_device, warmup_vocoder  # noqa: E402
from aquatts.inference.streaming import load_bigvgan  # noqa: E402
from tests.test_graph_runtime import make_decoder, streaming  # noqa: E402

PAIRS = list(itertools.permutations(range(torch.cuda.device_count()), 2))
PAIR_IDS = [f"current{current}-target{target}" for current, target in PAIRS]


@pytest.fixture(autouse=True)
def isolated_work(monkeypatch):
    assert not cuda_execution._owners
    monkeypatch.setattr(streaming, "_CUDA_GRAPH_WARMUP_STEPS", 2)
    original_device = torch.cuda.current_device()
    yield
    assert not cuda_execution._owners
    assert torch.cuda.current_device() == original_device
    gc.collect()


class CudaVocoder(torch.nn.Module):
    """Small real GPU computation, with observable device and RNG behavior."""

    def __init__(self, device="cpu"):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1, 7, 1, device=device, dtype=torch.float16))
        self.h = SimpleNamespace(num_mels=7)
        self.calls = []
        self.fail = False

    def remove_weight_norm(self):
        pass

    def forward(self, mel):
        assert torch.is_inference_mode_enabled()
        assert torch.cuda.current_device() == self.weight.device.index == mel.device.index
        assert mel.dtype == self.weight.dtype
        self.calls.append((torch.cuda.current_device(), tuple(mel.shape)))
        torch.rand(1)
        noise = torch.rand_like(mel)
        if self.fail:
            raise RuntimeError("injected forward failure after consuming RNG")
        return ((mel + noise * .01) * self.weight).sum(dim=1, keepdim=True)


def rng_states():
    return [torch.get_rng_state(), *torch.cuda.get_rng_state_all()]


@pytest.mark.parametrize("current,target", PAIRS, ids=PAIR_IDS)
def test_real_device_resolution_and_allocation(current, target):
    with torch.cuda.device(current):
        assert resolve_cuda_device("cuda") == torch.device("cuda", current)
        selected = resolve_cuda_device(f"cuda:{target}")
        assert selected == torch.device("cuda", target)
        data = torch.arange(16, device=selected)
        assert data.device == selected and data.sum().item() == 120
        assert torch.cuda.current_device() == current


@pytest.mark.parametrize("current,target", PAIRS, ids=PAIR_IDS)
@pytest.mark.parametrize("fail", [False, True], ids=["success", "failure"])
def test_warmup_on_noncurrent_gpu_restores_context_and_all_rng(current, target, fail):
    model = CudaVocoder(f"cuda:{target}")
    model.fail = fail
    with torch.cuda.device(current):
        before = rng_states()
        stats = warmup_vocoder(model, [9, 23], 2)
        assert stats["status"] == ("failed" if fail else "completed"), stats
        assert stats["device"] == f"cuda:{target}"
        assert stats["forward_calls"] == (0 if fail else 4)
        assert model.calls == ([(target, (1, 7, 9))] if fail else
                               [(target, (1, 7, 9))] * 2 + [(target, (1, 7, 23))] * 2)
        assert torch.cuda.current_device() == current
        assert all(torch.equal(a, b) for a, b in zip(before, rng_states()))


@pytest.mark.parametrize("current,target", PAIRS, ids=PAIR_IDS)
def test_regular_vocoder_forward_on_noncurrent_gpu(current, target):
    value = TTSInferencer.__new__(TTSInferencer)
    value.bigvgan_model = CudaVocoder(f"cuda:{target}")
    value.vocoder_shape_counts = {}
    mel = torch.ones(1, 7, 23, device=f"cuda:{target}", dtype=torch.float16)
    with torch.cuda.device(current):
        output = value._run_bigvgan(mel)
        assert output.device == mel.device and output.shape == (1, 1, 23)
        assert torch.isfinite(output).all().item()
        assert value.bigvgan_model.calls == [(target, (1, 7, 23))]
        assert value.vocoder_shape_counts == {23: 1}
        assert torch.cuda.current_device() == current


@pytest.mark.parametrize("current,target", PAIRS, ids=PAIR_IDS)
@pytest.mark.parametrize("loader", ["inferencer", "kernel_fallback", "auxiliary"])
def test_loaders_select_real_target_gpu_and_restore_context(monkeypatch, tmp_path, current, target, loader):
    import aquatts.inferencer as module

    calls = []

    def factory(_path, *, use_cuda_kernel, **options):
        assert options == {"local_files_only": True}
        assert torch.cuda.current_device() == target
        # Real implicit allocation verifies the device guard around the loader.
        probe = torch.ones(2, device="cuda")
        assert probe.device.index == target and probe.sum().item() == 2
        calls.append(use_cuda_kernel)
        if use_cuda_kernel:
            raise RuntimeError("injected kernel load failure")
        return CudaVocoder()

    # Only checkpoint IO is replaced. CUDA contexts, transfers and forwards run.
    monkeypatch.setattr(module.bigvgan.BigVGAN, "from_pretrained", factory)
    monkeypatch.setenv("BIGVGAN_USE_CUDA_KERNEL", "1" if loader == "kernel_fallback" else "0")
    for filename in ("config.json", "bigvgan_generator.pt"):
        (tmp_path / filename).touch()
    with torch.cuda.device(current):
        if loader == "auxiliary":
            model = load_bigvgan(str(tmp_path), device=f"cuda:{target}", offline_only=True)
        else:
            value = TTSInferencer.__new__(TTSInferencer)
            value.device = torch.device("cuda", target)
            value._tts_device_idx, value.is_half, value.model_version = target, True, "v3"
            value.bigvgan_path, value.offline_only = str(tmp_path), True
            value._bigvgan_warmup_shapes, value._bigvgan_warmup_repeats = (9,), 1
            value._load_bigvgan_model()
            model = value.bigvgan_model
            assert value.vocoder_warmup_stats["status"] == "completed"
            assert value.bigvgan_kernel_mode == "pytorch"
        assert calls == ([True, False] if loader == "kernel_fallback" else [False])
        assert next(model.parameters()).device == torch.device("cuda", target)
        assert torch.cuda.current_device() == current


def prepared_decoder(target, layout):
    model = make_decoder(f"cuda:{target}", layout=layout, buckets=[256])
    assert model.precapture_cuda_graph([256]) == {256: True}
    inputs = model.bucket_static_inputs[256]
    tensors = [inputs["xy_pos"], inputs["pos_idx"], inputs["key_valid_mask"],
               *inputs["k_cache"], *inputs["v_cache"], *model.bucket_static_outputs[256].values()]
    assert all(t.device == torch.device("cuda", target) for t in tensors)
    return model


def request(model, target, graph, top_k=1):
    device = torch.device("cuda", target)
    x = torch.arange(1, 17, device=device).unsqueeze(0)
    bert = torch.zeros(1, 1024, 16, device=device, dtype=torch.float16)
    with torch.no_grad():
        tokens, count = model.infer_panel(
            x, torch.tensor([16], device=device), x.clone(), bert,
            top_k=top_k, early_stop_num=8, enable_cuda_graph=graph)
    assert tokens.device == device and tokens.numel() and torch.isfinite(tokens).all().item()
    assert (model.cuda_graph_stats["graph_replay_steps"] > 0) is graph
    torch.cuda.synchronize(device)
    return tokens.cpu(), count


@pytest.mark.parametrize("current,target", PAIRS, ids=PAIR_IDS)
@pytest.mark.parametrize("layout", ["bucket", "nested"])
def test_cross_gpu_capture_order_and_alternating_replay(current, target, layout):
    # Capture on A while B is current, then on B while A is current. Reverse
    # parametrization covers the opposite first-capture order in the same process.
    with torch.cuda.device(target):
        first = prepared_decoder(current, layout)
        assert torch.cuda.current_device() == target
    with torch.cuda.device(current):
        second = prepared_decoder(target, layout)
        assert torch.cuda.current_device() == current
    for model, selected, other in ((first, current, target), (second, target, current)):
        with torch.cuda.device(other):
            expected, expected_count = request(model, selected, False)
            actual, count = request(model, selected, True)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert count == expected_count and torch.cuda.current_device() == other
    # Return to the first graph after replaying the second device's graph.
    for model, selected, other in ((first, current, target), (second, target, current)):
        with torch.cuda.device(other):
            request(model, selected, True)
            assert torch.cuda.current_device() == other


@pytest.mark.parametrize("current,target", PAIRS, ids=PAIR_IDS)
@pytest.mark.parametrize("layout", ["bucket", "nested"])
def test_concurrent_replay_on_distinct_gpus(current, target, layout):
    with torch.cuda.device(target):
        first = prepared_decoder(current, layout)
    with torch.cuda.device(current):
        second = prepared_decoder(target, layout)
    barrier = threading.Barrier(2)

    def run(model, selected, other):
        with torch.cuda.device(other):
            barrier.wait(timeout=10)
            # Keep the real upstream randomized sampler enabled.
            result = request(model, selected, True, top_k=5)
            assert torch.cuda.current_device() == other
            return result

    with ThreadPoolExecutor(2) as pool:
        a = pool.submit(run, first, current, target)
        b = pool.submit(run, second, target, current)
        assert a.result(timeout=30)[0].numel() and b.result(timeout=30)[0].numel()


@pytest.mark.parametrize("current,target", PAIRS, ids=PAIR_IDS)
def test_capture_defers_for_work_on_other_gpu_then_retries(current, target):
    decoder = make_decoder(f"cuda:{target}", buckets=[256])
    started, release = threading.Event(), threading.Event()

    @cuda_work
    def work():
        with torch.cuda.device(current):
            torch.empty(32, device=f"cuda:{current}").exponential_()
            started.set()
            assert release.wait(10)
            result = torch.randn(32, device=f"cuda:{current}")
            torch.cuda.synchronize(current)
            assert torch.isfinite(result).all().item()

    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(work)
        try:
            assert started.wait(10)
            with torch.cuda.device(current):
                assert decoder.precapture_cuda_graph([256]) == {256: False}
                assert not decoder.bucket_graphs
                assert torch.cuda.current_device() == current
        finally:
            release.set()
        future.result(timeout=15)
    with torch.cuda.device(current):
        assert decoder.precapture_cuda_graph([256]) == {256: True}
        request(decoder, target, True)
        assert torch.cuda.current_device() == current
