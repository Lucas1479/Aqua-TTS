"""Shared Graph state survives competing requests and queued CUDA work."""
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

torch = pytest.importorskip("torch")
from tests.test_graph_runtime import make_decoder, streaming, upstream  # noqa: E402
from aquatts.inference.workspace import GraphWorkspaceRegistry  # noqa: E402


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA CUDA required")
@pytest.mark.parametrize("different_key", [False, True])
@pytest.mark.parametrize("attention", ["sdpa", "flash"])
@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("layout", ["bucket", "nested"])
def test_concurrent_requests_match_serial_logits(monkeypatch, different_key, attention, prepared, layout):
    monkeypatch.setattr(streaming, "_CUDA_GRAPH_WARMUP_STEPS", 2)
    local = threading.local()

    def deterministic_sample(logits, _history, **_kwargs):
        local.trace.append(logits.detach().clone())
        return logits.argmax(-1, keepdim=True), None

    monkeypatch.setattr(upstream, "sample", deterministic_sample)
    with torch.cuda.device(0), torch.no_grad():
        decoder = make_decoder("cuda:0", layout=layout)
        if attention == "flash":
            from aquatts.modeling.t2s_flash_attn import apply_flash_attn_patch, is_flash_attn_available
            if not is_flash_attn_available():
                pytest.skip("FlashAttention2 is not installed")
            assert apply_flash_attn_patch(decoder, mode="valid")
            def reject_fallback(*_args, **_kwargs):
                pytest.fail("FlashAttention test silently fell back to SDPA")
            for block in decoder.t2s_transformer_static.blocks:
                monkeypatch.setattr(block.sdpa_block, "decode_next_token_with_static_cache", reject_fallback)
        if prepared:
            assert decoder._warmup_and_capture_bucket(256, "cuda:0")
            if different_key:
                assert decoder._warmup_and_capture_bucket(448, "cuda:0")
        inputs = []
        for index, length in enumerate((16, 144 if different_key else 16)):
            x = (torch.arange(length, device="cuda:0") + 1 + index * 11).remainder(62).unsqueeze(0)
            prompt = torch.arange(1, 17, device="cuda:0").unsqueeze(0)
            bert = torch.zeros(1, 1024, length, device="cuda:0", dtype=torch.float16)
            inputs.append((x, prompt, bert))
        torch.cuda.synchronize(0)

    def run(index, barrier=None):
        with torch.cuda.device(0), torch.cuda.stream(torch.cuda.Stream(device=0)), torch.no_grad():
            local.trace = []
            if barrier is not None:
                barrier.wait(timeout=10)
            x, prompt, bert = inputs[index]
            tokens, count = decoder.infer_panel(x, None, prompt, bert, top_k=1, early_stop_num=15)
            return tokens.cpu(), count, [logits.cpu() for logits in local.trace]

    serial = [run(0), run(1)]
    if not prepared:
        # The serial reference warmed these keys. Make the concurrent pair
        # exercise genuine lazy misses, rather than only prepared replay.
        decoder.bucket_graphs.clear()
        decoder.bucket_static_inputs.clear()
        decoder.bucket_static_outputs.clear()
    barrier = threading.Barrier(2)
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(run, index, barrier) for index in (0, 1)]
        concurrent = [future.result(timeout=30) for future in futures]
    for expected, actual in zip(serial, concurrent):
        torch.testing.assert_close(expected[0], actual[0], rtol=0, atol=0)
        assert expected[1] == actual[1]
        assert len(expected[2]) == len(actual[2])
        for left, right in zip(expected[2], actual[2]):
            torch.testing.assert_close(left, right, rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA CUDA required")
def test_handoff_waits_for_queued_write_without_device_synchronize():
    registry = GraphWorkspaceRegistry()
    with torch.cuda.device(0):
        data = torch.zeros(128, device="cuda:0")
        first, second = torch.cuda.Stream(), torch.cuda.Stream()
        first.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(first), registry.lease("root", "cuda:0"):
            torch.cuda._sleep(2_000_000)
            data.fill_(19)
        with torch.cuda.stream(second), registry.lease("root", "cuda:0"):
            observed = data.clone()
        second.synchronize()
        assert observed.eq(19).all()
