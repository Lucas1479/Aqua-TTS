"""Prepared-only promotion preserves KV and clears nonfinite unwritten tails."""
import pytest

torch = pytest.importorskip("torch")
from tests.test_graph_runtime import make_decoder, streaming, upstream  # noqa: E402


@pytest.mark.parametrize("layout", ["bucket", "nested"])
@pytest.mark.parametrize("poison", [float("nan"), float("inf")])
def test_only_newly_exposed_tail_is_cleared(layout, poison):
    model = make_decoder(layout=layout)
    old_k, old_v = streaming._graph_kv_buffers(model, 256, "cpu", torch.float32)
    keys, values = streaming._graph_kv_buffers(model, 448, "cpu", torch.float32)
    for roots in model._graph_kv_roots.values():
        for root in roots:
            root.fill_(7)
    for item in old_k + old_v:
        item.fill_(17)
    for item in keys + values:
        item[:, 256:].fill_(poison)
    before = {key: [root.clone() for root in roots] for key, roots in model._graph_kv_roots.items()}
    mask = torch.ones(1, 448, dtype=torch.bool)
    position = torch.full((1, 1, 64), -1, dtype=torch.long)
    model.bucket_graphs = {256: object(), 448: object()}
    model.bucket_static_inputs[448] = {"k_cache": keys, "v_cache": values,
                                       "key_valid_mask": mask, "pos_idx": position}
    result = streaming._promote_prepared_bucket(model, 256, 256)
    assert result == (448 if layout == "nested" else None)
    if layout == "nested":
        assert mask[:, :256].all() and not mask[:, 256:].any()
        assert position.eq(256).all()
        for item in keys + values:
            assert item[:, :256].eq(17).all() and item[:, 256:].eq(0).all()
        for root in before[None]:
            root[:, :, 256:448] = 0
    for key, roots in model._graph_kv_roots.items():
        for expected, actual in zip(before[key], roots):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0, equal_nan=True)
    assert streaming._promote_prepared_bucket(model, 256, 255) is None
    assert streaming._promote_prepared_bucket(model, 448, 448) is None


def _flash(model, monkeypatch):
    from aquatts.modeling.t2s_flash_attn import apply_flash_attn_patch, is_flash_attn_available
    if not is_flash_attn_available():
        pytest.skip("FlashAttention2 is not installed")
    assert apply_flash_attn_patch(model, mode="valid")
    def reject(*_args, **_kwargs):
        pytest.fail("strict FA2 unexpectedly used SDPA decode")
    for block in model.t2s_transformer_static.blocks:
        monkeypatch.setattr(block.sdpa_block, "decode_next_token_with_static_cache", reject)


def _run(model, prefix, steps, trace, graph=True):
    trace.clear()
    x = torch.arange(prefix - 16, device="cuda:0").remainder(62).unsqueeze(0)
    prompt = torch.ones(1, 16, dtype=torch.long, device="cuda:0")
    bert = torch.zeros(1, 1024, prefix - 16, dtype=torch.float16, device="cuda:0")
    tokens, count = model.infer_panel(x, None, prompt, bert, early_stop_num=steps,
                                     top_k=1, enable_cuda_graph=graph)
    return tokens.clone(), count, [item.clone() for item in trace]


def _teacher_forcing(monkeypatch, trace):
    monkeypatch.setattr(streaming, "_CUDA_GRAPH_WARMUP_STEPS", 2)
    monkeypatch.setattr(streaming, "_should_stop_on_eos", lambda *_: False)
    def sample(logits, _history, **_kwargs):
        trace.append(logits.clone())
        return torch.full_like(logits[:, :1], (len(trace) - 1) % 62, dtype=torch.long), None
    monkeypatch.setattr(upstream, "sample", sample)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA CUDA required")
@pytest.mark.parametrize("attention", ["sdpa", "flash"])
@pytest.mark.parametrize("prepared,steps,promotions,dynamic,prefix", [
    ([256, 448, 512, 768, 1024], 600, 3, False, 100),
    ([256, 1024], 600, 1, False, 100),
    ([256], 200, 0, True, 100),
    ([256, 512], 500, 1, True, 100),
    ([256, 448, 512, 768, 1024], 1000, 4, True, 100),
    ([1024], 10, 0, True, 960),
])
def test_prepared_promotions_and_full_history_fallback(monkeypatch, attention, prepared, steps,
                                                      promotions, dynamic, prefix):
    trace = []
    _teacher_forcing(monkeypatch, trace)
    with torch.cuda.device(0), torch.no_grad():
        model = make_decoder("cuda:0", layout="nested")
        reference = make_decoder("cuda:0", layout="nested", buckets=[1024])
        if attention == "flash":
            _flash(model, monkeypatch)
            _flash(reference, monkeypatch)
        assert all(model.precapture_cuda_graph(prepared).values())
        assert all(reference.precapture_cuda_graph([1024]).values())
        keys = set(model.bucket_graphs)
        pointers = [root.data_ptr() for root in model._graph_kv_roots[None]]
        monkeypatch.setattr(streaming, "_capture_bucket", lambda *_: pytest.fail("captured live KV"))
        expected = _run(reference, prefix, steps, trace)
        actual = _run(model, prefix, steps, trace)
        torch.testing.assert_close(expected[0], actual[0], rtol=0, atol=0)
        assert expected[1] == actual[1]
        assert len(expected[2]) == len(actual[2])
        for left, right in zip(expected[2], actual[2]):
            torch.testing.assert_close(left, right, rtol=2e-3, atol=2e-3)
        stats = model.cuda_graph_stats
        assert stats["promotions"] == promotions
        assert (stats["dynamic_steps"] > 0) is dynamic
        assert stats["capture_attempts"] == 0
        assert stats["final_kv"] == prefix + steps
        assert stats["generated_tokens"] == steps
        assert stats["total_steps"] == 1 + stats["graph_replay_steps"] + stats["static_steps"] + stats["dynamic_steps"]
        assert set(model.bucket_graphs) == keys
        assert [root.data_ptr() for root in model._graph_kv_roots[None]] == pointers


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA CUDA required")
@pytest.mark.parametrize("attention", ["sdpa", "flash"])
@pytest.mark.parametrize("poison", [float("nan"), float("inf")])
def test_nonfinite_previous_request_tail_does_not_poison_promotion(monkeypatch, attention, poison):
    trace = []
    _teacher_forcing(monkeypatch, trace)
    with torch.cuda.device(0), torch.no_grad():
        model = make_decoder("cuda:0", layout="nested")
        reference = make_decoder("cuda:0", layout="nested", buckets=[448])
        if attention == "flash":
            _flash(model, monkeypatch)
            _flash(reference, monkeypatch)
        assert all(model.precapture_cuda_graph([256, 448]).values())
        assert all(reference.precapture_cuda_graph([448]).values())
        for root in model._graph_kv_roots[None]:
            root[:, :, 400].fill_(poison)
        expected = _run(reference, 100, 200, trace)
        actual = _run(model, 100, 200, trace)
        assert all(torch.isfinite(item).all() for item in actual[2])
        torch.testing.assert_close(expected[0], actual[0], rtol=0, atol=0)
        for left, right in zip(expected[2], actual[2]):
            torch.testing.assert_close(left, right, rtol=2e-3, atol=2e-3)
        assert model.cuda_graph_stats["promotions"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA CUDA required")
@pytest.mark.parametrize("attention", ["sdpa", "flash"])
def test_replay_failure_preserves_active_prefix_for_eager_decode(monkeypatch, attention):
    trace = []
    _teacher_forcing(monkeypatch, trace)
    with torch.cuda.device(0), torch.no_grad():
        model = make_decoder("cuda:0", layout="nested")
        if attention == "flash":
            _flash(model, monkeypatch)
        assert all(model.precapture_cuda_graph([256, 448]).values())
        expected = _run(model, 100, 50, trace, graph=False)
        def fail_before_launch(*_args):
            raise RuntimeError("injected before Graph launch")
        monkeypatch.setattr(streaming, "_replay_cuda_graph", fail_before_launch)
        actual = _run(model, 100, 50, trace)
        torch.testing.assert_close(expected[0], actual[0], rtol=0, atol=0)
        assert len(expected[2]) == len(actual[2])
        for left, right in zip(expected[2], actual[2]):
            torch.testing.assert_close(left, right, rtol=2e-3, atol=2e-3)
        stats = model.cuda_graph_stats
        assert stats["graph_replay_steps"] == stats["promotions"] == 0
        assert stats["static_steps"] == 50 and stats["dynamic_steps"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA CUDA required")
def test_initial_lazy_miss_is_allowed_but_promotion_does_not_capture(monkeypatch):
    trace = []
    _teacher_forcing(monkeypatch, trace)
    with torch.cuda.device(0), torch.no_grad():
        model = make_decoder("cuda:0", layout="nested")
        assert all(model.precapture_cuda_graph([448]).values())
        pointers = [root.data_ptr() for root in model._graph_kv_roots[None]]
        capture = streaming._capture_bucket
        calls = []
        def initial_only(decoder, bucket, device):
            calls.append(bucket)
            assert calls == [256]
            return capture(decoder, bucket, device)
        monkeypatch.setattr(streaming, "_capture_bucket", initial_only)
        _run(model, 100, 200, trace)
        stats = model.cuda_graph_stats
        assert calls == [256]
        assert stats["capture_attempts"] == stats["promotions"] == 1
        assert stats["graph_replay_steps"] == 200
        assert stats["static_steps"] == stats["dynamic_steps"] == 0
        assert [root.data_ptr() for root in model._graph_kv_roots[None]] == pointers


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA CUDA required")
@pytest.mark.parametrize("attention", ["sdpa", "flash"])
def test_initial_capture_failure_uses_private_full_history_fallback(monkeypatch, attention):
    trace = []
    _teacher_forcing(monkeypatch, trace)
    with torch.cuda.device(0), torch.no_grad():
        model = make_decoder("cuda:0", layout="nested")
        reference = make_decoder("cuda:0", layout="nested", buckets=[448])
        if attention == "flash":
            _flash(model, monkeypatch)
            _flash(reference, monkeypatch)
        assert all(model.precapture_cuda_graph([448]).values())
        assert all(reference.precapture_cuda_graph([448]).values())
        roots = [item.clone() for item in model._graph_kv_roots[None]]
        monkeypatch.setattr(streaming, "_capture_bucket", lambda *_: False)
        expected = _run(reference, 100, 200, trace)
        actual = _run(model, 100, 200, trace)
        torch.testing.assert_close(expected[0], actual[0], rtol=0, atol=0)
        for left, right in zip(expected[2], actual[2]):
            torch.testing.assert_close(left, right, rtol=2e-3, atol=2e-3)
        stats = model.cuda_graph_stats
        assert stats["capture_attempts"] == 1
        assert stats["graph_replay_steps"] == stats["promotions"] == 0
        assert stats["static_steps"] == 156 and stats["dynamic_steps"] == 44
        assert stats["final_kv"] == 300
        for left, right in zip(roots, model._graph_kv_roots[None]):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
