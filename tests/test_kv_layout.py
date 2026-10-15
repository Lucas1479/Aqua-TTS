"""Storage aliasing and Graph identity retain the original computation buckets."""
import pytest

torch = pytest.importorskip("torch")
from tests.test_graph_runtime import make_decoder, streaming, upstream  # noqa: E402


@pytest.mark.parametrize("layout,owners", [("bucket", 2), ("nested", 1)])
def test_prefix_views_share_only_the_intended_storage(layout, owners):
    model = make_decoder(layout=layout)
    views = [streaming._graph_kv_buffers(model, b, "cpu", torch.float16) for b in (256, 448)]
    assert len(model._graph_kv_roots) == owners
    pointers = []
    for bucket, (keys, values) in zip((256, 448), views):
        assert all(item.shape == (1, bucket, 64) for item in keys + values)
        pointers.append(keys[0].untyped_storage().data_ptr())
        for item in keys + values:
            assert item.view(1, bucket, 2, 32).data_ptr() == item.data_ptr()
    assert len(set(pointers)) == owners
    assert streaming._get_bucket_lock(model, 256) is streaming._get_bucket_lock(model, 256)
    assert (streaming._get_bucket_lock(model, 256) is streaming._get_bucket_lock(model, 448)) is (layout == "nested")
    capacity = 1024 if layout == "nested" else 256 + 448
    assert sum(root.numel() * root.element_size() for roots in model._graph_kv_roots.values()
               for root in roots) == 2 * 2 * capacity * 64 * 2


def test_storage_cannot_change_dtype_device_capacity_or_layout():
    model = make_decoder(layout="nested")
    streaming._graph_kv_buffers(model, 256, "cpu", torch.float32)
    for device, dtype in (("cpu", torch.float16), ("cuda:0", torch.float32)):
        with pytest.raises(RuntimeError, match="new decoder"):
            streaming._graph_kv_buffers(model, 256, device, dtype)
    with pytest.raises(RuntimeError, match="new decoder"):
        streaming.apply_cuda_graph_patch(model, kv_cache_layout="bucket")
    model.kv_cache_buckets.append(2048)
    with pytest.raises(RuntimeError, match="new decoder"):
        streaming._graph_kv_buffers(model, 256, "cpu", torch.float32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA CUDA required")
@pytest.mark.parametrize("layout", ["bucket", "nested"])
@pytest.mark.parametrize("mode", ["grad", "no_grad", "inference"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_capture_boundary_keeps_mutable_normal_tensors(monkeypatch, layout, mode, dtype):
    monkeypatch.setattr(streaming, "_CUDA_GRAPH_WARMUP_STEPS", 2)
    with torch.cuda.device(0):
        model = make_decoder("cuda:0", layout=layout, dtype=dtype)
        context = {"grad": torch.enable_grad, "no_grad": torch.no_grad,
                   "inference": torch.inference_mode}[mode]
        with context():
            before = (torch.is_grad_enabled(), torch.is_inference_mode_enabled())
            assert model._warmup_and_capture_bucket(256, "cuda:0")
            assert before == (torch.is_grad_enabled(), torch.is_inference_mode_enabled())
            x = torch.arange(1, 17, device="cuda:0").unsqueeze(0)
            bert = torch.zeros(1, 1024, 16, dtype=dtype, device="cuda:0")
            tokens, _ = model.infer_panel(x, None, x.clone(), bert, early_stop_num=4, top_k=1)
            assert torch.isfinite(tokens).all()
            assert before == (torch.is_grad_enabled(), torch.is_inference_mode_enabled())
        for roots in model._graph_kv_roots.values():
            assert all(not torch.is_inference(root) and not root.requires_grad for root in roots)
        with torch.no_grad():
            model.bucket_static_inputs[256]["k_cache"][0][:, :16].fill_(3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA CUDA required")
@pytest.mark.parametrize("layout", ["bucket", "nested"])
def test_different_prefixes_reuse_prepared_bucket_graphs(monkeypatch, layout):
    monkeypatch.setattr(streaming, "_CUDA_GRAPH_WARMUP_STEPS", 2)
    trace = []
    def observe(logits, _history, **_kwargs):
        trace.append(logits.clone())
        return logits.argmax(-1, keepdim=True), None
    monkeypatch.setattr(upstream, "sample", observe)
    with torch.cuda.device(0), torch.no_grad():
        model = make_decoder("cuda:0", layout=layout)
        assert all(model.precapture_cuda_graph([256, 768, 1024]).values())
        keys = set(model.bucket_graphs)
        assert keys == {256, 768, 1024}
        def unexpected(*_args):
            pytest.fail("prepared bucket attempted a new prefix-length capture")
        monkeypatch.setattr(streaming, "_capture_bucket", unexpected)
        for prefix in (33, 64, 608, 800):
            length = prefix - 16
            x = torch.arange(length, device="cuda:0").remainder(62).unsqueeze(0)
            prompt = torch.ones(1, 16, device="cuda:0", dtype=torch.long)
            bert = torch.zeros(1, 1024, length, device="cuda:0", dtype=torch.float16)
            results = []
            for graph in (False, True):
                trace.clear()
                tokens, count = model.infer_panel(x, None, prompt, bert, early_stop_num=4,
                                                 top_k=1, enable_cuda_graph=graph)
                results.append((tokens.clone(), count, [value.clone() for value in trace]))
            torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
            assert results[0][1] == results[1][1]
            assert len(results[0][2]) == len(results[1][2])
            for expected, actual in zip(results[0][2], results[1][2]):
                torch.testing.assert_close(expected, actual, rtol=2e-3, atol=2e-3)
            assert model.cuda_graph_stats["graph_replay_steps"] > 0
            assert set(model.bucket_graphs) == keys


def test_precapture_empty_is_not_full_and_ignores_duplicate_targets(monkeypatch):
    model = make_decoder()
    model.cuda_graph_enabled = True
    calls = []
    monkeypatch.setattr(streaming, "_warmup_and_capture_bucket",
                        lambda _model, bucket, _device: calls.append(bucket) or True)
    assert model.precapture_cuda_graph([]) == {}
    assert model.precapture_cuda_graph([256, 256, 512]) == {256: True, 512: True}
    assert calls == [256, 512]
    assert model.precapture_cuda_graph([999]) == {999: False}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA CUDA required")
@pytest.mark.parametrize("layout", ["bucket", "nested"])
def test_private_requests_do_not_touch_existing_graph_roots(monkeypatch, layout):
    monkeypatch.setattr(streaming, "_CUDA_GRAPH_WARMUP_STEPS", 2)
    with torch.cuda.device(0), torch.no_grad():
        model = make_decoder("cuda:0", layout=layout)
        assert model._warmup_and_capture_bucket(256, "cuda:0")
        roots = [root for pair in model._graph_kv_roots.values() for root in pair]
        expected = [root.clone() for root in roots]
        for batch, enable_graph in ((1, False), (2, True)):
            x = torch.arange(1, 17, device="cuda:0").unsqueeze(0).repeat(batch, 1)
            prompt = x.clone()
            bert = torch.zeros(batch, 1024, 16, dtype=torch.float16, device="cuda:0")
            tokens, _ = model.infer_panel(x, None, prompt, bert, top_k=1,
                                         early_stop_num=4, enable_cuda_graph=enable_graph)
            assert tokens.shape[0] == batch
            assert model.cuda_graph_stats["graph_replay_steps"] == 0
            for before, after in zip(expected, roots):
                torch.testing.assert_close(before, after, rtol=0, atol=0)
