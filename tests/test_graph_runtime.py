"""Optional real decoder checks, including NVIDIA capture and replay."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
upstream = pytest.importorskip("AR.models.t2s_model")
streaming = pytest.importorskip("aquatts.modeling.t2s_streaming")
from aquatts.inference.presets import resolve_cuda_graph_config  # noqa: E402


def make_decoder(device="cpu", preset="lazy"):
    torch.manual_seed(73)
    model = upstream.Text2SemanticDecoder({"model": {
        "hidden_dim": 64, "embedding_dim": 64, "head": 2, "n_layer": 2,
        "vocab_size": 64, "phoneme_vocab_size": 64, "dropout": 0., "EOS": 63,
    }}).eval().to(device)
    if str(device).startswith("cuda"):
        model.half()
    return streaming.apply_cuda_graph_patch(
        model, graph_config=resolve_cuda_graph_config(preset, {}))


@pytest.mark.parametrize("preset", ["full", "minimal", "lazy", "off"])
def test_patch_records_default_without_changing_cpu_eligibility(preset):
    model = make_decoder(preset=preset)
    assert not model.cuda_graph_available
    assert not model.cuda_graph_enabled


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA CUDA required")
@pytest.mark.parametrize("device", [f"cuda:{i}" for i in range(torch.cuda.device_count())])
@pytest.mark.parametrize("preset,graph_request,replays", [
    ("full", None, True), ("minimal", None, True), ("lazy", None, True),
    ("off", None, False), ("off", True, True), ("full", False, False),
])
def test_real_request_inherits_default_and_replays(monkeypatch, device, preset, graph_request, replays):
    monkeypatch.setattr(streaming, "_CUDA_GRAPH_WARMUP_STEPS", 2)
    with torch.cuda.device(device), torch.no_grad():
        model = make_decoder(device, preset)
        # Full/minimal preparation does not decide whether this request replays.
        if model.cuda_graph_enabled and preset in {"full", "minimal"}:
            assert model._warmup_and_capture_bucket(256, 32, device)
        x = torch.arange(1, 17, device=device).unsqueeze(0)
        prompt = x.clone()
        bert = torch.zeros(1, 1024, 16, device=device, dtype=torch.float16)
        tokens, count = model.infer_panel(x, torch.tensor([16], device=device), prompt,
                                         bert, top_k=1, early_stop_num=4,
                                         enable_cuda_graph=graph_request)
        assert count >= 1 and torch.isfinite(tokens).all()
        assert (model.cuda_graph_stats["graph_replay_steps"] > 0) is replays


def test_precapture_off_and_lazy_do_not_capture(monkeypatch):
    module = pytest.importorskip("aquatts.inferencer")
    for preset in ("off", "lazy"):
        def unexpected(*_args):
            pytest.fail(f"{preset} performed startup capture")
        instance = module.TTSInferencer.__new__(module.TTSInferencer)
        instance._cuda_graph_preset = preset
        instance._cuda_graph_config = resolve_cuda_graph_config(preset, {})
        instance.t2s_model = SimpleNamespace(model=SimpleNamespace(
            cuda_graph_enabled=preset != "off", use_static_kv_cache=True,
            kv_cache_buckets=[256], precapture_cuda_graph=unexpected))
        instance._maybe_precapture_t2s_graph()
