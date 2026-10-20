"""Exercise the actual upstream cached CFM/DiT with small deterministic weights."""
from unittest.mock import Mock
import random

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("aquatts.inferencer")
from GPT_SoVITS.module.models import CFM  # noqa: E402
from GPT_SoVITS.f5_tts.model.backbones.dit import DiT  # noqa: E402
from aquatts.inference.cfm import UnpaddedCFM  # noqa: E402


def small_cfm(device="cpu", dtype=None):
    torch.manual_seed(11)
    dit = DiT(dim=32, depth=2, heads=2, dim_head=16, mel_dim=4,
              text_dim=8, conv_layers=1, dropout=0)
    return CFM(4, dit).to(device=device, dtype=dtype or torch.float32).eval()


def inputs(model, batch=1, length=17, prompt=5):
    parameter = next(model.parameters())
    mu = torch.randn(batch, length, 8, device=parameter.device, dtype=parameter.dtype)
    ref = torch.randn(batch, 4, prompt, device=parameter.device, dtype=parameter.dtype)
    lengths = torch.tensor([length] * batch, device=parameter.device)
    return mu, lengths, ref


@pytest.mark.parametrize("steps", [4, 16, 32])
@pytest.mark.parametrize("cfg", [0., 1.])
def test_same_noise_preserves_output_and_conditioner_cache(steps, cfg, monkeypatch):
    model = small_cfm()
    mu, lengths, prompt = inputs(model)
    original_forward = DiT.forward
    mask = Mock(wraps=original_forward.__globals__["sequence_mask"])
    monkeypatch.setitem(original_forward.__globals__, "sequence_mask", mask)
    adapter = UnpaddedCFM(model)
    counts = {"text": 0, "dt": 0}
    def count(name):
        def hook(*_):
            counts[name] += 1
        return hook
    model.estimator.text_embed.register_forward_hook(count("text"))
    model.estimator.d_embed.register_forward_hook(count("dt"))
    torch.manual_seed(17)
    baseline = model.inference(mu, lengths, prompt, steps, inference_cfg_rate=cfg)
    assert mask.call_count == steps * (2 if cfg else 1)
    expected_counts = dict(counts)
    assert expected_counts == {"text": 2 if cfg else 1, "dt": 1}
    counts.update(text=0, dt=0)
    mask.reset_mock()
    torch.manual_seed(17)
    actual = adapter(mu, lengths, prompt, steps, unpadded_length=mu.size(1), inference_cfg_rate=cfg)
    torch.testing.assert_close(actual, baseline, rtol=2e-5, atol=2e-5)
    assert counts == expected_counts and mask.call_count == 0
    assert DiT.forward is original_forward
    assert actual.shape == baseline.shape and torch.isfinite(actual).all()
    assert adapter.last_stats["path"] == "unpadded"


@pytest.mark.parametrize("batch,valid,host", [(1, 15, 15), (2, 17, 17), (1, 17, None)])
@pytest.mark.parametrize("device,dtype", [
    ("cpu", torch.float32),
    pytest.param("cuda:0", torch.float32, marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")),
    pytest.param("cuda:0", torch.float16, marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")),
])
def test_padding_batch_and_unknown_metadata_build_one_mask(batch, valid, host, device, dtype, monkeypatch):
    model = small_cfm(device, dtype)
    mu, lengths, prompt = inputs(model, batch=batch)
    lengths[0] = valid
    mask = Mock(wraps=DiT.forward.__globals__["sequence_mask"])
    monkeypatch.setitem(DiT.forward.__globals__, "sequence_mask", mask)
    adapter = UnpaddedCFM(model)
    torch.manual_seed(21)
    baseline = model.inference(mu, lengths, prompt, 4, inference_cfg_rate=1)
    mask.reset_mock()
    torch.manual_seed(21)
    actual = adapter(mu, lengths, prompt, 4, unpadded_length=host, inference_cfg_rate=1)
    torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
    assert mask.call_count == 1 and adapter.last_stats["path"] == "masked"


def test_disabled_call_keeps_original_estimator_and_v2_is_rejected():
    from types import SimpleNamespace
    from aquatts.inferencer import TTSInferencer
    model = small_cfm()
    mu, lengths, prompt = inputs(model)
    original = model.estimator.forward.__func__
    value = TTSInferencer.__new__(TTSInferencer)
    value.vq_model = SimpleNamespace(cfm=model)
    value.cfm_unpadded = False
    value._run_cfm(mu, lengths, prompt, 4)
    assert model.estimator.forward.__func__ is original
    value.model_version = "v2"
    with pytest.raises(ValueError, match="V3"):
        value._prepare_cfm_adapter()


def test_training_and_exception_cleanup_preserve_original_mask(monkeypatch):
    model = small_cfm()
    mu, lengths, prompt = inputs(model)
    adapter = UnpaddedCFM(model)
    original_inference = model.inference
    def fail(*_args, **_kwargs):
        raise RuntimeError("injected diffusion failure")
    monkeypatch.setattr(model, "inference", fail)
    with pytest.raises(RuntimeError, match="injected"):
        adapter(mu, lengths, prompt, 4, unpadded_length=17)
    assert adapter._prepared.get() is None
    monkeypatch.setattr(model, "inference", original_inference)
    masks = []
    model.estimator.transformer_blocks[0].register_forward_pre_hook(
        lambda _module, _args, kwargs: masks.append(kwargs["mask"]), with_kwargs=True)
    x = torch.randn(1, 4, 17)
    result = model.estimator(x, torch.zeros_like(x), lengths, torch.ones(1), torch.ones(1),
                             mu.transpose(1, 2), infer=False)
    assert isinstance(result, torch.Tensor) and result.requires_grad
    result.sum().backward()
    assert masks and masks[0] is not None
    assert model.estimator.proj_out.weight.grad is not None


def test_full_training_loss_and_state_dict_are_unchanged():
    model = small_cfm()
    mu, lengths, _ = inputs(model)
    x = torch.randn(1, 4, 17)
    prompt_lengths = torch.tensor([5])
    keys = list(model.state_dict())
    torch.manual_seed(13)
    random.seed(13)
    baseline = model(x, lengths, prompt_lengths, mu.transpose(1, 2), False)
    UnpaddedCFM(model)
    torch.manual_seed(13)
    random.seed(13)
    actual = model(x, lengths, prompt_lengths, mu.transpose(1, 2), False)
    torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
    actual.backward()
    assert list(model.state_dict()) == keys


def test_cache_disabled_keeps_upstream_recomputation():
    model = small_cfm()
    model.use_conditioner_cache = False
    mu, lengths, prompt = inputs(model)
    adapter = UnpaddedCFM(model)
    counts = []
    model.estimator.text_embed.register_forward_hook(lambda *_: counts.append("text"))
    model.estimator.d_embed.register_forward_hook(lambda *_: counts.append("dt"))
    adapter(mu, lengths, prompt, 4, unpadded_length=17, inference_cfg_rate=1)
    assert counts.count("text") == counts.count("dt") == 8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("steps", [4, 16, 32])
@pytest.mark.parametrize("cfg", [0., 1.])
def test_cuda_same_noise(dtype, steps, cfg):
    model = small_cfm("cuda:0", dtype)
    mu, lengths, prompt = inputs(model, length=31, prompt=17)
    adapter = UnpaddedCFM(model)
    torch.manual_seed(19)
    baseline = model.inference(mu, lengths, prompt, steps, inference_cfg_rate=cfg)
    torch.manual_seed(19)
    actual = adapter(mu, lengths, prompt, steps, unpadded_length=31, inference_cfg_rate=cfg)
    # Fixed before measurement; this gate is for small models, not audio quality.
    tolerance = 3e-3 if dtype == torch.float16 else 3e-5
    torch.testing.assert_close(actual, baseline, rtol=tolerance, atol=tolerance)
    assert torch.isfinite(actual).all()
