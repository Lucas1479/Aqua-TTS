"""Selected-device calls, bounded warmup, RNG preservation and kernel fallback."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
module = pytest.importorskip("aquatts.inferencer")
from aquatts.inference.cuda_runtime import warmup_vocoder  # noqa: E402


class TinyVocoder(torch.nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1, device=device, dtype=torch.float16))
        self.h = SimpleNamespace(num_mels=7)
        self.calls = []
        self.fail = False

    def remove_weight_norm(self):
        pass

    def forward(self, mel):
        assert torch.is_inference_mode_enabled()
        assert mel.device == self.weight.device and mel.dtype == self.weight.dtype
        if mel.is_cuda:
            assert torch.cuda.current_device() == mel.device.index
        self.calls.append(tuple(mel.shape))
        # Make RNG preservation observable even if a future vocoder uses randomness.
        torch.rand(1)
        torch.rand_like(mel)
        if self.fail:
            raise RuntimeError("injected vocoder failure")
        return mel[:, :1]


def test_invalid_device_fails_before_model_loading(monkeypatch):
    def fail(*_args, **_kwargs):
        pytest.fail("invalid CUDA selection reached model loading")
    monkeypatch.setattr(module.TTSInferencer, "_configure_fast_langdetect", fail)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match="visible device count"):
        module.TTSInferencer(device="cuda:1")


def test_disabled_and_cpu_warmup_do_not_call_model():
    model = TinyVocoder()
    assert warmup_vocoder(model, [], 5)["status"] == "disabled"
    assert warmup_vocoder(model, [9], 5)["status"] == "skipped"
    assert model.calls == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA CUDA required")
@pytest.mark.parametrize("fail", [False, True])
def test_warmup_obeys_model_shape_device_and_preserves_rng(fail):
    model = TinyVocoder("cuda:0")
    model.fail = fail
    cpu_before, cuda_before = torch.get_rng_state(), torch.cuda.get_rng_state(0)
    stats = warmup_vocoder(model, [9, 23], 2)
    assert torch.equal(cpu_before, torch.get_rng_state())
    assert torch.equal(cuda_before, torch.cuda.get_rng_state(0))
    assert stats["status"] == ("failed" if fail else "completed")
    assert stats["device"] == "cuda:0" and stats["mel_channels"] == 7
    assert stats["forward_calls"] == (0 if fail else 4)
    assert model.calls == ([(1, 7, 9)] if fail else [(1, 7, 9)] * 2 + [(1, 7, 23)] * 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA CUDA required")
def test_regular_vocoder_call_counts_shapes_without_synchronize(monkeypatch):
    value = module.TTSInferencer.__new__(module.TTSInferencer)
    value.bigvgan_model = TinyVocoder("cuda:0")
    value.vocoder_shape_counts = {}
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: pytest.fail("normal vocoder synchronized"))
    mel = torch.zeros(1, 7, 23, device="cuda:0", dtype=torch.float16)
    assert value._run_bigvgan(mel).shape == (1, 1, 23)
    value._run_bigvgan(mel)
    assert value.vocoder_shape_counts == {23: 2}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA CUDA required")
def test_kernel_failure_falls_back_on_selected_device(monkeypatch):
    value = module.TTSInferencer.__new__(module.TTSInferencer)
    value.device, value._tts_device_idx, value.is_half = torch.device("cuda:0"), 0, True
    value.model_version, value.bigvgan_path = "v3", "unused"
    value._bigvgan_warmup_shapes, value._bigvgan_warmup_repeats = (), 1
    calls = []
    def load(_path, *, use_cuda_kernel):
        calls.append((use_cuda_kernel, torch.cuda.current_device()))
        if use_cuda_kernel:
            raise RuntimeError("injected compiler failure")
        return TinyVocoder()
    monkeypatch.setenv("BIGVGAN_USE_CUDA_KERNEL", "1")
    monkeypatch.setattr(module.bigvgan.BigVGAN, "from_pretrained", load)
    value._load_bigvgan_model()
    assert calls == [(True, 0), (False, 0)]
    assert value.bigvgan_kernel_mode == "pytorch"
    assert "compiler failure" in value.bigvgan_kernel_fallback_reason
    assert next(value.bigvgan_model.parameters()).device == torch.device("cuda:0")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA CUDA required")
def test_auxiliary_loader_uses_explicit_selected_device(monkeypatch):
    from aquatts.inference.streaming import load_bigvgan
    calls = []
    def load(_path, *, use_cuda_kernel):
        calls.append(torch.cuda.current_device())
        return TinyVocoder()
    monkeypatch.setattr(module.bigvgan.BigVGAN, "from_pretrained", load)
    model = load_bigvgan("unused", device="cuda:0")
    assert calls == [0] and next(model.parameters()).device == torch.device("cuda:0")
