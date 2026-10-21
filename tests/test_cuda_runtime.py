"""Configuration and early device selection do not require installed Torch."""
from types import SimpleNamespace

import pytest

from aquatts.inference.cuda_runtime import resolve_cuda_device, resolve_vocoder_warmup


def fake_torch(*, available=True, current=1, count=2, hip=None):
    def device(value, index=None):
        if hasattr(value, "type"):
            return value
        parts = value.split(":")
        return SimpleNamespace(type=parts[0], index=index if index is not None else
                               (int(parts[1]) if len(parts) == 2 else None))
    return SimpleNamespace(device=device, version=SimpleNamespace(hip=hip),
                           cuda=SimpleNamespace(is_available=lambda: available,
                                                current_device=lambda: current,
                                                device_count=lambda: count))


def test_unindexed_cuda_uses_current_visible_device():
    backend = fake_torch()
    assert resolve_cuda_device("cuda", torch_module=backend).index == 1
    assert resolve_cuda_device("cuda:0", torch_module=backend).index == 0
    assert resolve_cuda_device("cpu", torch_module=backend).type == "cpu"


@pytest.mark.parametrize("available,hip", [(False, None), (True, "6.0")])
def test_unavailable_nvidia_cuda_is_explicit(available, hip):
    with pytest.raises(RuntimeError, match="NVIDIA CUDA"):
        resolve_cuda_device("cuda", torch_module=fake_torch(available=available, hip=hip))


@pytest.mark.parametrize("requested", ["cuda:2", "cuda:-1"])
def test_out_of_visible_range_is_rejected(requested):
    with pytest.raises(ValueError, match="visible device count"):
        resolve_cuda_device(requested, torch_module=fake_torch())


def test_warmup_defaults_overrides_and_disable():
    assert resolve_vocoder_warmup(environ={}) == ((40, 70, 128), 5)
    env = {"AQUATTS_BIGVGAN_WARMUP_SHAPES": "9,23,9,46", "AQUATTS_BIGVGAN_WARMUP_REPEATS": "2"}
    assert resolve_vocoder_warmup(environ=env) == ((9, 23, 46), 2)
    assert resolve_vocoder_warmup([], 1, environ=env) == ((), 1)
    assert resolve_vocoder_warmup([40], 3, environ={key: "invalid" for key in env}) == ((40,), 3)
    assert resolve_vocoder_warmup(environ={"AQUATTS_BIGVGAN_WARMUP_SHAPES": ""}) == ((), 5)


@pytest.mark.parametrize("shapes,repeats", [
    ([0], 5), ([-1], 5), ([1.5], 5), ([True], 5), (False, 5), ([9], 0),
    ([9], True), ([9], "1.5"), ([4097], 1), (list(range(1, 18)), 1),
    ([9], 21), ([4096, 4095], 20), ("bad", 1),
])
def test_invalid_or_unbounded_warmup_is_rejected(shapes, repeats):
    with pytest.raises(ValueError):
        resolve_vocoder_warmup(shapes, repeats, environ={})
