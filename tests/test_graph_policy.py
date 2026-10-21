"""Graph defaults and request overrides work without the optional model runtime."""
import pytest

from aquatts.inference.params import get_sovits_params
from aquatts.inference.presets import (
    resolve_cuda_graph_config,
    resolve_cuda_graph_request,
)


@pytest.mark.parametrize("preset", ["full", "minimal", "lazy", "off"])
@pytest.mark.parametrize("override", [None, "0", "1"])
def test_environment_overrides_instance_default(preset, override):
    env = {} if override is None else {"ENABLE_CUDA_GRAPH": override}
    config = resolve_cuda_graph_config(preset, env)
    expected = preset != "off" if override is None else override == "1"
    assert config["enable"] is expected
    assert config["precapture"] is (preset in {"full", "minimal"})


@pytest.mark.parametrize("default", [False, True])
@pytest.mark.parametrize("requested", [None, False, True])
@pytest.mark.parametrize("available,static_kv", [(True, True), (False, True), (True, False)])
def test_request_override_cannot_bypass_capability(default, requested, available, static_kv):
    expected = (default if requested is None else requested) and available and static_kv
    assert resolve_cuda_graph_request(requested, default=default,
                                      available=available, static_kv=static_kv) is expected


def test_explicit_off_can_lazily_opt_in():
    default = resolve_cuda_graph_config("off", {})["enable"]
    assert not resolve_cuda_graph_request(None, default=default, available=True, static_kv=True)
    assert resolve_cuda_graph_request(True, default=default, available=True, static_kv=True)


def test_precapture_overrides_are_normalized_without_mutating_presets():
    env = {"ENABLE_CUDA_GRAPH_PRECAPTURE": "0",
           "CUDA_GRAPH_PRECAPTURE_BUCKETS": "512,256,512,invalid,-1,0"}
    config = resolve_cuda_graph_config("full", env)
    assert not config["precapture"]
    assert config["buckets"] == [512, 256]
    config["buckets"].clear()
    assert resolve_cuda_graph_config("full", {})["buckets"] == [128, 256, 448, 512, 768, 1024]


@pytest.mark.parametrize("raw", ["garbage", "0,-1", ",,"])
def test_invalid_bucket_override_is_visible(raw):
    with pytest.raises(ValueError, match="positive integer"):
        resolve_cuda_graph_config("full", {"CUDA_GRAPH_PRECAPTURE_BUCKETS": raw})


@pytest.mark.parametrize("first,text", [(True, "hello"), (False, "hello"), (False, "x" * 100)])
@pytest.mark.parametrize("value,expected", [(None, None), ("0", False), ("1", True)])
def test_text_parameter_helper_preserves_inheritance(monkeypatch, first, text, value, expected):
    monkeypatch.delenv("ENABLE_CUDA_GRAPH", raising=False)
    if value is not None:
        monkeypatch.setenv("ENABLE_CUDA_GRAPH", value)
    assert get_sovits_params(text, first)["enable_cuda_graph"] is expected


def test_invalid_request_type_is_not_silently_truthy():
    with pytest.raises(TypeError, match="enable_cuda_graph"):
        resolve_cuda_graph_request("false", default=True, available=True, static_kv=True)
