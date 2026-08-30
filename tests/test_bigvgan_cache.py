import os
import pytest
import sys

from aquatts.bigvgan.cuda.load import _sanitize_cache_token


class TestSanitizeCacheToken:
    def test_simple_name(self):
        assert _sanitize_cache_token("NVIDIA GeForce RTX 4070") == "nvidia_geforce_rtx_4070"

    def test_special_chars(self):
        assert _sanitize_cache_token("GPU@#$Name!") == "gpu_name"

    def test_empty_string(self):
        assert _sanitize_cache_token("   ") == "unknown"

    def test_trailing_underscores(self):
        assert _sanitize_cache_token("__test__") == "test"


def test_cache_root_can_be_owned_by_embedding_product(tmp_path, monkeypatch):
    from aquatts.bigvgan.cuda.load import _get_cache_root

    monkeypatch.setenv("BIGVGAN_CACHE_ROOT", str(tmp_path))
    assert _get_cache_root() == tmp_path.resolve()


def test_cuda_package_exports_loader_module():
    from aquatts.bigvgan import cuda

    assert hasattr(cuda.load, "load")


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="_get_gpu_cache_suffix requires torch; tested locally",
)
class TestGetGpuCacheSuffix:
    def test_env_override(self):
        pytest.importorskip("torch")
        from aquatts.bigvgan.cuda.load import _get_gpu_cache_suffix

        os.environ["BIGVGAN_CACHE_ID"] = "my_custom_cache"
        result = _get_gpu_cache_suffix(0)
        assert result == "my_custom_cache"
        del os.environ["BIGVGAN_CACHE_ID"]

    def test_runtime_identity_is_part_of_default_cache_key(self, monkeypatch):
        torch = pytest.importorskip("torch")
        from aquatts.bigvgan.cuda.load import _get_gpu_cache_suffix

        class Props:
            name = "Test GPU"
            major = 8
            minor = 9
            total_memory = 16 * 1024**3

        monkeypatch.delenv("BIGVGAN_CACHE_ID", raising=False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _idx: Props())

        result = _get_gpu_cache_suffix(0)

        assert result.startswith("sm89_16gb_test_gpu_")
        assert f"py{sys.version_info.major}{sys.version_info.minor}" in result
        assert "torch" in result
        assert "cu" in result


def test_cached_extension_does_not_probe_local_cuda_toolkit(monkeypatch):
    pytest.importorskip("torch")
    import importlib

    loader = importlib.import_module("aquatts.bigvgan.cuda.load")

    sentinel = object()
    monkeypatch.setattr(loader, "_get_gpu_cache_suffix", lambda _idx: "test")
    monkeypatch.setattr(loader, "_create_build_dir", lambda _path: None)
    monkeypatch.setattr(loader, "_load_cached_extension", lambda _path: sentinel)
    monkeypatch.setattr(
        loader,
        "_get_cuda_bare_metal_version",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("cached loads must not inspect the CUDA toolkit")
        ),
    )

    assert loader.load() is sentinel
