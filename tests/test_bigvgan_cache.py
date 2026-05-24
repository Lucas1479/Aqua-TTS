import os
import pytest

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


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="_get_gpu_cache_suffix requires torch; tested locally",
)
class TestGetGpuCacheSuffix:
    def test_env_override(self):
        torch = pytest.importorskip("torch")
        from aquatts.bigvgan.cuda.load import _get_gpu_cache_suffix

        os.environ["BIGVGAN_CACHE_ID"] = "my_custom_cache"
        result = _get_gpu_cache_suffix(0)
        assert result == "my_custom_cache"
        del os.environ["BIGVGAN_CACHE_ID"]
