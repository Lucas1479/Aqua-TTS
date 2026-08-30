"""Compatibility bridge to Aqua's canonical BigVGAN extension loader."""

from aquatts.bigvgan.cuda.load import (  # noqa: F401
    _create_build_dir,
    _find_vcvars64,
    _get_cache_root,
    _get_cuda_bare_metal_version,
    _get_gpu_cache_suffix,
    _get_tts_device_index,
    _sanitize_cache_token,
    load,
)


__all__ = ["load"]
