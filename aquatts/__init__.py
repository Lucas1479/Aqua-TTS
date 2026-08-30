# -*- coding: utf-8 -*-
"""Aqua-TTS: GPU-optimized runtime for GPT-SoVITS v3. / Aqua-TTS：针对 GPT-SoVITS v3 的 GPU 优化运行时。"""

import os
from contextlib import contextmanager

from aquatts.upstream import (
    GPTSoVITSConfigurationError,
    configure_gpt_sovits,
    configured_gpt_sovits_home,
    upstream_t2s_model_path,
)


__version__ = "0.2.1"

# ── Internal path configuration (内部路径配置) ──────────────────────────────────────────
# The bundled namespace bridge only redirects BigVGAN's CUDA extension loader.
# Text2SemanticDecoder itself comes from the configured upstream checkout and
# is optimized in-place after its checkpoint has loaded.
configure_gpt_sovits(require=False)

# ── Public API (公共 API) ───────────────────────────────────────────────────────────
# TTSInferencer is imported lazily — import aqua does not trigger
# the full GPT-SoVITS import chain. Use `from aquatts import TTSInferencer`
# or `from aquatts.inferencer import TTSInferencer` to load it.
# TTSInferencer 采用延迟导入 — import aqua 不会触发完整的 GPT-SoVITS 导入链。
# 使用 `from aquatts import TTSInferencer` 或 `from aquatts.inferencer import TTSInferencer` 来加载它。

__all__ = [
    "__version__",
    "GPTSoVITSConfigurationError",
    "configure_gpt_sovits",
    "configured_gpt_sovits_home",
    "upstream_t2s_model_path",
    "TTSInferencer",
    "VoiceRegistry",
    "Voice",
    "registry_from_env",
    "apply_preset",
    "list_presets",
    "apply_cuda_graph_preset",
    "list_cuda_graph_presets",
    "start_server",
]

_LAZY_ATTRS = {
    "TTSInferencer": ("aquatts.inferencer", "TTSInferencer"),
    "VoiceRegistry": ("aquatts.voice_registry", "VoiceRegistry"),
    "Voice": ("aquatts.voice_registry", "Voice"),
    "registry_from_env": ("aquatts.voice_registry", "registry_from_env"),
    "apply_preset": ("aquatts.inference.presets", "apply_preset"),
    "list_presets": ("aquatts.inference.presets", "list_presets"),
    "apply_cuda_graph_preset": ("aquatts.inference.presets", "apply_cuda_graph_preset"),
    "list_cuda_graph_presets": ("aquatts.inference.presets", "list_cuda_graph_presets"),
    "start_server": ("aquatts.server", "start_server"),
}


@contextmanager
def _gpt_sovits_import_context():
    """Temporarily use the upstream repo as cwd while importing GPT-SoVITS.

    Some upstream modules resolve resources relative to the process cwd and can
    fail across Windows drives. Restoring cwd after import avoids surprising
    callers merely because they imported Aqua-TTS.
    """
    original_cwd = os.getcwd()
    changed_cwd = False
    upstream_home = configured_gpt_sovits_home()
    if upstream_home is not None:
        try:
            os.chdir(upstream_home)
            changed_cwd = True
        except OSError:
            pass
    try:
        yield
        # tools.i18n stores a cwd-relative locale path at import time. Make it
        # absolute before restoring the caller's cwd.
        try:
            from tools.i18n import i18n as _i18n_module

            _i18n_module.I18N_JSON_DIR = os.path.join(
                os.path.dirname(_i18n_module.__file__), "locale"
            )
        except ImportError:
            pass
    finally:
        if changed_cwd:
            os.chdir(original_cwd)


def __getattr__(name):
    if name in _LAZY_ATTRS:
        mod_name, attr = _LAZY_ATTRS[name]
        import importlib

        with _gpt_sovits_import_context():
            mod = importlib.import_module(mod_name)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
