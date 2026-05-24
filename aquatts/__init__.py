# -*- coding: utf-8 -*-
"""Aqua-TTS: GPU-optimized runtime for GPT-SoVITS v3. / Aqua-TTS：针对 GPT-SoVITS v3 的 GPU 优化运行时。"""

import os
import sys

__version__ = "1.0.0"

# ── Internal path configuration (内部路径配置) ──────────────────────────────────────────
# Ensure vendored GPT_SoVITS overrides take precedence over the main repo.
# The _vendor dir contains our static-KV + CUDA Graph t2s_model.py and the
# BigVGAN CUDA kernel loader — these must be on sys.path BEFORE the main
# GPT-SoVITS repo for the namespace package (pkgutil.extend_path) to work.
# 确保 vendored 的 GPT_SoVITS 覆盖优先于主仓库。_vendor 目录包含我们的静态 KV + CUDA Graph
# t2s_model.py 和 BigVGAN CUDA 内核加载器 — 这些必须在主 GPT-SoVITS 仓库之前加入 sys.path，
# 以确保命名空间包 (pkgutil.extend_path) 正常工作。

# Main GPT-SoVITS repo — required for config/, tools/, and the rest of
# GPT_SoVITS/ modules that we don't vendor.
# 主 GPT-SoVITS 仓库 — config/、tools/ 以及其余未 vendored 的 GPT_SoVITS/ 模块所需。
_GPT_SOVITS_HOME = os.environ.get("GPT_SOVITS_HOME", "")
if _GPT_SOVITS_HOME:
    _gpt_sovits_pkg = os.path.join(_GPT_SOVITS_HOME, "GPT_SoVITS")
    for _p in (_GPT_SOVITS_HOME, _gpt_sovits_pkg):
        if os.path.isdir(_p) and _p not in sys.path:
            sys.path.insert(0, _p)
    # tools.i18n uses os.path.relpath which fails cross-drive on Windows
    # tools.i18n 使用 os.path.relpath，在 Windows 跨驱动器时会失败
    try:
        os.chdir(_GPT_SOVITS_HOME)
    except OSError:
        pass

# Vendored overrides MUST be last (insert(0) → ends up at position 0).
# Order is: _vendor < GPT_SoVITS/ < repo-root, so the vendored t2s_model.py
# and BigVGAN CUDA loader take precedence at import time.
# Vendored 覆盖必须最后添加 (insert(0) → 最终在位置 0)。顺序为：_vendor < GPT_SoVITS/ < repo-root，
# 因此 vendored 的 t2s_model.py 和 BigVGAN CUDA 加载器在导入时具有优先权。
_VENDOR_DIR = os.path.join(os.path.dirname(__file__), "_vendor")
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)

# ── Public API (公共 API) ───────────────────────────────────────────────────────────
# TTSInferencer is imported lazily — import aqua does not trigger
# the full GPT-SoVITS import chain. Use `from aquatts import TTSInferencer`
# or `from aquatts.inferencer import TTSInferencer` to load it.
# TTSInferencer 采用延迟导入 — import aqua 不会触发完整的 GPT-SoVITS 导入链。
# 使用 `from aquatts import TTSInferencer` 或 `from aquatts.inferencer import TTSInferencer` 来加载它。

__all__ = [
    "__version__",
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


def __getattr__(name):
    if name in _LAZY_ATTRS:
        mod_name, attr = _LAZY_ATTRS[name]
        import importlib
        mod = importlib.import_module(mod_name)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
