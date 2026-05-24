# -*- coding: utf-8 -*-
"""Spectralis-TTS: GPU-optimized runtime for GPT-SoVITS v3."""

import os
import sys

__version__ = "1.0.0"

# ── Internal path configuration ──────────────────────────────────────────
# Ensure vendored GPT_SoVITS overrides take precedence over the main repo.
# The _vendor dir contains our static-KV + CUDA Graph t2s_model.py and the
# BigVGAN CUDA kernel loader — these must be on sys.path BEFORE the main
# GPT-SoVITS repo for the namespace package (pkgutil.extend_path) to work.

# Main GPT-SoVITS repo — required for config/, tools/, and the rest of
# GPT_SoVITS/ modules that we don't vendor.
_GPT_SOVITS_HOME = os.environ.get("GPT_SOVITS_HOME", "")
if _GPT_SOVITS_HOME:
    _gpt_sovits_pkg = os.path.join(_GPT_SOVITS_HOME, "GPT_SoVITS")
    for _p in (_GPT_SOVITS_HOME, _gpt_sovits_pkg):
        if os.path.isdir(_p) and _p not in sys.path:
            sys.path.insert(0, _p)
    # tools.i18n uses os.path.relpath which fails cross-drive on Windows
    try:
        os.chdir(_GPT_SOVITS_HOME)
    except OSError:
        pass

# Vendored overrides MUST be last (insert(0) → ends up at position 0).
# Order is: _vendor < GPT_SoVITS/ < repo-root, so the vendored t2s_model.py
# and BigVGAN CUDA loader take precedence at import time.
_VENDOR_DIR = os.path.join(os.path.dirname(__file__), "_vendor")
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)

# ── Public API ───────────────────────────────────────────────────────────
# TTSInferencer is imported lazily — import spectralis does not trigger
# the full GPT-SoVITS import chain. Use `from spectralis import TTSInferencer`
# or `from spectralis.inferencer import TTSInferencer` to load it.

__all__ = [
    "__version__",
    "TTSInferencer",
    "apply_preset",
    "list_presets",
    "apply_cuda_graph_preset",
    "list_cuda_graph_presets",
    "start_server",
]

_LAZY_ATTRS = {
    "TTSInferencer": ("spectralis.inferencer", "TTSInferencer"),
    "apply_preset": ("spectralis.inference.presets", "apply_preset"),
    "list_presets": ("spectralis.inference.presets", "list_presets"),
    "apply_cuda_graph_preset": ("spectralis.inference.presets", "apply_cuda_graph_preset"),
    "list_cuda_graph_presets": ("spectralis.inference.presets", "list_cuda_graph_presets"),
    "start_server": ("spectralis.server", "start_server"),
}


def __getattr__(name):
    if name in _LAZY_ATTRS:
        mod_name, attr = _LAZY_ATTRS[name]
        import importlib
        mod = importlib.import_module(mod_name)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
