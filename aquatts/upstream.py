"""Configure the upstream GPT-SoVITS source tree used by Aqua-TTS.

Aqua-TTS is an optimization layer, not a fork of GPT-SoVITS. The upstream
repository supplies model definitions, text processing, and checkpoint
loading; Aqua supplies narrowly scoped T2S and BigVGAN runtime overrides.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


class GPTSoVITSConfigurationError(RuntimeError):
    """The configured upstream GPT-SoVITS tree is missing or incompatible."""


_PACKAGE_DIR = Path(__file__).resolve().parent
_VENDOR_DIR = _PACKAGE_DIR / "_vendor"
_VENDOR_GPT_SOVITS_DIR = _VENDOR_DIR / "GPT_SoVITS"


def _prepend_paths(paths: list[Path]) -> None:
    resolved = [str(path) for path in paths if path.is_dir()]
    for path in resolved:
        while path in sys.path:
            sys.path.remove(path)
    sys.path[:0] = resolved


def _resolve_layout(value: str | os.PathLike[str]) -> tuple[Path, Path]:
    supplied = Path(value).expanduser().resolve()
    repo_root = supplied
    package_root = supplied / "GPT_SoVITS"

    if (supplied / "AR" / "models" / "t2s_model.py").is_file():
        package_root = supplied
        repo_root = supplied.parent

    marker = package_root / "AR" / "models" / "t2s_model.py"
    if not marker.is_file():
        raise GPTSoVITSConfigurationError(
            "GPT_SOVITS_HOME must point to an upstream GPT-SoVITS checkout "
            f"(missing {marker})"
        )
    return repo_root, package_root


def configure_gpt_sovits(
    home: str | os.PathLike[str] | None = None,
    *,
    require: bool = False,
) -> Path | None:
    """Configure import paths for one upstream GPT-SoVITS checkout.

    ``home`` may be either the repository root or its ``GPT_SoVITS`` package
    directory. Call this before importing :class:`aquatts.TTSInferencer`.
    Aqua's small BigVGAN bridge remains first on ``sys.path`` while the T2S
    model itself is loaded from the selected upstream checkout.
    """

    raw = str(home or os.environ.get("GPT_SOVITS_HOME", "")).strip()
    if not raw:
        _prepend_paths([_VENDOR_DIR, _VENDOR_GPT_SOVITS_DIR])
        if require:
            raise GPTSoVITSConfigurationError(
                "GPT_SOVITS_HOME is required for TTS inference"
            )
        return None

    repo_root, package_root = _resolve_layout(raw)
    os.environ["GPT_SOVITS_HOME"] = str(repo_root)
    _prepend_paths(
        [
            _VENDOR_DIR,
            _VENDOR_GPT_SOVITS_DIR,
            repo_root,
            package_root,
        ]
    )
    return repo_root


def configured_gpt_sovits_home() -> Path | None:
    """Return the validated configured upstream repository root, if any."""

    raw = os.environ.get("GPT_SOVITS_HOME", "").strip()
    if not raw:
        return None
    try:
        repo_root, _ = _resolve_layout(raw)
    except GPTSoVITSConfigurationError:
        return None
    return repo_root


def upstream_t2s_model_path() -> Path | None:
    """Return the selected upstream T2S model source path."""

    home = configured_gpt_sovits_home()
    if home is None:
        return None
    return home / "GPT_SoVITS" / "AR" / "models" / "t2s_model.py"
