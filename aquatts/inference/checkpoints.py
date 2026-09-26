"""Architecture selection using upstream checkpoint metadata, never filenames."""

SUPPORTED_MODELS = frozenset({"v1", "v2", "v2Pro", "v2ProPlus", "v3"})
PRO_MODELS = frozenset({"v2Pro", "v2ProPlus"})


def detect_sovits_version(path, *, reader=None):
    """Return (text-symbol version, model architecture, LoRA flag).

    Upstream recognizes official base weights by their prefix digest, new
    fine-tunes by their two-byte header, and legacy weights by their size.
    Unknown headers and unsupported architectures must not become ordinary v2.
    """
    if reader is None:
        from GPT_SoVITS.process_ckpt import get_sovits_version_from_path_fast

        reader = get_sovits_version_from_path_fast
    try:
        symbols, model, is_lora = reader(str(path))
    except KeyError as exc:
        raise ValueError(f"Unrecognized SoVITS checkpoint header: {path}") from exc
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported SoVITS architecture {model!r}: {path}")
    return symbols, model, bool(is_lora)
