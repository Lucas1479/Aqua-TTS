# -*- coding: utf-8 -*-
"""Named presets for generation quality and CUDA Graph capture strategy."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# ── Generation presets ────────────────────────────────────────────────────
# Control per-request quality/speed tradeoffs.

GENERATION_PRESETS: Dict[str, Dict[str, Any]] = {
    "fast": {
        "speed": 1.3,
        "sample_steps": 4,
        "top_k": 3,
        "top_p": 1.0,
        "temperature": 0.6,
    },
    "balanced": {
        "speed": 1.1,
        "sample_steps": 4,
        "top_k": 5,
        "top_p": 1.0,
        "temperature": 0.6,
    },
    "quality": {
        "speed": 1.0,
        "sample_steps": 16,
        "top_k": 8,
        "top_p": 0.8,
        "temperature": 0.8,
    },
}


def apply_preset(
    preset: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return inference params for a named preset, optionally overridden.

    Args:
        preset: One of 'fast', 'balanced', 'quality'.
        overrides: Optional dict of parameter overrides applied on top.

    Returns:
        Dict suitable for splatting into TTSInferencer.infer_stream(**params).

    Raises:
        ValueError: If the preset name is unknown.
    """
    if preset not in GENERATION_PRESETS:
        raise ValueError(
            f"Unknown preset {preset!r}. Choose from: {list(GENERATION_PRESETS)}"
        )
    params = dict(GENERATION_PRESETS[preset])
    if overrides:
        params.update(overrides)
    return params


def list_presets() -> List[str]:
    """Return available generation preset names."""
    return sorted(GENERATION_PRESETS)


# ── CUDA Graph presets ────────────────────────────────────────────────────
# Control the capture strategy at model-load time.

CUDA_GRAPH_PRESETS: Dict[str, Dict[str, Any]] = {
    "full": {
        "enable": True,
        "precapture": True,
        "buckets": [128, 256, 448, 512, 768, 1024],
    },
    "minimal": {
        "enable": True,
        "precapture": True,
        "buckets": [256, 512, 1024],
    },
    "lazy": {
        "enable": True,
        "precapture": False,
    },
    "off": {
        "enable": False,
    },
}


def apply_cuda_graph_preset(
    preset: str,
) -> Dict[str, Any]:
    """Return CUDA Graph config for a named preset.

    Args:
        preset: One of 'full', 'minimal', 'lazy', 'off'.

    Returns:
        Dict with keys 'enable', 'precapture', 'buckets' (may be None).

    Raises:
        ValueError: If the preset name is unknown.
    """
    if preset not in CUDA_GRAPH_PRESETS:
        raise ValueError(
            f"Unknown CUDA Graph preset {preset!r}. "
            f"Choose from: {list(CUDA_GRAPH_PRESETS)}"
        )
    cfg = CUDA_GRAPH_PRESETS[preset]
    return {
        "enable": cfg.get("enable", True),
        "precapture": cfg.get("precapture", False),
        "buckets": cfg.get("buckets"),
    }


def list_cuda_graph_presets() -> List[str]:
    """Return available CUDA Graph preset names."""
    return sorted(CUDA_GRAPH_PRESETS)
