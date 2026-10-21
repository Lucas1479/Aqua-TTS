# -*- coding: utf-8 -*-
"""Named presets for generation quality and CUDA Graph capture strategy. / 用于生成质量和 CUDA Graph 捕获策略的命名预设。"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

# ── Generation presets (生成预设) ──────────────────────────────────────────
# Control per-request quality/speed tradeoffs. / 控制每个请求的质量/速度权衡。

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
    """Return inference params for a named preset, optionally overridden. / 返回指定命名预设的推理参数，可选择覆盖。

    Args: / 参数：
        preset: One of 'fast', 'balanced', 'quality'. / 可选值：'fast'、'balanced'、'quality'。
        overrides: Optional dict of parameter overrides applied on top. / 可选的参数覆盖字典，叠加应用。

    Returns: / 返回值：
        Dict suitable for splatting into TTSInferencer.infer_stream(**params). / 适合展开传入 TTSInferencer.infer_stream(**params) 的字典。

    Raises: / 异常：
        ValueError: If the preset name is unknown. / 如果预设名称未知。
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
    """Return available generation preset names. / 返回可用的生成预设名称。"""
    return sorted(GENERATION_PRESETS)


# ── CUDA Graph presets (CUDA Graph 预设) ──────────────────────────────────
# Control the capture strategy at model-load time. / 控制模型加载时的捕获策略。

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
    """Return CUDA Graph config for a named preset. / 返回指定命名预设的 CUDA Graph 配置。

    Args: / 参数：
        preset: One of 'full', 'minimal', 'lazy', 'off'. / 可选值：'full'、'minimal'、'lazy'、'off'。

    Returns: / 返回值：
        Dict with keys 'enable', 'precapture', 'buckets' (may be None). / 包含 'enable'、'precapture'、'buckets' 键的字典（buckets 可能为 None）。

    Raises: / 异常：
        ValueError: If the preset name is unknown. / 如果预设名称未知。
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
        "buckets": list(cfg["buckets"]) if "buckets" in cfg else None,
    }


def resolve_cuda_graph_config(preset: str, environ=None) -> Dict[str, Any]:
    """Resolve instance defaults once, without changing process environment."""
    config = apply_cuda_graph_preset(preset)
    env = os.environ if environ is None else environ
    for name, field in (("ENABLE_CUDA_GRAPH", "enable"),
                        ("ENABLE_CUDA_GRAPH_PRECAPTURE", "precapture")):
        value = env.get(name, "").strip()
        if value in {"0", "1"}:
            config[field] = value == "1"
    raw = env.get("CUDA_GRAPH_PRECAPTURE_BUCKETS", "").strip()
    if raw:
        buckets = []
        for item in raw.split(","):
            try:
                bucket = int(item.strip())
            except ValueError:
                continue
            if bucket > 0 and bucket not in buckets:
                buckets.append(bucket)
        if not buckets:
            raise ValueError("CUDA_GRAPH_PRECAPTURE_BUCKETS needs positive integer buckets")
        config["buckets"] = buckets
    return config


def resolve_cuda_graph_request(requested, *, default: bool, available: bool,
                               static_kv: bool) -> bool:
    """None inherits the instance; explicit choices still require CUDA/static KV."""
    if requested is not None and not isinstance(requested, bool):
        raise TypeError("enable_cuda_graph must be None, True or False")
    preference = default if requested is None else requested
    return bool(preference and available and static_kv)


def list_cuda_graph_presets() -> List[str]:
    """Return available CUDA Graph preset names. / 返回可用的 CUDA Graph 预设名称。"""
    return sorted(CUDA_GRAPH_PRESETS)
