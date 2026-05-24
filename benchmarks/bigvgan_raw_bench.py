# -*- coding: utf-8 -*-
"""Standalone BigVGAN kernel timing — no wrapper, no empty_cache, proper warmup.

Loads BigVGAN directly (bypasses TTSInferencer) to avoid warmup path pollution.

Usage:
    python benchmarks/bigvgan_raw_bench.py

Uses the same BigVGAN loading logic as aqua.inferencer._load_bigvgan_model().
"""
from __future__ import annotations

import os, statistics, sys, time

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_REPO = os.environ.get("GPT_SOVITS_HOME")
if not MAIN_REPO:
    sys.exit("GPT_SOVITS_HOME must be set to your GPT-SoVITS repo root")
MAIN_GPT_SOVITS = os.path.join(MAIN_REPO, "GPT_SoVITS")

sys.path.insert(0, os.path.join(ROOT, "Aqua", "_vendor"))
sys.path.insert(0, ROOT)
sys.path.insert(0, MAIN_GPT_SOVITS)
sys.path.insert(0, MAIN_REPO)
os.chdir(MAIN_REPO)

import torch

MEL_T_SIZES = [70, 128, 298, 598]
WARMUP = 10
MEASURE = 20


def load_bigvgan():
    """Load BigVGAN model exactly as aqua.inferencer._load_bigvgan_model() does."""
    import pathlib

    bigvgan_path = os.path.join(
        MAIN_REPO, "GPT_SoVITS", "pretrained_models",
        "models--nvidia--bigvgan_v2_24khz_100band_256x",
    )

    # Determine cache suffix
    props = torch.cuda.get_device_properties(0)
    import re as _re
    name = _re.sub(r"[^A-Za-z0-9_.-]+", "_", props.name.lower()).strip("_") or "unknown"
    mem_gb = int(round(props.total_memory / (1024 ** 3)))
    cache_suffix = f"sm{props.major}{props.minor}_{mem_gb}gb_{name}"

    cuda_pyd = (
        pathlib.Path(ROOT)
        / "Aqua/_vendor/GPT_SoVITS/BigVGAN/alias_free_activation/cuda"
        / f"build_{cache_suffix}"
        / "anti_alias_activation_cuda.pyd"
    )

    use_cuda_kernel = cuda_pyd.exists()

    print(f"BigVGAN path: {bigvgan_path}")
    print(f"Cache suffix: {cache_suffix}")
    print(f"CUDA .pyd exists: {cuda_pyd.exists()}")

    from GPT_SoVITS.BigVGAN import bigvgan

    with torch.cuda.device(0):
        model = bigvgan.BigVGAN.from_pretrained(bigvgan_path, use_cuda_kernel=use_cuda_kernel)

    model.remove_weight_norm()
    model = model.eval()
    model = model.half().to("cuda")

    return model


def main():
    print(f"Loading BigVGAN directly...")
    t0 = time.perf_counter()
    model = load_bigvgan()
    print(f"Loaded in {time.perf_counter() - t0:.1f}s")

    use_cuda = model.h.get("use_cuda_kernel", False)
    print(f"  use_cuda_kernel = {use_cuda}")
    print(f"  dtype = {next(model.parameters()).dtype}")
    print(f"  num_mels = {model.h.num_mels}")

    num_mels = model.h.num_mels  # 100
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Determine precision label
    precision = "FP16" if dtype == torch.float16 else str(dtype)

    print(f"\n{'='*70}")
    print(f"  BigVGAN raw forward — CUDA={'ON' if use_cuda else 'OFF'}  {precision}")
    print(f"  Each size: {WARMUP} warmup + {MEASURE} measured (sync before+after)")
    print(f"{'='*70}")

    for mel_t in MEL_T_SIZES:
        # Warmup
        mel = torch.randn(1, num_mels, mel_t, device=device, dtype=dtype)
        for _ in range(WARMUP):
            _ = model(mel)

        # Measure
        times = []
        for _ in range(MEASURE):
            mel = torch.randn(1, num_mels, mel_t, device=device, dtype=dtype)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(mel)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000.0
            times.append(dt)

        median_ms = statistics.median(times)
        min_ms = min(times)
        max_ms = max(times)
        print(f"  mel_T={mel_t:>4d}  median={median_ms:6.1f}ms  "
              f"min={min_ms:6.1f}ms  max={max_ms:6.1f}ms")

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
