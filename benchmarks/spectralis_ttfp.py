# -*- coding: utf-8 -*-
"""Spectralis TTFP (Time-To-First-Packet) benchmark.

Measures the optimized inference path with static KV cache, CUDA Graph,
and pre-compiled BigVGAN CUDA kernel.

Usage:
    python benchmarks/spectralis_ttfp.py \
        --gpt-model GPT_weights_v3/xxx-e15.ckpt \
        --sovits-model SoVITS_weights_v3/xxx_e2_s174_l32.pth \
        --ref-audio "reference audio/kurisu_reference2.wav" \
        --ref-text "your reference transcript"
"""
from __future__ import annotations

import argparse, os, statistics, sys, time

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "GPT_SoVITS"))

import torch

BENCH_TEXTS = [
    ("short",  "実験？"),
    ("medium", "何の実験だか気になるけど、まあいいわ。"),
    ("long",   "Paxosが提案ベースでメッセージフローが複雑なのに対して、"
               "Raftは明確なリーダー選出とログ複製のフェーズに分けられているわ。"),
]
WARMUP_TEXTS = ["これはテストです。", "何の実験だか気になるけど、まあいいわ。"]
REPEATS = 5


def main():
    parser = argparse.ArgumentParser(description="Spectralis TTFP benchmark")
    parser.add_argument("--gpt-model", required=True)
    parser.add_argument("--sovits-model", required=True)
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--ref-text", required=True)
    parser.add_argument("--ref-lang", default="日文")
    parser.add_argument("--text-lang", default="日文")
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument("--no-static-kv", action="store_true")
    parser.add_argument("--no-bigvgan-kernel", action="store_true")
    args = parser.parse_args()

    # Configure optimizations via env vars before importing TTSInferencer
    if args.no_cuda_graph:
        os.environ["ENABLE_CUDA_GRAPH"] = "0"
    else:
        os.environ["ENABLE_CUDA_GRAPH"] = "1"

    os.environ["ENABLE_CUDA_GRAPH_PRECAPTURE"] = "1" if not args.no_cuda_graph else "0"

    from local_tts_infer import TTSInferencer

    print(f"\n{'='*80}")
    print(f"  Spectralis TTFP Benchmark")
    print(f"{'='*80}")
    flags = []
    if not args.no_cuda_graph: flags.append("CUDA Graph")
    if not args.no_static_kv: flags.append("Static KV")
    if not args.no_bigvgan_kernel: flags.append("BigVGAN Kernel")
    print(f"  Optimizations: {', '.join(flags) if flags else 'ALL OFF'}")
    print(f"  GPT: {args.gpt_model}")
    print(f"  SoVITS: {args.sovits_model}")

    print("[bench] Loading TTS pipeline...")
    t0 = time.perf_counter()
    inferencer = TTSInferencer(
        device="cuda",
        gpt_path=args.gpt_model,
        sovits_path=args.sovits_model,
    )
    # BigVGAN kernel toggle (set before first inference)
    if args.no_bigvgan_kernel and hasattr(inferencer, 'bigvgan_model'):
        inferencer.bigvgan_model.use_cuda_kernel = False
    print(f"[bench] Loaded in {time.perf_counter() - t0:.2f}s")

    def _measure(text: str) -> dict:
        t0 = time.perf_counter()
        first_chunk_ms = None
        chunk_count = 0

        for sr, chunk in inferencer.infer_stream(
            text=text,
            ref_audio_path=args.ref_audio,
            prompt_text=args.ref_text,
            text_language=args.text_lang,
            prompt_language=args.ref_lang,
            how_to_cut="不切",
            top_k=5, top_p=1, temperature=0.6,
            speed=1.1, sample_steps=4,
            enable_cuda_graph=not args.no_cuda_graph,
            enable_static_kv=not args.no_static_kv,
        ):
            if chunk is None or len(chunk) == 0:
                continue
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if first_chunk_ms is None:
                first_chunk_ms = elapsed_ms
            chunk_count += 1

        total_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "first_chunk_ms": float(first_chunk_ms or total_ms),
            "total_ms": float(total_ms),
            "chunk_count": chunk_count,
        }

    print("[bench] Warmup...")
    for wt in WARMUP_TEXTS:
        _measure(wt)
    print("[bench] Warmup done\n")

    print(f"--- Spectralis TTFP ---")
    hdr = f"{'case':<8} {'chars':>5}  {'first_chunk':>12}  {'total':>10}  {'chunks':>6}  text"
    print(hdr)
    print("-" * 80)

    for label, text in BENCH_TEXTS:
        ttfp_vals, total_vals = [], []
        for rep in range(REPEATS):
            torch.cuda.empty_cache()
            r = _measure(text)
            ttfp_vals.append(r["first_chunk_ms"])
            total_vals.append(r["total_ms"])
            print(f"{label:<8} {len(text.strip()):>5}  "
                  f"{r['first_chunk_ms']:>9.1f}ms  {r['total_ms']:>8.1f}ms  "
                  f"{r['chunk_count']:>6}  {text[:40]}  [rep {rep + 1}]")
        print(f"  -> median ttfp={statistics.median(ttfp_vals):.1f}ms  "
              f"median total={statistics.median(total_vals):.1f}ms")

    print("-" * 60)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    print(f"\nGPU: {gpu}")


if __name__ == "__main__":
    main()
