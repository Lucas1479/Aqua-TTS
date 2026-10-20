# -*- coding: utf-8 -*-
"""Aqua TTFP (Time-To-First-Playback) benchmark.

Measures the optimized streaming inference path with static KV cache,
CUDA Graph, and pre-compiled BigVGAN CUDA kernel. By default, timing stops
at the first non-empty audio chunk that can be played.

Usage:
    python benchmarks/AQUA_ttfp.py \
        --gpt-model GPT_weights_v3/xxx-e15.ckpt \
        --sovits-model SoVITS_weights_v3/xxx_e2_s174_l32.pth \
        --ref-audio "reference audio/ref_audio.wav" \
        --ref-text "your reference transcript"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
import statistics
import sys
import time

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

if not os.environ.get("GPT_SOVITS_HOME"):
    sys.exit("GPT_SOVITS_HOME must be set to your GPT-SoVITS repo root")

import torch  # noqa: E402

BENCH_TEXTS = [
    ("short",  "実験？"),
    ("medium", "何の実験だか気になるけど、まあいいわ。"),
    ("long",   "Paxosが提案ベースでメッセージフローが複雑なのに対して、"
               "Raftは明確なリーダー選出とログ複製のフェーズに分けられているわ。"),
]
WARMUP_TEXTS = ["これはテストです。", "何の実験だか気になるけど、まあいいわ。"]
REPEATS = 5


def main():
    parser = argparse.ArgumentParser(description="Aqua TTFP benchmark")
    parser.add_argument("--gpt-model", required=True)
    parser.add_argument("--sovits-model", required=True)
    parser.add_argument("--sovits-pretrain")
    parser.add_argument("--sv-model", help="ERes2Net checkpoint for v2Pro/v2ProPlus")
    parser.add_argument("--bert-model")
    parser.add_argument("--cnhubert-model")
    parser.add_argument("--bigvgan-model")
    parser.add_argument("--fast-langdetect-model")
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--ref-text", required=True)
    parser.add_argument("--ref-lang", default="日文")
    parser.add_argument("--text-lang", default="日文")
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument("--no-static-kv", action="store_true")
    parser.add_argument("--no-bigvgan-kernel", action="store_true")
    parser.add_argument("--how-to-cut", default="按标点符号切")
    parser.add_argument("--chunk-size-seconds", type=float, default=0.25,
                        help="Audio output chunk size; v1/v2/Pro decode whole sentences first. Set <=0 to disable.")
    parser.add_argument("--measure-total", action="store_true",
                        help="Continue after first playable chunk to measure full generation time.")
    parser.add_argument("--json-output", type=Path, help="Save original-protocol trials and adjacent JSONL")
    args = parser.parse_args()
    from benchmarks.evidence import BenchmarkJournal, file_identity, source_identity
    journal = BenchmarkJournal(args.json_output)
    source = source_identity(ROOT)
    identities = {name: file_identity(value) for name, value in (
        ("gpt", args.gpt_model), ("sovits", args.sovits_model), ("reference_audio", args.ref_audio))}
    journal.record("started", source=source, identities=identities)

    # Configure optimizations via env vars before importing TTSInferencer
    if args.no_cuda_graph:
        os.environ["ENABLE_CUDA_GRAPH"] = "0"
    else:
        os.environ["ENABLE_CUDA_GRAPH"] = "1"

    os.environ["ENABLE_CUDA_GRAPH_PRECAPTURE"] = "1" if not args.no_cuda_graph else "0"

    from aquatts import TTSInferencer

    print(f"\n{'='*80}")
    print("  Aqua TTFP Benchmark")
    print(f"{'='*80}")
    flags = []
    if not args.no_cuda_graph:
        flags.append("CUDA Graph")
    if not args.no_static_kv:
        flags.append("Static KV")
    if not args.no_bigvgan_kernel:
        flags.append("BigVGAN Kernel")
    print(f"  Optimizations: {', '.join(flags) if flags else 'ALL OFF'}")
    print(f"  Segmentation: {args.how_to_cut}")
    if args.chunk_size_seconds and args.chunk_size_seconds > 0:
        print(f"  Stream chunk: {args.chunk_size_seconds:.2f}s")
    else:
        print("  Stream chunk: disabled")
    print(f"  Timing mode: {'first playback + full total' if args.measure_total else 'first playback only'}")
    print(f"  GPT: {args.gpt_model}")
    print(f"  SoVITS: {args.sovits_model}")

    print("[bench] Loading TTS pipeline...")
    t0 = time.perf_counter()
    inferencer = TTSInferencer(
        device="cuda",
        gpt_path=args.gpt_model,
        sovits_path=args.sovits_model,
        sovits_pretrain_path=args.sovits_pretrain,
        sv_model_path=args.sv_model,
        bert_path=args.bert_model,
        cnhubert_path=args.cnhubert_model,
        bigvgan_path=args.bigvgan_model,
        fast_langdetect_path=args.fast_langdetect_model,
    )
    # BigVGAN kernel toggle (set before first inference)
    if args.no_bigvgan_kernel and inferencer.bigvgan_model is not None:
        inferencer.bigvgan_model.use_cuda_kernel = False
    load_seconds = time.perf_counter() - t0
    print(f"[bench] Loaded in {load_seconds:.2f}s")
    decoder = inferencer.t2s_model.model
    backend = {
        "flash_active": getattr(decoder, "use_flash_attn_kvcache", False),
        "bigvgan_kernel": getattr(inferencer, "bigvgan_kernel_mode", None),
        "kv_cache_layout": getattr(decoder, "kv_cache_layout", None),
        "graph_count_after_load": len(getattr(decoder, "bucket_graphs", {})),
    }
    journal.record("loaded", seconds=load_seconds, **backend)

    def _measure(text: str) -> dict:
        t0 = time.perf_counter()
        first_chunk_ms = None
        chunk_count = 0
        t2s_stats_start = len(getattr(inferencer, "t2s_stats", []))

        for sr, chunk, _text in inferencer.infer_stream(
            text=text,
            ref_audio_path=args.ref_audio,
            prompt_text=args.ref_text,
            text_language=args.text_lang,
            prompt_language=args.ref_lang,
            how_to_cut=args.how_to_cut,
            top_k=5, top_p=1, temperature=0.6,
            speed=1.1, sample_steps=4,
            enable_cuda_graph=not args.no_cuda_graph,
            enable_static_kv=not args.no_static_kv,
            chunk_size_seconds=args.chunk_size_seconds if args.chunk_size_seconds > 0 else None,
            collect_t2s_stats=True,
        ):
            if chunk is None or len(chunk) == 0:
                continue
            if len(getattr(inferencer, "t2s_stats", [])) <= t2s_stats_start:
                raise RuntimeError(
                    "TTFP benchmark received audio before any T2S stats were recorded. "
                    "This usually means inference hit the error fallback path; rerun with logs visible."
                )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if first_chunk_ms is None:
                first_chunk_ms = elapsed_ms
            chunk_count += 1
            if not args.measure_total:
                break

        total_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "first_chunk_ms": float(first_chunk_ms or total_ms),
            "total_ms": float(total_ms),
            "chunk_count": chunk_count,
        }

    print("[bench] Warmup...")
    warmup_rows = []
    for index, wt in enumerate(WARMUP_TEXTS):
        row = _measure(wt)
        warmup_rows.append(row)
        journal.record("trial", phase="warmup", index=index, result=row)
    print("[bench] Warmup done\n")

    print("--- Aqua TTFP ---")
    elapsed_label = "total" if args.measure_total else "elapsed"
    hdr = f"{'case':<8} {'chars':>5}  {'ttfp':>12}  {elapsed_label:>10}  {'chunks':>6}  text"
    print(hdr)
    print("-" * 80)

    cases = {}
    for label, text in BENCH_TEXTS:
        ttfp_vals, total_vals, rows = [], [], []
        for rep in range(REPEATS):
            r = _measure(text)
            rows.append(r)
            journal.record("trial", phase="measured", case=label, index=rep, result=r)
            ttfp_vals.append(r["first_chunk_ms"])
            total_vals.append(r["total_ms"])
            print(f"{label:<8} {len(text.strip()):>5}  "
                  f"{r['first_chunk_ms']:>9.1f}ms  {r['total_ms']:>8.1f}ms  "
                  f"{r['chunk_count']:>6}  {text[:40]}  [rep {rep + 1}]")
        cases[label] = {"text": text, "characters": len(text), "trials": rows,
                        "median_first_chunk_ms": statistics.median(ttfp_vals)}
        print(f"  -> median ttfp={statistics.median(ttfp_vals):.1f}ms  "
              f"median {elapsed_label}={statistics.median(total_vals):.1f}ms")

    print("-" * 60)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    print(f"\nGPU: {gpu}")
    result = {
        "protocol": "original-aqua-ttfp", "source": source, "identities": identities,
        "gpu": gpu, "torch": torch.__version__, "model_load_seconds": load_seconds,
        "backend": backend, "warmup": warmup_rows, "cases": cases, "repeats": REPEATS,
        "chunk_seconds": args.chunk_size_seconds, "segmentation": args.how_to_cut,
        "measure_total": args.measure_total, "collect_t2s_stats": True,
        "benchmark_seed_reset": False, "stop_after_first_nonempty_chunk": not args.measure_total,
        "measurement_function": "unchanged from reviewed baseline 4206bb1",
    }
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    journal.record("completed", cases=list(cases))


if __name__ == "__main__":
    main()
