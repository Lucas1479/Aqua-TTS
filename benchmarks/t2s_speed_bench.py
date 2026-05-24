# -*- coding: utf-8 -*-
"""T2S (Text-to-Semantic) speed benchmark.

Measures AR decoder throughput in iterations/second with optional CUDA Graph.

Usage:
    python benchmarks/t2s_speed_bench.py \
        --gpt-model GPT_weights_v3/xxx-e15.ckpt
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_REPO = os.environ.get("GPT_SOVITS_HOME")
if not MAIN_REPO:
    sys.exit("GPT_SOVITS_HOME must be set to your GPT-SoVITS repo root")
MAIN_GPT_SOVITS = os.path.join(MAIN_REPO, "GPT_SoVITS")
sys.path[:0] = [
    os.path.join(ROOT, "aquatts", "_vendor", "GPT_SoVITS"),
    os.path.join(ROOT, "aquatts", "_vendor"),
    ROOT,
    MAIN_GPT_SOVITS,
    MAIN_REPO,
]
os.chdir(MAIN_REPO)

import torch


def main():
    parser = argparse.ArgumentParser(description="T2S speed benchmark")
    parser.add_argument("--gpt-model", required=True)
    parser.add_argument("--token-count", type=int, default=500,
                        help="Number of tokens to generate per run")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--bench-official", action="store_true",
                        help="Benchmark official (no CUDA Graph) path")
    args = parser.parse_args()

    if args.bench_official:
        os.environ["ENABLE_CUDA_GRAPH"] = "0"
    else:
        os.environ["ENABLE_CUDA_GRAPH"] = "1"

    from GPT_SoVITS.AR.models.t2s_lightning_module import Text2SemanticLightningModule

    print(f"\n{'='*60}")
    print(f"  T2S Speed Benchmark (CUDA Graph={'ON' if not args.bench_official else 'OFF'})")
    print(f"{'='*60}")

    print("[bench] Loading model...")
    dict_s1 = torch.load(args.gpt_model, map_location="cpu")
    config = dict_s1["config"]
    model = Text2SemanticLightningModule(config, "****", is_train=False)
    model.load_state_dict(dict_s1["weight"])
    model = model.half().cuda().eval()

    from aquatts.modeling import apply_cuda_graph_patch
    apply_cuda_graph_patch(model.model)

    if not args.bench_official:
        model.model.precapture_cuda_graph()
        print("[bench] CUDA Graphs pre-captured")

    # Build dummy input
    device = next(model.parameters()).device
    n_tokens = args.token_count
    x = torch.randint(0, config["model"]["phoneme_vocab_size"], (1, 20)).long().cuda()
    x_lens = torch.tensor([20]).long().cuda()
    prompts = torch.randint(0, config["model"]["vocab_size"], (1, 3)).long().cuda()
    bert = torch.randn(1, 1024, 20).half().cuda()

    print("[bench] Warmup...")
    for _ in range(3):
        model.model.infer_panel_naive(x, x_lens, prompts, bert, top_k=5, top_p=1, temperature=0.6)

    print(f"[bench] Running {args.repeats} trials...")
    rates = []
    for trial in range(args.repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result, steps = model.model.infer_panel_naive(x, x_lens, prompts, bert, top_k=5, top_p=1, temperature=0.6)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        rate = steps / elapsed if elapsed > 0 else 0
        rates.append(rate)
        print(f"  Trial {trial + 1}: {steps} tokens in {elapsed:.3f}s = {rate:.0f} it/s")

    median = statistics.median(rates)
    print(f"\n  Median T2S speed: {median:.0f} it/s")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    print(f"  GPU: {gpu}")


if __name__ == "__main__":
    main()
