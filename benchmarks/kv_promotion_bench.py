"""Fixed-work, real-weight prepared-promotion benchmark (one process per source).

Compare the same command with and without --baseline-source exported from N2.
Synthetic conditioning and forced early stop measure T2S, not speech quality.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks import t2s_speed_bench as bench  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpt-model", type=Path, required=True)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--flash-attn", choices=("off", "valid"), default="valid")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        parser.error("warmup must be nonnegative; repeats must be positive")

    from aquatts import TTSInferencer  # noqa: F401 - initialize upstream import context
    import aquatts.modeling as modeling
    import aquatts.modeling.t2s_streaming as runtime

    if args.baseline_source:
        spec = importlib.util.spec_from_file_location("promotion_baseline", args.baseline_source)
        runtime = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = runtime
        spec.loader.exec_module(runtime)
        modeling.apply_cuda_graph_patch = runtime.apply_cuda_graph_patch
    os.environ["AQUATTS_KV_CACHE_LAYOUT"] = "nested"
    args.engine = "aqua"
    buckets = [448, 512, 768, 1024]
    if args.flash_attn != "off":
        # Reject fallback during capture as well as eager runtime calls.
        from aquatts.modeling.t2s_flash_attn import T2SBlockWithStaticCache
        def reject_fallback(*_args, **_kwargs):
            raise AssertionError("FA2 silently fell back to SDPA")
        T2SBlockWithStaticCache.decode_next_token_with_static_cache = reject_fallback
    torch, model, decoder, config, enabled = bench._load_decoder(
        args, Path(os.environ["GPT_SOVITS_HOME"]), args.gpt_model, buckets)
    device = next(model.parameters()).device
    cases = [bench.BenchmarkCase("short_control", 117, 191, 51, 448),
             bench.BenchmarkCase("one_promotion", 144, 191, 224, 512),
             bench.BenchmarkCase("two_promotions", 144, 191, 600, 512)]
    decoder.EOS = -1
    results = {}
    with torch.cuda.device(device):
        for case in cases:
            inputs = bench._make_inputs(torch, config, case, device)
            def run(seed):
                row = bench._run_once(torch, decoder, inputs, case, graph_enabled=enabled, seed=seed)
                row["path"] = dict(decoder.cuda_graph_stats)
                return row
            cold = run(args.seed)
            for warmup in range(args.warmup):
                run(args.seed + warmup)
            rows = [run(args.seed + repeat) for repeat in range(args.repeats)]
            results[case.name] = {**asdict(case), "initial_len": case.initial_len,
                                  "cold": cold, **bench._summarize(rows)}
            print(case.name, results[case.name]["median_sync_it_s"], flush=True)
    report = {
        "source_sha256_lf": hashlib.sha256(Path(runtime.__file__).read_text(encoding="utf-8")
                                            .replace("\r\n", "\n").encode()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(args.gpt_model.read_bytes()).hexdigest(),
        "baseline": args.baseline_source is not None,
        "gpu": torch.cuda.get_device_name(device), "device": str(device),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "attention": args.flash_attn, "dtype": str(next(model.parameters()).dtype),
        "layout": "nested", "prepared_buckets": buckets,
        "warmup_per_case": args.warmup, "repeats": args.repeats, "seed": args.seed,
        "cases": results,
        "scope": "Synthetic fixed-work T2S; not end-to-end latency or speech quality.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
