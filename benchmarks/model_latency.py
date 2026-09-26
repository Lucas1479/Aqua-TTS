"""Warm model-side latency by checkpoint and T2S engine (one process per run).

Accepts the same private model/reference config as model_smoke.py. Results omit
absolute paths and reference transcripts; every measured repeat is preserved.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def summarize(rows):
    if not rows:
        raise ValueError("Cannot summarize empty latency results")
    return {
        **{f"median_{key}": (statistics.median(row[key] for row in rows)
                            if all(row.get(key) is not None for row in rows) else None)
           for key in ("first_audio_ms", "total_ms", "audio_seconds", "rtf", "semantic_ms")},
        "p95_first_audio_ms": sorted(row["first_audio_ms"] for row in rows)[math.ceil(.95 * len(rows)) - 1],
        "min_first_audio_ms": min(row["first_audio_ms"] for row in rows),
        "max_first_audio_ms": max(row["first_audio_ms"] for row in rows),
    }


@contextmanager
def gpu_monitor(uuid, destination):
    """Record one physical GPU without repeatedly spawning the NVIDIA CLI."""
    if destination is None:
        yield
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = "timestamp,pstate,utilization.gpu,utilization.memory,memory.used,clocks.sm,clocks.mem,power.draw,temperature.gpu"
    with destination.open("w", encoding="utf-8") as output, destination.with_suffix(".stderr.log").open("w") as errors:
        process = subprocess.Popen(
            ["nvidia-smi", "-i", uuid, f"--query-gpu={fields}",
             "--format=csv,nounits", "--loop-ms=500"],
            stdout=output, stderr=errors,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            time.sleep(3)  # Background-load samples before importing/loading models.
            if process.poll() is not None:
                raise RuntimeError("GPU telemetry exited before the benchmark; inspect its stderr log")
            yield
        finally:
            process.terminate()
            process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--engine", choices=("upstream", "sdpa", "flash"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--steady", action="store_true", help="Fixed-seed, per-case warmup; separate diagnostic timing")
    parser.add_argument("--warmup-per-case", type=int, default=5)
    parser.add_argument("--diagnostic-repeats", type=int, default=3)
    parser.add_argument("--gpu-uuid", help="Lock CUDA to one physical NVIDIA GPU")
    parser.add_argument("--telemetry-output", type=Path, help="500 ms NVIDIA utilization/clock CSV")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.warmup_per_case < 1 or args.diagnostic_repeats < 1:
        parser.error("warmup and diagnostic repeats must be positive")
    if args.telemetry_output and not args.gpu_uuid:
        parser.error("--telemetry-output requires --gpu-uuid")
    if args.gpu_uuid:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    with gpu_monitor(args.gpu_uuid, args.telemetry_output):
        run_benchmark(args)


def run_benchmark(args):
    os.environ["TQDM_DISABLE"] = "1"
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import torch
    from aquatts import TTSInferencer
    from aqua_ttfp import BENCH_TEXTS, WARMUP_TEXTS

    class UpstreamInferencer(TTSInferencer):
        def _apply_aqua_t2s_patch(self):
            # Keep native infer_panel()/infer_panel_naive() intact. The frontend,
            # reference cache, semantic guard and SoVITS path are common to all.
            pass

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.gpu_uuid:
        config["inferencer"]["device"] = "cuda:0"
    cls = UpstreamInferencer if args.engine == "upstream" else TTSInferencer
    started = time.perf_counter()
    inferencer = cls(**config["inferencer"], use_flash_attn=args.engine == "flash",
                     cuda_graph_preset="off" if args.engine == "upstream" else "lazy")
    load_seconds = time.perf_counter() - started
    assert inferencer.model_version == config["expected_version"]
    decoder = inferencer.t2s_model.model
    if args.engine == "upstream":
        assert not hasattr(decoder, "_aquatts_original_infer_panel")
    if args.engine == "flash":
        assert getattr(decoder, "use_flash_attn_kvcache", False)

        def reject_fallback(*_args, **_kwargs):
            raise AssertionError("FA2 benchmark fell back to SDPA")

        # A configured FA2 flag alone does not prove that its kernels ran.
        # This benchmark must fail if an unsupported kernel silently falls back.
        for block in decoder.t2s_transformer_static.blocks:
            block.sdpa_block.decode_next_token_with_static_cache = reject_fallback
    logging.getLogger("tts_inference").setLevel(logging.WARNING)
    semantic_events = []
    stage_times = {"frontend": [], "acoustic": []}
    diagnostic_mode = False
    original_semantic = inferencer._infer_semantic_with_guard

    def capture_semantic(**kwargs):
        result = original_semantic(**kwargs)
        semantic_events.append(result)
        return result

    inferencer._infer_semantic_with_guard = capture_semantic

    def timed_stage(method, name):
        def wrapper(*positional, **kwargs):
            if not diagnostic_mode:
                return method(*positional, **kwargs)
            torch.cuda.synchronize(inferencer.device)
            begin = time.perf_counter()
            result = method(*positional, **kwargs)
            torch.cuda.synchronize(inferencer.device)
            stage_times[name].append((time.perf_counter() - begin) * 1000)
            return result
        return wrapper

    if args.steady:
        inferencer.get_phones_and_bert = timed_stage(inferencer.get_phones_and_bert, "frontend")
        inferencer._decode_v2 = timed_stage(inferencer._decode_v2, "acoustic")

    def measure(text, seed, *, diagnostic=False):
        nonlocal diagnostic_mode
        diagnostic_mode = diagnostic
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        semantic_events.clear()
        for values in stage_times.values():
            values.clear()
        before = len(inferencer.t2s_stats)
        graphs_before = len(getattr(decoder, "bucket_graphs", {}))
        first = None
        samples = 0
        peak = 0.0
        chunks = []
        # Clear pending work outside the caller-visible timing window.
        torch.cuda.synchronize(inferencer.device)
        timestamp = datetime.now().astimezone().isoformat()
        begin = time.perf_counter()
        for rate, chunk, _ in inferencer.infer_stream(
            **config["reference"], text=text, text_language="日文", how_to_cut="按标点符号切",
            top_k=5, top_p=1., temperature=.6, speed=1.1, sample_steps=4,
            chunk_size_seconds=.25, pause_second=.3, collect_t2s_stats=diagnostic or not args.steady,
            enable_cuda_graph=args.engine != "upstream", enable_static_kv=args.engine != "upstream",
        ):
            if chunk is None or not len(chunk):
                continue
            if first is None:
                first = (time.perf_counter() - begin) * 1000
            if args.steady:
                chunks.append(chunk)
            else:
                assert np.isfinite(chunk).all()
                peak = max(peak, float(np.abs(chunk).max()))
                samples += len(chunk)
        total = (time.perf_counter() - begin) * 1000
        if args.steady:
            for chunk in chunks:
                assert np.isfinite(chunk).all()
                peak = max(peak, float(np.abs(chunk).max()))
                samples += len(chunk)
        assert first is not None and peak > 1e-4
        stats = inferencer.t2s_stats[before:]
        assert semantic_events
        if diagnostic or not args.steady:
            assert stats
        hashes = [hashlib.sha256(item[0].detach().cpu().numpy().tobytes()).hexdigest()
                  for item in semantic_events]
        seconds = samples / rate
        return dict(seed=seed, started_at=timestamp, first_audio_ms=first, total_ms=total, audio_seconds=seconds,
                    rtf=total / 1000 / seconds, sample_rate=rate,
                    semantic_ms=sum(stat["elapsed_sec"] for stat in stats) * 1000 if stats else None,
                    frontend_ms=sum(stage_times["frontend"]) if diagnostic else None,
                    acoustic_ms=sum(stage_times["acoustic"]) if diagnostic else None,
                    new_graphs=len(getattr(decoder, "bucket_graphs", {})) - graphs_before,
                    semantic_tokens=sum(item[1] for item in semantic_events), semantic_hashes=hashes,
                    semantic_attempts=[item[2] for item in semantic_events])

    warmup = [] if args.steady else [measure(text, args.seed - index - 1) for index, text in enumerate(WARMUP_TEXTS)]
    cases = {}
    for name, text in BENCH_TEXTS:
        if args.steady:
            first_use = measure(text, args.seed)
            case_warmup = [measure(text, args.seed) for _ in range(args.warmup_per_case)]
            rows = [measure(text, args.seed) for _ in range(args.repeats)]
            assert all(row["new_graphs"] == 0 for row in rows), "Measured steady trials captured new graphs"
            diagnostics = [measure(text, args.seed, diagnostic=True) for _ in range(args.diagnostic_repeats)]
            cases[name] = dict(text=text, characters=len(text), first_use=first_use, warmup=case_warmup,
                               summary=summarize(rows), trials=rows, diagnostics=diagnostics,
                               semantic_variants=len({tuple(row["semantic_hashes"]) for row in rows}))
        else:
            rows = [measure(text, args.seed + repeat) for repeat in range(args.repeats)]
            cases[name] = dict(text=text, characters=len(text), summary=summarize(rows), trials=rows)
    if args.engine != "upstream":
        assert decoder.bucket_graphs and decoder.cuda_graph_stats["graph_replay_steps"] > 0
    result = dict(model=inferencer.model_version, engine=args.engine,
                  protocol="fixed-seed-steady" if args.steady else "paired-seed",
                  torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
                  gpu_uuid_sha256=hashlib.sha256(args.gpu_uuid.encode()).hexdigest() if args.gpu_uuid else None,
                  telemetry_file=args.telemetry_output.name if args.telemetry_output else None,
                  flash_fallback_forbidden=args.engine == "flash",
                  torch=torch.__version__, gpu=torch.cuda.get_device_name(inferencer.device),
                  gpt_checkpoint=Path(inferencer.gpt_path).name,
                  sovits_checkpoint=Path(inferencer.sovits_path).name,
                  model_load_seconds=load_seconds, warmup=warmup, repeats=args.repeats,
                  chunk_seconds=.25, pause_seconds=.3, speed=1.1,
                  segmentation="按标点符号切", cases=cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, value in cases.items():
        print(args.engine, name, json.dumps(value["summary"]))
    print(f"PASS {args.output}")


if __name__ == "__main__":
    main()
