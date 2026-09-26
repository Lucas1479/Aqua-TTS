"""Warm model-side latency by checkpoint and T2S engine (one process per run).

Accepts the same private model/reference config as model_smoke.py. Results omit
absolute paths and reference transcripts; every measured repeat is preserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def summarize(rows):
    if not rows:
        raise ValueError("Cannot summarize empty latency results")
    return {
        **{f"median_{key}": statistics.median(row[key] for row in rows)
           for key in ("first_audio_ms", "total_ms", "audio_seconds", "rtf", "semantic_ms")},
        "min_first_audio_ms": min(row["first_audio_ms"] for row in rows),
        "max_first_audio_ms": max(row["first_audio_ms"] for row in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--engine", choices=("upstream", "sdpa", "flash"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
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
    original_semantic = inferencer._infer_semantic_with_guard

    def capture_semantic(**kwargs):
        result = original_semantic(**kwargs)
        semantic_events.append(result)
        return result

    inferencer._infer_semantic_with_guard = capture_semantic

    def measure(text, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        semantic_events.clear()
        before = len(inferencer.t2s_stats)
        first = None
        samples = 0
        peak = 0.0
        # Clear pending work outside the caller-visible timing window.
        torch.cuda.synchronize(inferencer.device)
        begin = time.perf_counter()
        for rate, chunk, _ in inferencer.infer_stream(
            **config["reference"], text=text, text_language="日文", how_to_cut="按标点符号切",
            top_k=5, top_p=1., temperature=.6, speed=1.1, sample_steps=4,
            chunk_size_seconds=.25, pause_second=.3, collect_t2s_stats=True,
            enable_cuda_graph=args.engine != "upstream", enable_static_kv=args.engine != "upstream",
        ):
            if chunk is None or not len(chunk):
                continue
            if first is None:
                first = (time.perf_counter() - begin) * 1000
            assert np.isfinite(chunk).all()
            peak = max(peak, float(np.abs(chunk).max()))
            samples += len(chunk)
        total = (time.perf_counter() - begin) * 1000
        assert first is not None and peak > 1e-4
        stats = inferencer.t2s_stats[before:]
        assert stats and semantic_events
        hashes = [hashlib.sha256(item[0].detach().cpu().numpy().tobytes()).hexdigest()
                  for item in semantic_events]
        seconds = samples / rate
        return dict(seed=seed, first_audio_ms=first, total_ms=total, audio_seconds=seconds,
                    rtf=total / 1000 / seconds, sample_rate=rate,
                    semantic_ms=sum(stat["elapsed_sec"] for stat in stats) * 1000,
                    semantic_tokens=sum(stat["tokens"] for stat in stats), semantic_hashes=hashes,
                    semantic_attempts=[item[2] for item in semantic_events])

    warmup = [measure(text, args.seed - index - 1) for index, text in enumerate(WARMUP_TEXTS)]
    cases = {}
    for name, text in BENCH_TEXTS:
        rows = [measure(text, args.seed + repeat) for repeat in range(args.repeats)]
        cases[name] = dict(text=text, characters=len(text), summary=summarize(rows), trials=rows)
    if args.engine != "upstream":
        assert decoder.bucket_graphs and decoder.cuda_graph_stats["graph_replay_steps"] > 0
    result = dict(model=inferencer.model_version, engine=args.engine,
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
