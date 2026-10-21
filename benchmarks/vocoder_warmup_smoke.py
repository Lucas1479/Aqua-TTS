"""Selected-device BigVGAN preparation, first-shape timings and stream checks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", choices=("default", "custom", "off"), default="default")
    parser.add_argument("--kernel", choices=("auto", "pytorch"), default="auto")
    parser.add_argument("--graph-preset", choices=("lazy", "off"), default="lazy")
    args = parser.parse_args()
    if args.kernel == "pytorch":
        os.environ["BIGVGAN_USE_CUDA_KERNEL"] = "0"
    from aquatts import TTSInferencer
    import numpy as np
    import torch

    config = json.loads(args.config.read_text(encoding="utf-8"))
    options = dict(config["inferencer"])
    if args.warmup != "default":
        options["bigvgan_warmup_shapes"] = [9, 23, 47] if args.warmup == "custom" else []
        options["bigvgan_warmup_repeats"] = 2
    rng_checks = []
    original_warmup = TTSInferencer._warmup_bigvgan_shapes
    def checked_warmup(self):
        before = torch.get_rng_state(), torch.cuda.get_rng_state(self.device)
        original_warmup(self)
        after = torch.get_rng_state(), torch.cuda.get_rng_state(self.device)
        assert all(torch.equal(left, right) for left, right in zip(before, after))
        rng_checks.append(True)
    TTSInferencer._warmup_bigvgan_shapes = checked_warmup
    started = time.perf_counter()
    value = TTSInferencer(**options, cuda_graph_preset=args.graph_preset, use_flash_attn=True)
    load_seconds = time.perf_counter() - started
    assert value.model_version == "v3" and rng_checks == [True]
    assert value.vocoder_warmup_stats["status"] == ("disabled" if args.warmup == "off" else "completed")
    device = next(value.bigvgan_model.parameters()).device
    assert device == value.device
    parameter = next(value.bigvgan_model.parameters())
    shape_trials = {}
    for frames in (9, 23, 46, 40, 70, 128):
        mel = torch.zeros(1, value.bigvgan_model.h.num_mels, frames, device=device, dtype=parameter.dtype)
        times = []
        for _ in range(3):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            output = value._run_bigvgan(mel)
            torch.cuda.synchronize(device)
            times.append(time.perf_counter() - started)
            assert torch.isfinite(output).all()
        shape_trials[frames] = times
    value.vocoder_shape_counts.clear()
    value.warm_reference_cache(config["reference"]["ref_audio_path"],
                               config["reference"]["prompt_text"], config["reference"]["prompt_language"])
    value._stream_sync_timing_enabled = False
    rows = []
    original_sync = torch.cuda.synchronize
    for chunk_seconds in (.1, .25, .5, None):
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        torch.cuda.manual_seed_all(7)
        explicit_syncs = []
        def observed_sync(*positional, **kwargs):
            explicit_syncs.append(str(positional))
            return original_sync(*positional, **kwargs)
        torch.cuda.synchronize = observed_sync
        chunks, first = [], None
        started = time.perf_counter()
        try:
            for rate, chunk, _ in value.infer_stream(
                    **config["reference"], text="今日はいい天気ですね。", text_language="日文",
                    how_to_cut="不切", sample_steps=4, top_k=5, top_p=1., temperature=.6,
                    pause_second=0, enable_cuda_graph=False, chunk_size_seconds=chunk_seconds):
                if chunk is not None and chunk.size:
                    first = first if first is not None else time.perf_counter() - started
                    chunks.append(chunk)
        finally:
            torch.cuda.synchronize = original_sync
        elapsed = time.perf_counter() - started
        audio = np.concatenate(chunks)
        assert rate == 24000 and np.isfinite(audio).all() and np.max(np.abs(audio)) > 1e-4
        assert explicit_syncs == [], explicit_syncs
        rows.append({"chunk_seconds": chunk_seconds, "first_nonempty_seconds": first,
                     "total_seconds": elapsed, "samples": audio.size,
                     "explicit_device_synchronizations": len(explicit_syncs)})
    report = {"warmup": args.warmup, "kernel_requested": args.kernel,
              "kernel_actual": value.bigvgan_kernel_mode,
              "kernel_fallback_reason": value.bigvgan_kernel_fallback_reason,
              "gpu": torch.cuda.get_device_name(device), "device": str(device),
              "visible_devices": torch.cuda.device_count(), "torch": torch.__version__,
              "graph_preset": args.graph_preset, "load_seconds": load_seconds,
              "warmup_stats": value.vocoder_warmup_stats, "rng_preserved": True,
              "shape_seconds_first_and_two_repeats": shape_trials,
              "request_shape_counts": value.vocoder_shape_counts, "trials": rows,
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
              "source_sha256_lf": hashlib.sha256((ROOT / "aquatts/inferencer.py").read_text(encoding="utf-8").encode()).hexdigest(),
              "scope": "One fresh process per configuration. Request timings follow synthetic vocoder probes and reference warmup; not cold end-to-end latency. Graph is disabled for request sync checks."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
