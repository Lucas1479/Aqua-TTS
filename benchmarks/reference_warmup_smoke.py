"""Measure explicit reference readiness and verify subsequent encoder reuse."""
from __future__ import annotations

import argparse
import hashlib
import json
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
    args = parser.parse_args()
    from aquatts import TTSInferencer, Voice
    import numpy as np
    import soundfile as sf
    import torch

    config = json.loads(args.config.read_text(encoding="utf-8"))
    counts = {name: 0 for name in ("ssl", "spectrum", "reference_decode_encp", "speaker")}
    def track(owner, method, name, predicate=lambda *_args, **_kwargs: True):
        original = getattr(owner, method)
        def call(*positional, **kwargs):
            if predicate(*positional, **kwargs):
                counts[name] += 1
            return original(*positional, **kwargs)
        setattr(owner, method, call)
    load_models = TTSInferencer._load_models
    def load_and_observe(self):
        load_models(self)
        track(self.ssl_model.model, "forward", "ssl")
        track(self, "get_spepc", "spectrum")
        if self.model_version == "v3":
            track(self.vq_model, "decode_encp", "reference_decode_encp",
                  lambda *positional, **_: len(positional) == 3)
        if self.is_v2pro:
            track(self.sv_model, "encode", "speaker")
    TTSInferencer._load_models = load_and_observe
    reference = config["reference"]
    voice = Voice("warmup", reference["ref_audio_path"], reference["prompt_text"].rstrip("。."),
                  "all_ja" if reference["prompt_language"] == "日文" else reference["prompt_language"])
    started = time.perf_counter()
    inferencer = TTSInferencer(**config["inferencer"], warmup_references=[voice],
                              cuda_graph_preset="lazy", use_flash_attn=True)
    ready_seconds = time.perf_counter() - started
    assert inferencer.model_version == config["expected_version"]
    assert inferencer.reference_warmup_results[0]["ready"]
    generation = inferencer._reference_generation
    warm_counts = dict(counts)
    assert warm_counts == {"ssl": 1, "spectrum": 1,
                           "reference_decode_encp": int(inferencer.model_version == "v3"),
                           "speaker": int(inferencer.is_v2pro)}
    hit_started = time.perf_counter()
    hit = inferencer.warm_reference_cache(reference["ref_audio_path"], reference["prompt_text"],
                                          reference["prompt_language"])
    hit_seconds = time.perf_counter() - hit_started
    assert hit["status"] == "hit" and counts == warm_counts
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for streaming in (False, True):
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        torch.cuda.manual_seed_all(7)
        kwargs = dict(reference, text="今日はいい天気ですね。", text_language="日文",
                      how_to_cut="不切", sample_steps=4, top_k=5, top_p=1., temperature=.6,
                      pause_second=0)
        started = time.perf_counter()
        first = None
        if streaming:
            chunks = []
            for rate, chunk, _ in inferencer.infer_stream(**kwargs, chunk_size_seconds=.25):
                if chunk is not None and chunk.size:
                    first = first if first is not None else time.perf_counter() - started
                    chunks.append(chunk)
            audio = np.concatenate(chunks)
        else:
            rate, audio = inferencer.infer(**kwargs)
        elapsed = time.perf_counter() - started
        assert audio.dtype == np.float32 and np.isfinite(audio).all() and np.max(np.abs(audio)) > 1e-4
        assert counts == warm_counts, (counts, warm_counts)
        sf.write(args.output / f"{'stream' if streaming else 'ordinary'}.wav", audio, rate)
        rows.append({"streaming": streaming, "sample_rate": rate, "samples": audio.size,
                     "first_nonempty_seconds": first, "total_seconds": elapsed,
                     "reference_encoders": dict(counts)})
    cleared = inferencer.clear_reference_cache()
    assert cleared == 1 and inferencer._reference_generation == generation + 1
    rebuilt = inferencer.warm_reference_cache(reference["ref_audio_path"], reference["prompt_text"],
                                              reference["prompt_language"])
    assert rebuilt["status"] == "built" and counts == {key: 2 * value for key, value in warm_counts.items()}
    report = {"version": inferencer.model_version, "gpu": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "generation_at_ready": generation,
              "constructor_with_warmup_seconds": ready_seconds, "warm_hit_seconds": hit_seconds,
              "reference_encoders_after_warmup": warm_counts, "trials": rows,
              "reference_encoders_after_clear_and_rebuild": counts,
              "source_sha256_lf": hashlib.sha256((ROOT / "aquatts/inferencer.py").read_text(encoding="utf-8").encode()).hexdigest(),
              "scope": "Single-run functional and timing observations; no latency distribution claim."}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
