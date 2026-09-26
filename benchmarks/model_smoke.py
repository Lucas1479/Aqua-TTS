"""Real-weight compatibility matrix. Run each model in a fresh process.

The JSON config supplies `inferencer` constructor kwargs, `reference` inference
kwargs, and `expected_version`. Weights and reference material remain external.
Example config and invocation are documented in benchmarks/README.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class ErrorRecorder(logging.Handler):
    def __init__(self):
        super().__init__(logging.ERROR)
        self.errors = []

    def emit(self, record):
        self.errors.append(record.getMessage())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flash", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    from aquatts import TTSInferencer
    import soundfile as sf

    errors = ErrorRecorder()
    logging.getLogger("tts_inference").addHandler(errors)
    started = time.perf_counter()
    inferencer = TTSInferencer(**config["inferencer"], use_flash_attn=args.flash,
                              cuda_graph_preset="lazy")
    assert inferencer.model_version == config["expected_version"]
    model_load_seconds = time.perf_counter() - started
    texts = config.get("texts", [
        {"text": "今日はいい天気ですね。", "text_language": "日文"},
        {"text": "うーん……正直、まだ完全には分からないけど、もう一度確認してみるわ。", "text_language": "日文"},
        {"text": "This is a test of voice synthesis. Please check the complete sentence.", "text_language": "英文"},
    ])
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    expected_rate = 24000 if inferencer.model_version == "v3" else int(inferencer.hps.data.sampling_rate)
    speaker_calls = []
    if inferencer.sv_model is not None:
        encode = inferencer.sv_model.encode

        def recorded_encode(path):
            speaker_calls.append(str(path))
            return encode(path)

        inferencer.sv_model.encode = recorded_encode

    def reset_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def check_audio(rate, audio):
        assert rate == expected_rate, (rate, expected_rate)
        assert audio.ndim == 1 and audio.size > rate // 10
        assert audio.dtype == np.float32 and np.isfinite(audio).all()
        assert np.max(np.abs(audio)) > 1e-4, "silent fallback is not successful synthesis"
        assert not errors.errors, errors.errors

    shared = {**config["reference"], "top_k": 5, "top_p": 1., "temperature": .6,
              "pause_second": 0, "how_to_cut": "不切", "sample_steps": 4}
    reset_seed(7)
    rate, audio = inferencer.infer(**shared, **texts[0])
    check_audio(rate, audio)
    sf.write(args.output / "nonstream.wav", audio, rate)
    for mode, graph, static in (("dynamic", False, False), ("static", False, True), ("graph", True, True)):
        for seed in (7, 17):
            for index, text in enumerate(texts):
                reset_seed(seed)
                before = len(inferencer.t2s_stats)
                start = time.perf_counter()
                first = None
                chunks = []
                for rate, chunk, _ in inferencer.infer_stream(
                    **shared, **text, enable_cuda_graph=graph, enable_static_kv=static,
                    chunk_size_seconds=.25, collect_t2s_stats=True,
                ):
                    assert rate == expected_rate
                    if chunk is not None and chunk.size:
                        if first is None:
                            first = time.perf_counter() - start
                        chunks.append(chunk)
                total = time.perf_counter() - start
                assert len(inferencer.t2s_stats) > before, "missing semantic generation stats"
                audio = np.concatenate(chunks)
                check_audio(rate, audio)
                sf.write(args.output / f"{mode}-{seed}-{index}.wav", audio, rate)
                rows.append(dict(mode=mode, seed=seed, text_index=index, rate=rate,
                                 duration=audio.size / rate, first_chunk_seconds=first,
                                 total_seconds=total, chunks=len(chunks)))
    if inferencer.is_v2pro:
        assert len(speaker_calls) == 1, speaker_calls
    decoder = inferencer.t2s_model.model
    assert decoder.bucket_graphs, "graph trials must actually capture CUDA graphs"
    assert decoder.cuda_graph_stats["graph_replay_steps"] > 0
    if args.flash:
        assert getattr(decoder, "flash_attn_kvcache_mode", None), "FlashAttention was requested but not active"

    def identity(filename):
        path = Path(filename)
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return {"name": path.name, "sha256": digest.hexdigest()}

    report = dict(version=inferencer.model_version, torch=torch.__version__,
                  gpu=torch.cuda.get_device_name(0), sample_rate=expected_rate,
                  model_load_seconds=model_load_seconds, speaker_encodes=len(speaker_calls),
                  graph_count=len(getattr(decoder, "bucket_graphs", {})),
                  last_graph_replay_steps=decoder.cuda_graph_stats["graph_replay_steps"],
                  flash_requested=args.flash,
                  flash_mode=getattr(decoder, "flash_attn_kvcache_mode", None),
                  gpt=identity(inferencer.gpt_path), sovits=identity(inferencer.sovits_path),
                  nonstream_passed=True, stream_trials=rows)
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "stream_trials"}, indent=2))
    print(f"PASS: {len(rows)} stream trials; {args.output}")


if __name__ == "__main__":
    main()
