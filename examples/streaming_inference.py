# -*- coding: utf-8 -*-
"""Streaming inference example — play audio as it's generated.

Shows real-time streaming playback using PyAudio.

Usage:
    python examples/streaming_inference.py \
        --gpt-model GPT_weights_v3/xxx-e15.ckpt \
        --sovits-model SoVITS_weights_v3/xxx_e2_s174_l32.pth \
        --ref-audio "reference audio/kurisu_reference2.wav" \
        --ref-text "reference transcript" \
        --text "こんにちは、世界！"
"""
from __future__ import annotations

import argparse, os, sys

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("ENABLE_CUDA_GRAPH", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "GPT_SoVITS"))


def main():
    parser = argparse.ArgumentParser(description="Spectralis-TTS streaming inference")
    parser.add_argument("--gpt-model", required=True)
    parser.add_argument("--sovits-model", required=True)
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--ref-text", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--text-lang", default="日文")
    parser.add_argument("--ref-lang", default="日文")
    parser.add_argument("--no-cuda-graph", action="store_true")
    args = parser.parse_args()

    if args.no_cuda_graph:
        os.environ["ENABLE_CUDA_GRAPH"] = "0"

    import pyaudio
    import time
    from local_tts_infer import TTSInferencer

    print("Loading TTS pipeline...")
    tts = TTSInferencer(
        device="cuda",
        gpt_path=args.gpt_model,
        sovits_path=args.sovits_model,
    )

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=24000,
        output=True,
    )

    print(f"Streaming: {args.text}")
    t_start = time.perf_counter()
    total_samples = 0

    for sr, chunk in tts.infer_stream(
        text=args.text,
        ref_audio_path=args.ref_audio,
        prompt_text=args.ref_text,
        text_language=args.text_lang,
        prompt_language=args.ref_lang,
        how_to_cut="不切",
        top_k=5, top_p=1, temperature=0.6,
        speed=1.1, sample_steps=4,
        enable_cuda_graph=not args.no_cuda_graph,
        enable_static_kv=True,
    ):
        if chunk is not None and len(chunk) > 0:
            stream.write(chunk.tobytes())
            total_samples += len(chunk)

    stream.stop_stream()
    stream.close()
    p.terminate()

    elapsed = time.perf_counter() - t_start
    audio_dur = total_samples / 24000
    print(f"Done: {audio_dur:.1f}s audio in {elapsed:.1f}s "
          f"(RTF: {elapsed / audio_dur:.2f}x)")


if __name__ == "__main__":
    main()
