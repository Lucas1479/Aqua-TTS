# -*- coding: utf-8 -*-
"""Streaming inference example — play audio as it's generated.

Shows real-time streaming playback using PyAudio.

Usage:
    python examples/streaming_inference.py \
        --gpt-model GPT_weights_v3/xxx-e15.ckpt \
        --sovits-model SoVITS_weights_v3/xxx_e2_s174_l32.pth \
        --ref-audio "reference audio/ref_audio.wav" \
        --ref-text "reference transcript" \
        --text "こんにちは、世界！"
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("ENABLE_CUDA_GRAPH", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    parser = argparse.ArgumentParser(description="Aqua-TTS streaming inference")
    parser.add_argument("--gpt-model", required=True)
    parser.add_argument("--sovits-model", required=True)
    parser.add_argument("--sv-model", help="ERes2Net checkpoint for v2Pro/v2ProPlus")
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--ref-text", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--text-lang", default="日文")
    parser.add_argument("--ref-lang", default="日文")
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument("--gpt-sovits-home", default=os.environ.get("GPT_SOVITS_HOME", ""),
                        help="Path to main GPT-SoVITS repo")
    args = parser.parse_args()

    if args.gpt_sovits_home:
        os.environ["GPT_SOVITS_HOME"] = args.gpt_sovits_home
    if args.no_cuda_graph:
        os.environ["ENABLE_CUDA_GRAPH"] = "0"

    import pyaudio
    from aquatts import TTSInferencer

    print("Loading TTS pipeline...")
    tts = TTSInferencer(
        device="cuda",
        gpt_path=args.gpt_model,
        sovits_path=args.sovits_model,
        sv_model_path=args.sv_model,
    )

    p = pyaudio.PyAudio()
    stream = None
    audio_dur = 0.0

    print(f"Streaming: {args.text}")
    t_start = time.perf_counter()

    try:
        for sr, chunk, text in tts.infer_stream(
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
                if stream is None:
                    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=sr, output=True)
                stream.write(chunk.tobytes())
                audio_dur += len(chunk) / sr

    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()
        p.terminate()

    elapsed = time.perf_counter() - t_start
    print(f"Done: {audio_dur:.1f}s audio in {elapsed:.1f}s "
          f"(RTF: {(elapsed / audio_dur if audio_dur else 0):.2f}x)")


if __name__ == "__main__":
    main()
