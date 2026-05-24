# -*- coding: utf-8 -*-
"""Basic usage example for Spectralis-TTS.

Shows how to load models and generate speech with optimizations enabled.

Usage:
    python examples/basic_usage.py \
        --gpt-model GPT_weights_v3/xxx-e15.ckpt \
        --sovits-model SoVITS_weights_v3/xxx_e2_s174_l32.pth \
        --ref-audio "reference audio/ref_audio.wav" \
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


def main():
    parser = argparse.ArgumentParser(description="Spectralis-TTS basic usage")
    parser.add_argument("--gpt-model", required=True)
    parser.add_argument("--sovits-model", required=True)
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--ref-text", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", default="output.wav")
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

    import numpy as np
    import soundfile as sf
    from spectralis import TTSInferencer

    print("Loading TTS pipeline...")
    tts = TTSInferencer(
        device="cuda",
        gpt_path=args.gpt_model,
        sovits_path=args.sovits_model,
    )

    print(f"Generating: {args.text}")
    audio_parts = []
    sample_rate = 24000

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
            audio_parts.append(chunk)
            sample_rate = sr

    if audio_parts:
        audio = np.concatenate(audio_parts)
        sf.write(args.output, audio, sample_rate)
        print(f"Saved to {args.output} ({len(audio) / sample_rate:.1f}s)")
    else:
        print("No audio generated.")


if __name__ == "__main__":
    main()
