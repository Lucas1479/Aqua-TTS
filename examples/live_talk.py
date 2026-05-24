# -*- coding: utf-8 -*-
"""Interactive Japanese playback console for Aqua-TTS.

Load the voice once, then type Japanese text and hear it immediately.
The console output is intentionally concise for demo recording.

Usage:
    python examples/live_talk.py --gpt-sovits-home /path/to/GPT-SoVITS-v3lora
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(EXAMPLES_DIR)]

from play_ete import (  # noqa: E402
    DEFAULT_CHUNK_SECONDS,
    DEFAULT_SAMPLE_STEPS,
    DEFAULT_SPEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    KURISU_REF_TEXT,
    _configure_logs,
    _default_path,
    _load_pyaudio,
    _quiet_output,
    _require_path,
    _set_seed,
    _warmup,
    play_utterance,
)


def _parse_args():
    parser = argparse.ArgumentParser(description="Aqua-TTS live Japanese playback console")
    parser.add_argument("--gpt-sovits-home", default=os.environ.get("GPT_SOVITS_HOME", ""))
    parser.add_argument("--gpt-model", default=os.environ.get("AQUA_GPT_MODEL"))
    parser.add_argument("--sovits-model", default=os.environ.get("AQUA_SOVITS_MODEL"))
    parser.add_argument("--ref-audio", default=os.environ.get("AQUA_REF_AUDIO"))
    parser.add_argument("--ref-text", default=os.environ.get("TTS_REF_TEXT_JA", KURISU_REF_TEXT))
    parser.add_argument("--text-lang", default="日文")
    parser.add_argument("--ref-lang", default="日文")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED)
    parser.add_argument("--sample-steps", type=int, default=DEFAULT_SAMPLE_STEPS,
                        help="CFM sampling steps. Higher is steadier but slower.")
    parser.add_argument("--chunk-size-seconds", type=float, default=DEFAULT_CHUNK_SECONDS,
                        help="Streaming playback chunk size. Smaller starts sooner; larger can sound steadier.")
    parser.add_argument("--output-device-index", type=int, default=None)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--show-total", action="store_true",
                        help="Also print audio duration, wall time, and RTF.")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Sampling seed for repeatable demo output; set -1 for random.")
    parser.add_argument("--verbose", action="store_true",
                        help="Show internal model logs and tqdm progress.")
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = _parse_args()
    _configure_logs(args.verbose)
    _set_seed(args.seed)

    home = args.gpt_sovits_home
    if home:
        os.environ["GPT_SOVITS_HOME"] = home

    gpt_model = args.gpt_model or _default_path(home, "GPT_weights_v3", "xxx-e15.ckpt")
    sovits_model = args.sovits_model or _default_path(home, "SoVITS_weights_v3", "xxx_e2_s174_l32.pth")
    ref_audio = args.ref_audio or _default_path(home, "reference audio", "kurisu_reference.wav")

    pyaudio = _load_pyaudio()
    pa = pyaudio.PyAudio()
    if args.list_devices:
        for idx in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(idx)
            if info.get("maxOutputChannels", 0) > 0:
                print(f"{idx}: {info.get('name')} ({int(info.get('defaultSampleRate', 0))} Hz)")
        pa.terminate()
        return

    gpt_model = _require_path("GPT model", gpt_model)
    sovits_model = _require_path("SoVITS model", sovits_model)
    ref_audio = _require_path("Reference audio", ref_audio)

    if args.no_cuda_graph:
        os.environ["ENABLE_CUDA_GRAPH"] = "0"

    from aquatts import TTSInferencer

    print("[live] Loading voice...")
    t0 = time.perf_counter()
    with _quiet_output(not args.verbose):
        tts = TTSInferencer(
            device=args.device,
            gpt_path=gpt_model,
            sovits_path=sovits_model,
        )
    print(f"[live] Ready in {time.perf_counter() - t0:.2f}s")

    if not args.no_warmup:
        _set_seed(args.seed)
        _warmup(tts, args, ref_audio, quiet=not args.verbose)

    print("\nType Japanese text and press Enter. Use /q to quit.")
    line_no = 1
    try:
        while True:
            try:
                text = input("ja> ").strip()
            except EOFError:
                break
            if not text:
                continue
            if text.lower() in {"/q", "/quit", "quit", "exit"}:
                break
            play_utterance(tts, pa, pyaudio, args, ref_audio, f"line{line_no}", text)
            line_no += 1
    finally:
        pa.terminate()


if __name__ == "__main__":
    main()
