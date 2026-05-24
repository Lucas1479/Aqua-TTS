# -*- coding: utf-8 -*-
"""End-to-end streaming playback demo for Aqua-TTS.

This script loads a GPT-SoVITS v3 voice, streams generated audio to the
default sound card with PyAudio, and prints TTFP/RTF for each utterance.

Usage:
    python examples/play_ete.py --gpt-sovits-home /path/to/GPT-SoVITS-v3lora

For playback support:
    pip install -e ".[playback]"
"""
from __future__ import annotations

import argparse
import contextlib
import logging
import os
import queue
import random
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("ENABLE_CUDA_GRAPH", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

KURISU_REF_TEXT = (
    "そういえば,正式に自己紹介していませんでしたね……"
    "牧瀬紅莉栖です.改めてまして,よろしく"
)

KURISU_DEMO_TEXTS = [
    ("short", "実験？"),
    ("medium", "何の実験だか気になるけど、まあいいわ。"),
    (
        "long",
        "Paxosが提案ベースでメッセージフローが複雑なのに対して、"
        "Raftは明確なリーダー選出とログ複製のフェーズに分けられているわ。",
    ),
]
WARMUP_TEXT = "これはテストです。"
DEFAULT_TOP_K = 5
DEFAULT_TOP_P = 0.9
DEFAULT_TEMPERATURE = 0.6
DEFAULT_SPEED = 1.0
DEFAULT_SAMPLE_STEPS = 4
DEFAULT_CHUNK_SECONDS = 0.35


@contextlib.contextmanager
def _quiet_output(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def _next_quiet(iterator, quiet: bool):
    with _quiet_output(quiet):
        return next(iterator)


def _set_seed(seed: int | None) -> None:
    if seed is None or seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _configure_logs(verbose: bool) -> None:
    if verbose:
        return
    os.environ.setdefault("TQDM_DISABLE", "1")
    logging.getLogger().setLevel(logging.WARNING)
    for name in (
        "tts_inference",
        "aqua.t2s",
        "httpx",
        "httpcore",
        "urllib3",
        "gradio",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)


def _default_path(home: str, *relative_parts: str) -> str:
    if not home:
        return ""
    return str(Path(home, *relative_parts))


def _require_path(label: str, value: str) -> str:
    if not value:
        raise SystemExit(f"{label} is required. Pass it explicitly or set GPT_SOVITS_HOME.")
    path = Path(value)
    if not path.exists():
        raise SystemExit(f"{label} does not exist: {path}")
    return str(path)


def _load_pyaudio():
    try:
        import pyaudio
    except ImportError as exc:
        raise SystemExit(
            "PyAudio is required for speaker playback. Install it with:\n"
            '  pip install -e ".[playback]"\n'
            "or install PyAudio manually for your platform."
        ) from exc
    return pyaudio


def _float32_mono(chunk) -> np.ndarray:
    audio = np.asarray(chunk, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.reshape(-1)
    return np.clip(audio, -1.0, 1.0)


def _write_wav(path: Path, sample_rate: int, chunks: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16.tobytes())


def _summarize_t2s(stats: list[dict]) -> tuple[int, float, float]:
    tokens = sum(int(stat.get("tokens", 0)) for stat in stats)
    elapsed = sum(float(stat.get("elapsed_sec", 0.0)) for stat in stats)
    rate = tokens / elapsed if elapsed > 0 else 0.0
    return tokens, elapsed, rate


def _fade_edges(audio: np.ndarray, sample_rate: int, *, fade_in: bool, fade_out: bool) -> np.ndarray:
    fade_n = max(1, int(0.010 * sample_rate))
    if not fade_in and not fade_out:
        return audio
    audio = audio.copy()
    if fade_in:
        n = min(fade_n, len(audio))
        audio[:n] *= np.linspace(0.0, 1.0, n, dtype=np.float32)
    if fade_out:
        n = min(fade_n, len(audio))
        audio[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
    return audio


def _parse_args():
    parser = argparse.ArgumentParser(description="Aqua-TTS end-to-end playback demo")
    parser.add_argument("--gpt-sovits-home", default=os.environ.get("GPT_SOVITS_HOME", ""))
    parser.add_argument("--gpt-model", default=os.environ.get("AQUA_GPT_MODEL"))
    parser.add_argument("--sovits-model", default=os.environ.get("AQUA_SOVITS_MODEL"))
    parser.add_argument("--ref-audio", default=os.environ.get("AQUA_REF_AUDIO"))
    parser.add_argument("--ref-text", default=os.environ.get("TTS_REF_TEXT_JA", KURISU_REF_TEXT))
    parser.add_argument("--text", action="append",
                        help="Custom text to play. Repeat for multiple utterances.")
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
    parser.add_argument("--pause", type=float, default=0.45,
                        help="Seconds to wait between demo utterances.")
    parser.add_argument("--output-device-index", type=int, default=None)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--save-dir", default="",
                        help="Optional directory for writing each utterance as WAV.")
    parser.add_argument("--show-total", action="store_true",
                        help="Also print audio duration, wall time, and RTF.")
    parser.add_argument("--no-warmup", action="store_true",
                        help="Skip the silent warmup request before playback.")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Sampling seed for repeatable demo output; set -1 for random.")
    parser.add_argument("--verbose", action="store_true",
                        help="Show internal model logs and tqdm progress.")
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _warmup(tts, args, ref_audio: str, quiet: bool = True) -> None:
    print("[demo] Warmup streaming path...")
    start = time.perf_counter()
    iterator = tts.infer_stream(
        text=WARMUP_TEXT,
        ref_audio_path=ref_audio,
        prompt_text=args.ref_text,
        text_language=args.text_lang,
        prompt_language=args.ref_lang,
        how_to_cut="按标点符号切",
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        speed=args.speed,
        sample_steps=args.sample_steps,
        enable_cuda_graph=not args.no_cuda_graph,
        enable_static_kv=True,
        chunk_size_seconds=args.chunk_size_seconds,
        collect_t2s_stats=True,
    )
    while True:
        try:
            _sr, chunk, _text = _next_quiet(iterator, quiet)
        except StopIteration:
            break
        if chunk is not None and len(chunk) > 0:
            break
    print(f"[demo] Warmup done in {time.perf_counter() - start:.2f}s")


def play_utterance(tts, pa, pyaudio, args, ref_audio: str, label: str, text: str,
                   save_dir: Path | None = None) -> dict:
    print(f"\n[{label}] {text}")
    _set_seed(args.seed)
    stream = None
    first_audio_ms = None
    sample_rate = 24000
    total_samples = 0
    chunks: list[np.ndarray] = []
    t2s_stats_start = len(tts.t2s_stats)
    start = time.perf_counter()

    iterator = tts.infer_stream(
        text=text,
        ref_audio_path=ref_audio,
        prompt_text=args.ref_text,
        text_language=args.text_lang,
        prompt_language=args.ref_lang,
        how_to_cut="按标点符号切",
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        speed=args.speed,
        sample_steps=args.sample_steps,
        enable_cuda_graph=not args.no_cuda_graph,
        enable_static_kv=True,
        chunk_size_seconds=args.chunk_size_seconds,
        collect_t2s_stats=True,
    )
    audio_queue: queue.Queue = queue.Queue(maxsize=32)
    producer_errors: list[BaseException] = []

    def _produce_audio():
        try:
            while True:
                try:
                    sr, chunk, _text = _next_quiet(iterator, not args.verbose)
                except StopIteration:
                    break
                sample = sr or 24000
                if chunk is None or len(chunk) == 0:
                    continue
                audio_queue.put((sample, _float32_mono(chunk)))
        except BaseException as exc:  # propagate after the player drains
            producer_errors.append(exc)
        finally:
            audio_queue.put(None)

    producer = threading.Thread(target=_produce_audio, name=f"aqua-producer-{label}", daemon=True)
    producer.start()

    while True:
        item = audio_queue.get()
        if item is None:
            break
        sample_rate, audio = item
        chunks_to_play = [audio]
        eof_in_drain = False
        while True:
            try:
                nxt = audio_queue.get_nowait()
            except queue.Empty:
                break
            if nxt is None:
                eof_in_drain = True
                break
            sample_rate, next_audio = nxt
            chunks_to_play.append(next_audio)

        merged = np.concatenate(chunks_to_play) if len(chunks_to_play) > 1 else chunks_to_play[0]
        merged = _fade_edges(
            merged,
            sample_rate,
            fade_in=first_audio_ms is None,
            fade_out=eof_in_drain,
        )
        if first_audio_ms is None:
            first_audio_ms = (time.perf_counter() - start) * 1000.0
            stream = pa.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=sample_rate,
                output=True,
                output_device_index=args.output_device_index,
                frames_per_buffer=max(256, min(2048, len(audio))),
            )

        stream.write(merged.tobytes())
        total_samples += len(merged)
        if save_dir is not None:
            chunks.append(merged.copy())
        if eof_in_drain:
            pad = np.zeros(int(0.060 * sample_rate), dtype=np.float32)
            stream.write(pad.tobytes())
            break

    producer.join(timeout=1.0)
    if producer_errors:
        raise producer_errors[0]

    if stream is not None:
        stream.stop_stream()
        stream.close()

    elapsed = time.perf_counter() - start
    audio_sec = total_samples / float(sample_rate)
    rtf = elapsed / audio_sec if audio_sec > 0 else float("inf")
    t2s_tokens, t2s_elapsed, t2s_rate = _summarize_t2s(tts.t2s_stats[t2s_stats_start:])
    first_audio_text = f"{first_audio_ms:.1f}ms" if first_audio_ms is not None else "n/a"
    t2s_detail = f" ({t2s_tokens} tokens/{t2s_elapsed:.3f}s)" if args.verbose else ""
    summary = f"[{label}] first_audio={first_audio_text} | t2s_live={t2s_rate:.0f} it/s{t2s_detail}"
    if getattr(args, "show_total", False):
        summary += f" | audio={audio_sec:.2f}s | elapsed={elapsed:.2f}s | rtf={rtf:.2f}x"
    print(summary)

    if save_dir is not None:
        wav_path = save_dir / f"{label}.wav"
        _write_wav(wav_path, sample_rate, chunks)
        print(f"[{label}] saved: {wav_path}")

    return {
        "first_audio_ms": first_audio_ms if first_audio_ms is not None else float("nan"),
        "t2s_tokens": t2s_tokens,
        "t2s_elapsed": t2s_elapsed,
        "t2s_rate": t2s_rate,
        "audio_sec": audio_sec,
        "elapsed": elapsed,
        "rtf": rtf,
    }


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

    print("[demo] Loading TTS pipeline...")
    t0 = time.perf_counter()
    with _quiet_output(not args.verbose):
        tts = TTSInferencer(
            device=args.device,
            gpt_path=gpt_model,
            sovits_path=sovits_model,
        )
    print(f"[demo] Loaded in {time.perf_counter() - t0:.2f}s")

    if not args.no_warmup:
        _set_seed(args.seed)
        _warmup(tts, args, ref_audio, quiet=not args.verbose)

    texts = [(f"text{i + 1}", text) for i, text in enumerate(args.text)] if args.text else KURISU_DEMO_TEXTS
    save_dir = Path(args.save_dir) if args.save_dir else None

    try:
        for label, text in texts:
            play_utterance(tts, pa, pyaudio, args, ref_audio, label, text, save_dir)
            if args.pause > 0:
                time.sleep(args.pause)
    finally:
        pa.terminate()


if __name__ == "__main__":
    main()
