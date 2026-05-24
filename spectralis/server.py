# -*- coding: utf-8 -*-
"""Lightweight HTTP API for Spectralis-TTS streaming inference.

Usage::

    python -m spectralis.server --host 127.0.0.1 --port 8000

Endpoints
---------
GET  /health          — model load status
POST /tts             — text to speech (streaming audio/wav response)
POST /tts/file        — text to speech (downloadable .wav file)
GET  /presets         — list available generation presets
"""

from __future__ import annotations

import io
import logging
import wave
from typing import Optional

import numpy as np

logger = logging.getLogger("spectralis.server")


# ── Internal helpers ──────────────────────────────────────────────────────

def _build_wav(audio: np.ndarray, sample_rate: int = 24000) -> bytes:
    """Pack float32 mono audio into a WAV byte buffer."""
    buf = io.BytesIO()
    audio_int16 = (audio * 32767).astype(np.int16)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return buf.getvalue()


def _create_app(inferencer):
    """Build a FastAPI app wired to the given TTSInferencer instance."""
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import Response, StreamingResponse

    app = FastAPI(
        title="Spectralis-TTS",
        version="1.0.0",
        description="GPU-optimized GPT-SoVITS streaming TTS",
    )

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "device": str(inferencer.device),
            "gpt_loaded": inferencer.t2s_model is not None,
            "sovits_loaded": inferencer.sovits_model is not None,
            "cuda_graph_preset": getattr(inferencer, "_cuda_graph_preset", "unknown"),
        }

    @app.get("/presets")
    async def list_presets():
        from spectralis.inference.presets import (
            GENERATION_PRESETS,
            CUDA_GRAPH_PRESETS,
        )
        return {
            "generation": {k: v for k, v in GENERATION_PRESETS.items()},
            "cuda_graph": {k: v for k, v in CUDA_GRAPH_PRESETS.items()},
        }

    @app.post("/tts")
    async def tts_stream(
        text: str = Query(..., description="Text to synthesize"),
        ref_audio_path: str = Query(..., description="Path to reference audio (.wav)"),
        prompt_text: Optional[str] = Query(None, description="Reference transcript"),
        text_language: str = Query("日文"),
        prompt_language: str = Query("日文"),
        preset: Optional[str] = Query(None, description="Quality preset: fast, balanced, quality"),
    ):
        """Streaming TTS — returns audio/wav chunks as they are generated.

        Yields PCM audio chunks as multipart stream. Low TTFP (time-to-first-packet)
        due to Spectralis CUDA Graph + static KV cache optimizations.
        """
        if not text.strip():
            raise HTTPException(400, "text must not be empty")
        if not ref_audio_path.strip():
            raise HTTPException(400, "ref_audio_path is required")

        def _generate():
            try:
                for sr, chunk, _text in inferencer.infer_stream(
                    text=text,
                    ref_audio_path=ref_audio_path,
                    prompt_text=prompt_text,
                    text_language=text_language,
                    prompt_language=prompt_language,
                    preset=preset,
                    how_to_cut="不切",
                ):
                    if chunk is not None and len(chunk) > 0:
                        yield (chunk.astype(np.float32).tobytes() if chunk.dtype != np.float32
                               else chunk.tobytes())
            except Exception as exc:
                logger.error(f"TTS stream error: {exc}")
                raise

        return StreamingResponse(
            _generate(),
            media_type="application/octet-stream",
            headers={
                "X-Sample-Rate": "24000",
                "X-Channels": "1",
                "X-Dtype": "float32",
            },
        )

    @app.post("/tts/file")
    async def tts_file(
        text: str = Query(..., description="Text to synthesize"),
        ref_audio_path: str = Query(..., description="Path to reference audio (.wav)"),
        prompt_text: Optional[str] = Query(None, description="Reference transcript"),
        text_language: str = Query("日文"),
        prompt_language: str = Query("日文"),
        preset: Optional[str] = Query(None, description="Quality preset: fast, balanced, quality"),
    ):
        """One-shot TTS — collects all chunks and returns a downloadable .wav file."""
        if not text.strip():
            raise HTTPException(400, "text must not be empty")
        if not ref_audio_path.strip():
            raise HTTPException(400, "ref_audio_path is required")

        chunks = []
        sample_rate = 24000
        try:
            for sr, chunk, _text in inferencer.infer_stream(
                text=text,
                ref_audio_path=ref_audio_path,
                prompt_text=prompt_text,
                text_language=text_language,
                prompt_language=prompt_language,
                preset=preset,
                how_to_cut="不切",
            ):
                if chunk is not None and len(chunk) > 0:
                    chunks.append(chunk)
                    sample_rate = sr
        except Exception as exc:
            logger.error(f"TTS error: {exc}")
            raise HTTPException(500, str(exc))

        if not chunks:
            raise HTTPException(500, "No audio generated")

        audio = np.concatenate(chunks)
        wav_bytes = _build_wav(audio, sample_rate)
        return Response(content=wav_bytes, media_type="audio/wav")

    return app


# ── Public API ─────────────────────────────────────────────────────────────

def start_server(
    inferencer,
    host: str = "127.0.0.1",
    port: int = 8000,
    log_level: str = "info",
):
    """Start the Spectralis-TTS HTTP server (blocking).

    Args:
        inferencer: A pre-configured TTSInferencer instance.
        host: Bind address.
        port: Bind port.
        log_level: Uvicorn log level.
    """
    import uvicorn

    app = _create_app(inferencer)
    print(f"Spectralis-TTS server starting on http://{host}:{port}")
    print(f"  Health:  http://{host}:{port}/health")
    print(f"  Presets: http://{host}:{port}/presets")
    print(f"  TTS:     http://{host}:{port}/tts?text=...&ref_audio_path=...")
    uvicorn.run(app, host=host, port=port, log_level=log_level)


# ── CLI ────────────────────────────────────────────────────────────────────

def _main():
    """Entry point: python -m spectralis.server"""
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Spectralis-TTS HTTP Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpt-model", required=True, help="Path to GPT T2S checkpoint (.ckpt)")
    parser.add_argument("--sovits-model", required=True, help="Path to SoVITS checkpoint (.pth)")
    parser.add_argument("--cuda-graph-preset", default="full",
                        choices=["full", "minimal", "lazy", "off"],
                        help="CUDA Graph capture strategy")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.environ.setdefault("ENABLE_CUDA_GRAPH", "1")

    # Import triggers sys.path setup in spectralis/__init__.py
    from spectralis import TTSInferencer

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    logger.info("Loading TTS pipeline...")
    tts = TTSInferencer(
        device=args.device,
        gpt_path=args.gpt_model,
        sovits_path=args.sovits_model,
        cuda_graph_preset=args.cuda_graph_preset,
    )
    logger.info("TTS pipeline ready.")

    start_server(tts, host=args.host, port=args.port)


if __name__ == "__main__":
    _main()
