# -*- coding: utf-8 -*-
"""Lightweight HTTP API for Spectralis-TTS streaming inference. / Spectralis-TTS 流式推理的轻量级 HTTP API。

Usage / 用法::

    python -m spectralis.server --host 127.0.0.1 --port 8000

Endpoints / 端点
---------
GET  /health           — model load status / 模型加载状态
POST /tts              — text to speech (streaming audio/wav response) / 文本转语音（流式音频/wav 响应）
POST /tts/file         — text to speech (downloadable .wav file) / 文本转语音（可下载 .wav 文件）
GET  /presets          — list available generation presets / 列出可用的生成预设
GET  /voices           — list registered voices / 列出已注册的声音
POST /voices/add       — register a new voice / 注册新声音
DELETE /voices/{name}  — remove a registered voice / 移除已注册的声音
"""

from __future__ import annotations

import io
import logging
import os
import wave
from typing import Optional

import numpy as np

from spectralis.voice_registry import Voice, VoiceRegistry, registry_from_env

logger = logging.getLogger("spectralis.server")


# ── Internal helpers (内部辅助函数) ──────────────────────────────────────────────────────

def _build_wav(audio: np.ndarray, sample_rate: int = 24000) -> bytes:
    """Pack float32 mono audio into a WAV byte buffer. / 将 float32 单声道音频打包为 WAV 字节缓冲区。"""
    buf = io.BytesIO()
    audio_int16 = (audio * 32767).astype(np.int16)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return buf.getvalue()


def _create_app(inferencer, voice_registry=None):
    """Build a FastAPI app wired to the given TTSInferencer instance.
    / 构建一个与给定 TTSInferencer 实例连接的 FastAPI 应用。

    Args / 参数:
        inferencer: A configured TTSInferencer. / 已配置的 TTSInferencer 实例。
        voice_registry: Optional VoiceRegistry. If None, uses
            ``registry_from_env()`` which reads SPECTRALIS_VOICE_JSON
            or falls back to ``./voices.json``.
            / 可选的 VoiceRegistry。如果为 None，则使用 ``registry_from_env()``，
            该函数读取 SPECTRALIS_VOICE_JSON 环境变量，否则回退到 ``./voices.json``。
    """
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import Response, StreamingResponse
    from pydantic import BaseModel

    if voice_registry is None:
        voice_registry = registry_from_env()

    def _resolve_voice(voice_name, ref_audio_path, prompt_text, prompt_language):
        """Resolve a voice name to (ref_audio_path, prompt_text, prompt_language).
        / 将声音名称解析为 (ref_audio_path, prompt_text, prompt_language)。

        If ``voice_name`` is given, looks it up in the registry and returns
        the stored values (overriding per-request params). Otherwise returns
        the per-request params unchanged.
        / 如果给出了 ``voice_name``，则从注册表中查找并返回存储的值（覆盖每次请求的参数）。
        否则按原样返回每次请求的参数。
        """
        if voice_name:
            voice = voice_registry.get(voice_name)
            if voice is None:
                raise HTTPException(404, f"Voice {voice_name!r} not found")
            return voice.ref_audio_path, voice.prompt_text, voice.prompt_language
        return ref_audio_path, prompt_text, prompt_language

    app = FastAPI(
        title="Spectralis-TTS",
        version="1.0.0",
        description="GPU-optimized GPT-SoVITS streaming TTS",
    )

    class VoiceAddRequest(BaseModel):
        name: str
        ref_audio_path: str
        prompt_text: str = ""
        prompt_language: str = "日文"

    # ── Status & presets (状态与预设) ─────────────────────────────────────────────────

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

    # ── Voice management (声音管理) ─────────────────────────────────────────────────

    @app.get("/voices")
    async def list_voices():
        """List all registered voices. / 列出所有已注册的声音。"""
        return {"voices": [v.to_dict() for v in voice_registry.list()]}

    @app.post("/voices/add")
    async def add_voice(req: VoiceAddRequest):
        """Register a new voice or update an existing one. / 注册新声音或更新已有声音。"""
        voice = Voice(
            name=req.name,
            ref_audio_path=req.ref_audio_path,
            prompt_text=req.prompt_text,
            prompt_language=req.prompt_language,
        )
        voice_registry.add(voice)
        logger.info(f"Voice registered: {req.name}")
        return {"status": "ok", "voice": voice.to_dict()}

    @app.delete("/voices/{name}")
    async def remove_voice(name: str):
        """Remove a registered voice by name. / 按名称移除已注册的声音。"""
        if voice_registry.remove(name):
            logger.info(f"Voice removed: {name}")
            return {"status": "ok", "message": f"Voice {name!r} removed"}
        raise HTTPException(404, f"Voice {name!r} not found")

    # ── TTS endpoints (TTS 端点) ────────────────────────────────────────────────────

    @app.post("/tts")
    async def tts_stream(
        text: str = Query(..., description="Text to synthesize / 要合成的文本"),
        ref_audio_path: Optional[str] = Query(None, description="Path to reference audio (.wav) / 参考音频路径 (.wav)"),
        voice: Optional[str] = Query(None, description="Voice name from registry (overrides ref_audio_path) / 注册表中的声音名称（覆盖 ref_audio_path）"),
        prompt_text: Optional[str] = Query(None, description="Reference transcript / 参考文本转录"),
        text_language: str = Query("日文"),
        prompt_language: str = Query("日文"),
        preset: Optional[str] = Query(None, description="Quality preset: fast, balanced, quality / 质量预设：快速、均衡、高质量"),
    ):
        """Streaming TTS — returns audio/wav chunks as they are generated.
        / 流式 TTS — 在生成音频/wav 块的同时实时返回。

        Yields PCM audio chunks as multipart stream. Low TTFP (time-to-first-packet)
        due to Spectralis CUDA Graph + static KV cache optimizations.
        / 以多部分流的形式生成 PCM 音频块。利用 Spectralis CUDA Graph + 静态 KV 缓存优化，
        实现低首包延迟 (TTFP)。
        """
        if not text.strip():
            raise HTTPException(400, "text must not be empty")

        try:
            resolved_path, resolved_prompt_text, resolved_prompt_lang = _resolve_voice(
                voice, ref_audio_path, prompt_text, prompt_language
            )
        except HTTPException:
            raise

        if not resolved_path or not resolved_path.strip():
            raise HTTPException(400, "ref_audio_path or voice is required")

        def _generate():
            try:
                for sr, chunk, _text in inferencer.infer_stream(
                    text=text,
                    ref_audio_path=resolved_path,
                    prompt_text=resolved_prompt_text,
                    text_language=text_language,
                    prompt_language=resolved_prompt_lang,
                    preset=preset,
                    how_to_cut="不切",
                ):
                    if chunk is not None and len(chunk) > 0:
                        yield (
                            chunk.astype(np.float32).tobytes()
                            if chunk.dtype != np.float32
                            else chunk.tobytes()
                        )
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
        text: str = Query(..., description="Text to synthesize / 要合成的文本"),
        ref_audio_path: Optional[str] = Query(None, description="Path to reference audio (.wav) / 参考音频路径 (.wav)"),
        voice: Optional[str] = Query(None, description="Voice name from registry (overrides ref_audio_path) / 注册表中的声音名称（覆盖 ref_audio_path）"),
        prompt_text: Optional[str] = Query(None, description="Reference transcript / 参考文本转录"),
        text_language: str = Query("日文"),
        prompt_language: str = Query("日文"),
        preset: Optional[str] = Query(None, description="Quality preset: fast, balanced, quality / 质量预设：快速、均衡、高质量"),
    ):
        """One-shot TTS — collects all chunks and returns a downloadable .wav file. / 一次性 TTS — 收集所有音频块并返回可下载的 .wav 文件。"""
        if not text.strip():
            raise HTTPException(400, "text must not be empty")

        try:
            resolved_path, resolved_prompt_text, resolved_prompt_lang = _resolve_voice(
                voice, ref_audio_path, prompt_text, prompt_language
            )
        except HTTPException:
            raise

        if not resolved_path or not resolved_path.strip():
            raise HTTPException(400, "ref_audio_path or voice is required")

        chunks = []
        sample_rate = 24000
        try:
            for sr, chunk, _text in inferencer.infer_stream(
                text=text,
                ref_audio_path=resolved_path,
                prompt_text=resolved_prompt_text,
                text_language=text_language,
                prompt_language=resolved_prompt_lang,
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


# ── Public API (公共 API) ─────────────────────────────────────────────────────────────

def start_server(
    inferencer,
    host: str = "127.0.0.1",
    port: int = 8000,
    log_level: str = "info",
    voice_registry=None,
):
    """Start the Spectralis-TTS HTTP server (blocking). / 启动 Spectralis-TTS HTTP 服务器（阻塞模式）。

    Args / 参数:
        inferencer: A pre-configured TTSInferencer instance. / 已预先配置的 TTSInferencer 实例。
        host: Bind address. / 绑定地址。
        port: Bind port. / 绑定端口。
        log_level: Uvicorn log level. / Uvicorn 日志级别。
        voice_registry: Optional VoiceRegistry. If None, auto-creates from env.
            / 可选的 VoiceRegistry。如果为 None，则从环境变量自动创建。
    """
    import uvicorn

    app = _create_app(inferencer, voice_registry=voice_registry)
    print(f"Spectralis-TTS server starting on http://{host}:{port}")
    print(f"  Health:  http://{host}:{port}/health")
    print(f"  Presets: http://{host}:{port}/presets")
    print(f"  Voices:  http://{host}:{port}/voices")
    print(f"  TTS:     http://{host}:{port}/tts?text=...&voice=...")
    uvicorn.run(app, host=host, port=port, log_level=log_level)


# ── CLI (命令行接口) ────────────────────────────────────────────────────────────────────

def _main():
    """Entry point: python -m spectralis.server / 入口点：python -m spectralis.server"""
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
    parser.add_argument("--voice-registry", default=None,
                        help="Path to voice registry JSON file")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.environ.setdefault("ENABLE_CUDA_GRAPH", "1")

    # Import triggers sys.path setup in spectralis/__init__.py
    # 此导入触发 spectralis/__init__.py 中的 sys.path 设置
    from spectralis import TTSInferencer, VoiceRegistry

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    voice_registry = None
    if args.voice_registry:
        voice_registry = VoiceRegistry(json_path=args.voice_registry)
        logger.info(f"Voice registry loaded from {args.voice_registry}")

    logger.info("Loading TTS pipeline...")
    tts = TTSInferencer(
        device=args.device,
        gpt_path=args.gpt_model,
        sovits_path=args.sovits_model,
        cuda_graph_preset=args.cuda_graph_preset,
    )
    logger.info("TTS pipeline ready.")

    start_server(tts, host=args.host, port=args.port, voice_registry=voice_registry)


if __name__ == "__main__":
    _main()
