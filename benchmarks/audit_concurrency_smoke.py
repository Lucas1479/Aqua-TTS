"""Real V3 checks for worker migration, capture deferral and reference decoding."""
from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import httpx
    import numpy as np
    import torch
    import torchaudio
    from aquatts import TTSInferencer, VoiceRegistry
    from aquatts.inference.cuda_execution import cuda_execution
    from aquatts.server import _create_app

    def reject_torchcodec(*_args, **_kwargs):
        raise AssertionError("reference decoding entered torchaudio.load")
    torchaudio.load = reject_torchcodec
    config = json.loads(args.config.read_text(encoding="utf-8"))
    first = TTSInferencer(**config["inferencer"], cuda_graph_preset="lazy", use_flash_attn=True)
    common = {**config["reference"], "text": "This is a short test of voice synthesis.",
              "text_language": "英文", "top_k": 5, "top_p": 1., "temperature": .6,
              "sample_steps": 4, "how_to_cut": "不切", "pause_second": 0, "chunk_size_seconds": .25}
    paused = first.infer_stream(**common)
    assert next(paused) == (24000, None, "")
    assert len(cuda_execution._owners) == 1
    # A second constructor also performs vocoder warmup (CUDA RNG/synchronize).
    second = TTSInferencer(**config["inferencer"], cuda_graph_preset="minimal", use_flash_attn=True)
    assert not second.t2s_model.model.bucket_graphs, "constructor captured while another request was alive"
    paused.close()
    assert not cuda_execution._owners
    rows = []
    def audio_check(audio):
        assert audio.size > 2400 and np.isfinite(audio).all() and np.max(np.abs(audio)) > 1e-4

    original_stream = first.infer_stream
    def http_stream(**kwargs):
        return original_stream(**kwargs, chunk_size_seconds=.25, sample_steps=4, top_k=5,
                               top_p=1., temperature=.6, pause_second=0)
    first.infer_stream = http_stream
    app = _create_app(first, VoiceRegistry())
    async def reference_errors():
        failures = []
        # A real decoder failure after normalization, rather than a missing path
        # rejected before reference preparation begins.
        with tempfile.TemporaryDirectory() as directory:
            corrupt = Path(directory) / "corrupt.wav"
            corrupt.write_bytes(b"not an audio container")
            params = {**config["reference"], "ref_audio_path": str(corrupt),
                      "text": common["text"], "text_language": "英文"}
            transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                for endpoint in ("/tts", "/tts/file"):
                    response = await asyncio.wait_for(client.post(endpoint, params=params), 30)
                    assert response.status_code == 500, (endpoint, response.status_code)
                    assert "x-sample-rate" not in response.headers
                    assert "detail" in response.json()
                    assert not cuda_execution._owners
                    failures.append({"endpoint": endpoint, "status": response.status_code,
                                     "sample_rate_header": False, "active_work_after": 0})
        return failures
    reference_error_results = asyncio.run(reference_errors())

    async def http_requests():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            start = time.perf_counter()
            params = {**config["reference"], "text": common["text"], "text_language": "英文"}
            responses = await asyncio.wait_for(asyncio.gather(*[client.post("/tts", params=params) for _ in range(2)]), 90)
            for response in responses:
                assert response.status_code == 200 and response.headers["x-sample-rate"] == "24000"
                audio = np.frombuffer(response.content, dtype=np.float32)
                audio_check(audio)
                rows.append({"case": "same_instance_http", "samples": int(audio.size)})
            return time.perf_counter() - start
    http_seconds = asyncio.run(http_requests())
    first.infer_stream = original_stream
    assert not cuda_execution._owners

    barrier = threading.Barrier(2)
    def request(model):
        barrier.wait(timeout=10)
        chunks = [chunk for _, chunk, _ in model.infer_stream(**common) if chunk is not None and chunk.size]
        audio = np.concatenate(chunks)
        audio_check(audio)
        return {"case": "separate_instances", "samples": int(audio.size)}
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(request, model) for model in (first, second)]
        rows.extend(future.result(timeout=90) for future in futures)
    assert not cuda_execution._owners
    retry = second.t2s_model.model.precapture_cuda_graph([256])
    assert retry[256], "deferred capture did not become available after work completed"
    sequential = np.concatenate([chunk for _, chunk, _ in second.infer_stream(**common) if chunk is not None and chunk.size])
    audio_check(sequential)
    assert not cuda_execution._owners
    report = {"gpu": torch.cuda.get_device_name(first.device), "torch": torch.__version__,
              "constructor_capture_deferred": True, "capture_retry_passed": True,
              "torchcodec_load_blocked": True, "same_instance_http_seconds": http_seconds,
              "reference_error_http": reference_error_results,
              "trials": rows, "sequential_samples": int(sequential.size),
              "active_work_after_completion": len(cuda_execution._owners),
              "scope": "Functional concurrent real-model smoke, not a performance comparison; all requests use real sampling and CFM randomness."}
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
