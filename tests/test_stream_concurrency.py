"""Real infer_stream generator scheduling; no checkpoints are required."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock
import asyncio
from functools import partial
import threading
import time

import pytest

torch = pytest.importorskip("torch")
module = pytest.importorskip("aquatts.inferencer")


@pytest.fixture
def tiny_v3():
    value = module.TTSInferencer.__new__(module.TTSInferencer)
    value.device, value.is_half = torch.device("cpu"), False
    value.model_version, value.is_v2pro = "v3", False
    value.dict_language = {"英文": "en", "日文": "all_ja"}
    value.splits = {".", "。"}
    value.hps = SimpleNamespace(data=SimpleNamespace(sampling_rate=24000))
    value.hz, value.max_sec = 50, 30
    value._stream_sync_timing_enabled = False
    value._init_reference_cache()
    entry = {"prompt": torch.ones(1, 4), "phones1": [1, 2, 3], "bert1": torch.zeros(1024, 3),
             "refer_spec": torch.ones(1, 4, 8), "prompt_fea_ref": torch.ones(1, 8, 12),
             "prompt_ge": torch.ones(1, 1), "mel2_norm": torch.ones(1, 100, 12)}
    value._build_session_cache = lambda *_: entry
    value.get_phones_and_bert = lambda *_: ([4, 5, 6], torch.zeros(1024, 3), "Text.")
    value._infer_semantic_with_guard = Mock(return_value=(torch.ones(1, 1, 8), 8, 1))
    value.vq_model = SimpleNamespace(decode_encp=lambda *_: (torch.ones(1, 8, 30), torch.ones(1, 1)))
    value._run_cfm = lambda fea, *_: torch.ones(1, 100, fea.size(1))
    value._run_bigvgan = lambda mel: torch.full((1, 1, mel.size(-1) * 256), .25)
    return value


def test_actual_v3_stream_releases_decode_lock_between_next_calls(tiny_v3):
    stream = tiny_v3.infer_stream("Text", "reference.wav", "Reference", text_language="英文",
                                 how_to_cut="不切", pause_second=0, chunk_size_seconds=.1)
    with ThreadPoolExecutor(1) as first, ThreadPoolExecutor(1) as second:
        try:
            assert first.submit(next, stream).result(timeout=10)[1] is None
            assert first.submit(next, stream).result(timeout=10)[1].size
            def check_lock():
                acquired = tiny_v3._sovits_decode_lock.acquire(blocking=False)
                if acquired:
                    tiny_v3._sovits_decode_lock.release()
                return acquired
            assert second.submit(check_lock).result(timeout=5), "decode lock spans a yielded audio chunk"
            chunks = second.submit(list, stream).result(timeout=10)
            assert chunks and all(chunk is not None for _, chunk, _ in chunks)
        finally:
            first.submit(stream.close).result(timeout=5)


def test_two_http_requests_use_actual_v3_generator(tiny_v3):
    import httpx
    from aquatts.server import _create_app
    from aquatts.voice_registry import VoiceRegistry
    tiny_v3.infer_stream = partial(tiny_v3.infer_stream, chunk_size_seconds=.1, pause_second=0)
    original = tiny_v3._run_cfm
    guard = threading.Lock()
    active, maximum = 0, 0
    def cfm(*args):
        nonlocal active, maximum
        with guard:
            active += 1
            maximum = max(maximum, active)
        try:
            time.sleep(.01)
            return original(*args)
        finally:
            with guard:
                active -= 1
    tiny_v3._run_cfm = cfm
    app = _create_app(tiny_v3, VoiceRegistry())
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            responses = await asyncio.wait_for(asyncio.gather(*[
                client.post("/tts", params={"text": "Text", "ref_audio_path": "ref.wav", "prompt_text": "Reference",
                                             "text_language": "英文"}) for _ in range(2)]), 10)
            assert all(r.status_code == 200 and r.content for r in responses)
    asyncio.run(run())
    assert maximum == 1
    assert tiny_v3._sovits_decode_lock.acquire(blocking=False)
    tiny_v3._sovits_decode_lock.release()
