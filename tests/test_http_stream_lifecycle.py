import asyncio
import threading
import time
from urllib.parse import urlencode

import httpx
import numpy as np
import pytest

from aquatts.inference.streaming import locked_chunks
from aquatts.inference.cuda_execution import cuda_execution, cuda_work
from aquatts.server import _create_app
from aquatts.voice_registry import VoiceRegistry


class Decoder:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = 0
        self.maximum = 0
        self.closed = 0

    @cuda_work
    def infer_stream(self, **_):
        try:
            yield 24000, None, ""
            def chunks():
                for _ in range(5):
                    self.active += 1
                    self.maximum = max(self.maximum, self.active)
                    try:
                        time.sleep(.01)
                        chunk = np.full(128, .2, dtype=np.float32)
                    finally:
                        self.active -= 1
                    yield 24000, chunk, ""
            yield from locked_chunks(chunks(), self.lock)
        finally:
            self.closed += 1


def test_two_actual_http_streams_complete_with_mutual_exclusion():
    decoder = Decoder()
    app = _create_app(decoder, VoiceRegistry())
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            responses = await asyncio.wait_for(asyncio.gather(*[
                client.post("/tts", params={"text": "test", "ref_audio_path": "ref.wav"}) for _ in range(2)]), 10)
            assert all(r.status_code == 200 and len(r.content) == 5 * 128 * 4 for r in responses)
    asyncio.run(run())
    assert decoder.maximum == 1 and decoder.closed == 2
    assert not cuda_execution._owners
    assert decoder.lock.acquire(blocking=False)
    decoder.lock.release()


@pytest.mark.parametrize("cancel", [False, True])
def test_http_disconnect_or_cancellation_closes_generator(cancel):
    decoder = Decoder()
    app = _create_app(decoder, VoiceRegistry())
    async def run():
        entered = asyncio.Event()
        async def receive():
            await asyncio.Event().wait()
        async def send(event):
            if event["type"] == "http.response.body" and event.get("body"):
                entered.set()
                if cancel:
                    await asyncio.Event().wait()
                raise OSError("client disconnected")
        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
                 "http_version": "1.1", "method": "POST", "scheme": "http", "path": "/tts",
                 "query_string": urlencode({"text": "test", "ref_audio_path": "ref.wav"}).encode(),
                 "headers": [], "server": ("test", 80), "client": ("test", 1), "root_path": ""}
        task = asyncio.create_task(app(scope, receive, send))
        await asyncio.wait_for(entered.wait(), 5)
        if cancel:
            task.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else Exception):
            await asyncio.wait_for(task, 5)
    asyncio.run(run())
    assert decoder.closed == 1
    assert not cuda_execution._owners
    assert decoder.lock.acquire(blocking=False)
    decoder.lock.release()



@pytest.mark.parametrize("finalize", [False, True])
def test_private_generator_closes_while_decode_lock_is_held(finalize):
    """Explicit close and GC on a lock-owning thread must not reacquire it."""
    import gc

    lock = threading.Lock()
    finished, closed = threading.Event(), threading.Event()
    errors = []

    def chunks():
        try:
            yield "audio"
        finally:
            closed.set()

    def worker():
        try:
            stream = locked_chunks(chunks(), lock)
            assert next(stream) == "audio"
            lock.acquire()
            try:
                if finalize:
                    del stream
                    gc.collect()
                else:
                    stream.close()
            finally:
                lock.release()
        except BaseException as error:
            errors.append(error)
        finally:
            finished.set()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        assert finished.wait(3), "closing a private generator reacquired the decode lock"
        assert not errors
        assert closed.is_set()
    finally:
        # Release a regressed implementation instead of leaving a blocked worker.
        if not finished.is_set() and lock.locked():
            lock.release()
        thread.join(timeout=3)
