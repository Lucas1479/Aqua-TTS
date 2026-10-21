"""Request ownership for mutable CUDA Graph inputs, outputs and KV buffers."""
from contextlib import contextmanager
from dataclasses import dataclass, field
import threading
import time
import os


def resolve_kv_cache_layout(value=None):
    """Snapshot the caller's explicit layout or the Aqua environment default."""
    layout = os.environ.get("AQUATTS_KV_CACHE_LAYOUT", "bucket") if value is None else value
    if not isinstance(layout, str) or layout.strip().lower() not in {"bucket", "nested"}:
        raise ValueError("kv_cache_layout must be bucket or nested")
    return layout.strip().lower()


@dataclass
class _Entry:
    lock: object = field(default_factory=threading.RLock)
    local: object = field(default_factory=threading.local)
    completion: object = None


class GraphWorkspaceRegistry:
    """Serialize a storage owner and order its work across CUDA streams.

    Different keys may progress independently. Nested calls on the same thread
    reuse the outer lease, which records the completion event exactly once.
    Torch is imported only for CUDA leases; CPU contract tests need no runtime.
    """

    def __init__(self, cuda=None):
        self._cuda = cuda
        self._entries = {}
        self._guard = threading.Lock()

    def _entry(self, key):
        with self._guard:
            if key not in self._entries:
                self._entries[key] = _Entry()
            return self._entries[key]

    def lock_for(self, key):
        return self._entry(key).lock

    @contextmanager
    def lease(self, key, device):
        entry = self._entry(key)
        started = time.perf_counter()
        with entry.lock:
            if getattr(entry.local, "active", False):
                yield 0.0
                return
            entry.local.active = True
            stream = None
            cuda = self._cuda
            try:
                if str(device).split(":", 1)[0] == "cuda":
                    if cuda is None:
                        from torch import cuda
                    stream = cuda.current_stream(device)
                    if entry.completion is not None:
                        stream.wait_event(entry.completion)
                yield time.perf_counter() - started
            finally:
                try:
                    if stream is not None:
                        completed = cuda.Event()
                        completed.record(stream)
                        entry.completion = completed
                finally:
                    entry.local.active = False
