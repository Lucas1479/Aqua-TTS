"""Storage ownership contracts run in model-less CI."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from aquatts.inference.workspace import GraphWorkspaceRegistry, resolve_kv_cache_layout


def test_layout_default_and_explicit_override(monkeypatch):
    monkeypatch.delenv("AQUATTS_KV_CACHE_LAYOUT", raising=False)
    assert resolve_kv_cache_layout() == "bucket"
    monkeypatch.setenv("AQUATTS_KV_CACHE_LAYOUT", "nested")
    assert resolve_kv_cache_layout() == "nested"
    assert resolve_kv_cache_layout(" BUCKET ") == "bucket"


@pytest.mark.parametrize("layout", ["graph", "auto", "", False])
def test_unknown_layout_is_visible(layout):
    with pytest.raises(ValueError, match="bucket or nested"):
        resolve_kv_cache_layout(layout)


def test_same_owner_cannot_enter_between_request_steps():
    registry = GraphWorkspaceRegistry()
    trying, entered = Event(), Event()

    def contender():
        trying.set()
        with registry.lease("shared", "cpu"):
            entered.set()

    with ThreadPoolExecutor(1) as pool:
        with registry.lease("shared", "cpu"):
            future = pool.submit(contender)
            assert trying.wait(5)
            assert not entered.wait(.05)
        future.result(timeout=5)
    assert entered.is_set()


def test_independent_owners_and_registries_do_not_share_a_lock():
    first, second = GraphWorkspaceRegistry(), GraphWorkspaceRegistry()

    def enter(registry, key):
        with registry.lease(key, "cpu"):
            return True

    with ThreadPoolExecutor(1) as pool, first.lease("a", "cpu"):
        assert pool.submit(enter, first, "b").result(timeout=5)
        assert pool.submit(enter, second, "a").result(timeout=5)


class FakeCUDA:
    def __init__(self):
        self.events = []
        self.stream = self
        self.waits = []

    def current_stream(self, device):
        assert device == "cuda:1"
        return self.stream

    def wait_event(self, event):
        self.waits.append(event)

    def Event(self):
        owner = self

        class Completion:
            def record(self, stream):
                owner.events.append((self, stream))

        return Completion()


def test_nested_lease_records_one_event_and_next_owner_waits():
    cuda = FakeCUDA()
    registry = GraphWorkspaceRegistry(cuda)
    with registry.lease(256, "cuda:1"):
        with registry.lease(256, "cuda:1"):
            assert not cuda.events
    assert len(cuda.events) == 1
    first = cuda.events[0][0]
    with registry.lease(256, "cuda:1"):
        assert cuda.waits == [first]
    assert len(cuda.events) == 2


def test_exception_returns_owner_and_preserves_gpu_handoff():
    cuda = FakeCUDA()
    registry = GraphWorkspaceRegistry(cuda)
    with pytest.raises(ValueError, match="request failed"):
        with registry.lease(256, "cuda:1"):
            raise ValueError("request failed")
    with ThreadPoolExecutor(1) as pool:
        def next_request():
            with registry.lease(256, "cuda:1"):
                return True
        assert pool.submit(next_request).result(timeout=5)
    assert len(cuda.events) == 2
    assert cuda.waits == [cuda.events[0][0]]
