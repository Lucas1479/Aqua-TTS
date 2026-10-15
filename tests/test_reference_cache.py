"""Reference normalization and publication contracts without optional Torch."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import threading

import pytest

from aquatts.inference.reference_cache import ReferenceCache, normalize_reference, reference_key


def normalized(path, text="Reference", language="英文", **kwargs):
    options = dict(model_version="v3", languages={"英文": "en", "日文": "all_ja"},
                   splits={".", "。", ",", "，"}, fallback_text="既定の参照。", fallback_language="all_ja")
    options.update(kwargs)
    return normalize_reference(path, text, language, **options)


def test_alias_path_and_punctuation_share_identity(tmp_path, monkeypatch):
    path = tmp_path / "reference.wav"
    path.write_bytes(b"reference")
    monkeypatch.chdir(tmp_path)
    first = normalized("reference.wav", " Reference ", "英文")
    second = normalized(path, "Reference.", "en")
    assert first == second
    assert normalized(path, "Reference,", "en").text == "Reference,"
    assert normalized(path, "参照", "日文") == normalized(path, "参照。", "all_ja")


@pytest.mark.parametrize("version,free", [("v1", True), ("v2", True), ("v2Pro", True), ("v3", False)])
def test_empty_text_and_v3_fallback_are_shared(tmp_path, version, free):
    first = normalized(tmp_path / "a.wav", " ", "英文", model_version=version)
    assert first.ref_free is free
    assert first.text == ("" if free else "既定の参照。")
    assert first.language == ("" if free else "all_ja")
    assert first == normalized(tmp_path / "a.wav", None, "日文", model_version=version)


def test_file_replacement_and_runtime_changes_invalidate_keys(tmp_path):
    path = tmp_path / "a.wav"
    path.write_bytes(b"a")
    ref = normalized(path)
    context = dict(generation=0, version="v3", device="cuda:0", dtype="float16", is_pro=False)
    initial = reference_key(ref, **context)
    for field, value in [("generation", 1), ("device", "cuda:1"), ("dtype", "float32"), ("version", "v2")]:
        assert reference_key(ref, **{**context, field: value}) != initial
    path.write_bytes(b"replacement")
    assert reference_key(ref, **context) != initial
    assert reference_key(replace(ref, text="Changed."), **context) != reference_key(ref, **context)


def test_fifo_capacity_and_failed_build_do_not_publish_or_evict():
    cache = ReferenceCache(2)
    first, status, _ = cache.get_or_build("a", lambda: object())
    assert status == "built"
    cache.get_or_build("b", lambda: object())
    assert cache.get_or_build("a", lambda: pytest.fail("rebuilt hit"))[:2] == (first, "hit")
    def fail():
        raise RuntimeError("partial build")
    with pytest.raises(RuntimeError, match="partial"):
        cache.get_or_build("c", fail)
    assert len(cache) == 2 and cache.contains("a") and not cache.contains("c")
    assert cache.get_or_build("c", object)[2] == "a"
    assert cache.clear() == 2 and len(cache) == 0


def test_concurrent_same_key_builds_once_and_waits_for_ready():
    entered, release = threading.Event(), threading.Event()
    calls, waits = [], []
    marker = object()
    cache = ReferenceCache(2, record_ready=lambda: marker, wait_ready=waits.append)
    def build():
        calls.append(True)
        entered.set()
        assert release.wait(5)
        return {"complete": True}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(cache.get_or_build, "key", build) for _ in range(4)]
        assert entered.wait(5)
        release.set()
        results = [future.result(timeout=5) for future in futures]
    assert calls == [True] and waits == [marker] * 3
    assert sum(status == "built" for _, status, _ in results) == 1
    assert all(value is results[0][0] for value, _, _ in results)


@pytest.mark.parametrize("capacity", [0, -1, True, "8", 1.5])
def test_invalid_capacity_is_rejected(capacity):
    with pytest.raises(ValueError, match="positive integer"):
        ReferenceCache(capacity)
