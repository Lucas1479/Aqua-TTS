"""Reference identity and bounded, atomic publication without model imports."""
from dataclasses import dataclass
import os
from pathlib import Path
import threading

REFERENCE_LANGUAGE_CODES = frozenset({"en", "all_zh", "all_ja", "all_ko", "all_yue",
                                      "zh", "ja", "ko", "yue", "auto", "auto_yue"})


@dataclass(frozen=True)
class NormalizedReference:
    audio_path: str
    text: str
    language: str
    ref_free: bool


def normalize_reference(audio_path, text, language, *, model_version, languages,
                        splits, fallback_text, fallback_language, ref_free=False):
    path = str(Path(audio_path).expanduser().resolve())
    text = (text or "").strip()
    language = languages.get(language, language if language in REFERENCE_LANGUAGE_CODES else fallback_language)
    ref_free = bool(ref_free or not text)
    if model_version == "v3":
        if not text:
            text, language = fallback_text.strip(), fallback_language
        ref_free = False
    if ref_free:
        text, language = "", ""
    elif text and text[-1] not in splits:
        text += "." if language == "en" else "。"
    return NormalizedReference(path, text, language, ref_free)


def reference_key(reference, *, generation, version, device, dtype, is_pro):
    stat = Path(reference.audio_path).stat()
    return (os.path.normcase(reference.audio_path), stat.st_mtime_ns, stat.st_size,
            reference.text, reference.language, reference.ref_free, generation,
            version, str(device), str(dtype), bool(is_pro))


class ReferenceCache:
    """Serialize construction; failed or incomplete builds are never published.

    Ready/wait callbacks can order CUDA streams without synchronizing the device.
    Insertion order is FIFO; hits do not extend a reference's residency.
    """
    def __init__(self, capacity, *, record_ready=lambda: None, wait_ready=lambda _: None):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("reference cache capacity must be a positive integer")
        self.capacity = capacity
        self._items = {}
        self._lock = threading.RLock()
        self._record_ready = record_ready
        self._wait_ready = wait_ready

    def get_or_build(self, key, build):
        with self._lock:
            if key in self._items:
                value, ready = self._items[key]
                self._wait_ready(ready)
                return value, "hit", None
            value = build()
            ready = self._record_ready()
            evicted = None
            if len(self._items) >= self.capacity:
                evicted = next(iter(self._items))
                del self._items[evicted]
            self._items[key] = (value, ready)
            return value, "built", evicted

    def clear(self):
        with self._lock:
            count = len(self._items)
            self._items.clear()
            return count

    def contains(self, key):
        with self._lock:
            return key in self._items

    def __len__(self):
        with self._lock:
            return len(self._items)
