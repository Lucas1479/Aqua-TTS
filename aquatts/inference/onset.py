"""Locate speech in mono float PCM without changing its voiced samples.

The generated lead is distinct from the caller's inter-segment pause. Detection
uses 10 ms RMS windows at -45 dBFS and retains 50 ms before the first active
window. This is an amplitude boundary, not a linguistic voice detector.
"""
from __future__ import annotations

import numpy as np


def voice_offset(audio, sample_rate: int, *, complete_only: bool = False) -> int | None:
    """Find the first active window; streaming callers defer incomplete windows."""
    values = np.asarray(audio).reshape(-1).astype(np.float64, copy=False)
    width = max(1, round(sample_rate / 100))
    complete = values.size // width * width
    threshold = 10 ** (-45 / 10)
    if complete:
        windows = values[:complete].reshape(-1, width)
        matches = np.flatnonzero(np.mean(windows * windows, axis=1) >= threshold)
        if matches.size:
            return int(matches[0]) * width
    if not complete_only and complete < values.size:
        tail = values[complete:]
        if np.mean(tail * tail) >= threshold:
            return complete
    return None


def leading_cut(audio, sample_rate: int) -> int:
    """Return a cut position, leaving entirely quiet audio intact."""
    offset = voice_offset(audio, sample_rate)
    return max(0, offset - round(sample_rate * 0.05)) if offset is not None else 0


class LeadingSilence:
    """Buffer one item's lead until its first complete active frame or EOF.

    ``onset_sample`` is an observation in the original PCM timeline, also useful
    for reporting first voiced audio separately from first-packet availability.
    After opening, all subsequent samples pass through unchanged.
    """

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.onset_sample: int | None = None
        self._parts: list[np.ndarray] = []
        self._released = False

    def feed(self, audio, *, final: bool = False) -> np.ndarray | None:
        if self._released:
            return audio
        self._parts.append(np.asarray(audio, dtype=np.float32).reshape(-1))
        joined = np.concatenate(self._parts)
        self.onset_sample = voice_offset(joined, self.sample_rate, complete_only=not final)
        if self.onset_sample is None and not final:
            return None
        start = (max(0, self.onset_sample - round(self.sample_rate * 0.05))
                 if self.onset_sample is not None else 0)
        self._parts.clear()
        self._released = True
        return joined[start:]

    def finish(self) -> np.ndarray | None:
        """Close the item, preserving quiet items and evaluating the final frame."""
        if self._released or not self._parts:
            return None
        return self.feed(np.empty(0, dtype=np.float32), final=True)
