import numpy as np
import pytest

from aquatts.inference.onset import LeadingSilence, leading_cut, voice_offset


@pytest.mark.parametrize("rate", [16000, 24000, 32000, 44100, 48000])
@pytest.mark.parametrize("size", [113, 2048, 8448, 96000])
def test_chunked_and_complete_items_keep_identical_pcm(rate, size):
    audio = np.zeros(rate * 2, dtype=np.float32)
    audio[round(rate * .8):round(rate * .95)] = .1
    # An internal pause and a second voiced region must survive untouched.
    audio[round(rate * 1.2):round(rate * 1.6)] = -.2
    expected = round(rate * .75)
    assert leading_cut(audio, rate) == expected
    gate = LeadingSilence(rate)
    parts = [gate.feed(audio[i:i + size]) for i in range(0, len(audio), size)]
    parts.append(gate.finish())
    np.testing.assert_array_equal(np.concatenate([p for p in parts if p is not None]), audio[expected:])
    assert gate.onset_sample == round(rate * .8)
    assert gate.finish() is None


@pytest.mark.parametrize("speech_follows", [False, True])
def test_partial_chunk_transient_cannot_open_gate(speech_follows):
    audio = np.zeros(24000, dtype=np.float32)
    audio[8400:8448] = .006  # Active alone; below threshold over a complete 240-sample frame.
    if speech_follows:
        audio[19200:] = .1
    gate = LeadingSilence(24000)
    assert gate.feed(audio[:8448]) is None
    parts = [gate.feed(audio[8448:]), gate.finish()]
    actual = np.concatenate([p for p in parts if p is not None])
    np.testing.assert_array_equal(actual, audio[18000:] if speech_follows else audio)
    assert gate.onset_sample == (19200 if speech_follows else None)


@pytest.mark.parametrize("lead", [0, 19200])
@pytest.mark.parametrize("amplitude", [0., .001, .1])
def test_eof_handles_partial_voiced_or_quiet_frame(lead, amplitude):
    audio = np.zeros(lead + 48, dtype=np.float32)
    audio[lead:] = amplitude
    gate = LeadingSilence(24000)
    for start in range(0, len(audio), 113):
        assert gate.feed(audio[start:start + 113]) is None
    expected_cut = max(0, lead - 1200) if amplitude == .1 else 0
    np.testing.assert_array_equal(gate.finish(), audio[expected_cut:])
    assert gate.finish() is None


def test_empty_and_voiced_from_start_need_no_cut():
    assert voice_offset([], 24000) is None
    assert leading_cut([], 24000) == 0
    audio = np.full(2400, .1, dtype=np.float32)
    assert leading_cut(audio, 24000) == 0
    gate = LeadingSilence(24000)
    np.testing.assert_array_equal(gate.feed(audio), audio)
    tail = np.zeros(500, dtype=np.float32)
    assert gate.feed(tail) is tail


def test_preroll_keeps_a_weak_onset_rise():
    audio = np.zeros(24000, dtype=np.float32)
    audio[18000:19200] = np.linspace(.0001, .01, 1200)
    audio[19200:] = .1
    cut = leading_cut(audio, 24000)
    assert 0 < cut <= 18000
    assert voice_offset(audio, 24000) - cut == 1200
