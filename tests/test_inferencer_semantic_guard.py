import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch", reason="Torch runtime is optional in unit CI")


inferencer_module = pytest.importorskip(
    "aquatts.inferencer",
    reason="full GPT-SoVITS runtime is optional in unit CI",
)
from aquatts.inference.semantic_stability import SemanticGenerationError  # noqa: E402


class _FakeSemanticDecoder:
    def __init__(self, candidates, reported_counts=None):
        self.candidates = candidates
        self.reported_counts = reported_counts
        self.penalties = []

    def infer_panel(self, *_args, **kwargs):
        self.penalties.append(float(kwargs["repetition_penalty"]))
        index = len(self.penalties) - 1
        tokens = self.candidates[index]
        count = (
            self.reported_counts[index]
            if self.reported_counts is not None
            else len(tokens)
        )
        return torch.tensor([tokens], dtype=torch.long), count


def _inferencer(candidates, reported_counts=None):
    decoder = _FakeSemanticDecoder(candidates, reported_counts)
    inferencer = inferencer_module.TTSInferencer.__new__(
        inferencer_module.TTSInferencer
    )
    inferencer.hz = 50
    inferencer.t2s_model = SimpleNamespace(model=decoder)
    return inferencer, decoder


def _call(inferencer):
    return inferencer._infer_semantic_with_guard(
        text_item="うーん……正直、まだ完全には分からないわ。",
        target_phone_count=48,
        all_phoneme_ids=torch.zeros(1, 4, dtype=torch.long),
        all_phoneme_len=torch.tensor([4]),
        prompt=torch.zeros(1, 2, dtype=torch.long),
        bert=torch.zeros(1, 1024, 4),
        top_k=5,
        top_p=1.0,
        temperature=0.6,
        effective_max_sec=8.0,
        enable_cuda_graph=True,
        enable_static_kv=True,
    )


@contextmanager
def _guard_setting(value):
    previous = os.environ.get("AQUATTS_SEMANTIC_GUARD")
    os.environ["AQUATTS_SEMANTIC_GUARD"] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("AQUATTS_SEMANTIC_GUARD", None)
        else:
            os.environ["AQUATTS_SEMANTIC_GUARD"] = previous


def test_guard_retries_once_and_returns_only_recovered_candidate():
    with _guard_setting("1"):
        bad = list(range(20)) + [937] * 60
        good = [index % 23 for index in range(70)]
        inferencer, decoder = _inferencer([bad, good])

        candidate, count, attempts = _call(inferencer)

    assert attempts == 2
    assert count == len(good)
    assert candidate.reshape(-1).tolist() == good
    assert decoder.penalties == [1.35, 1.5]


def test_guard_fails_closed_after_two_collapsed_candidates():
    with _guard_setting("1"):
        inferencer, decoder = _inferencer([[41] * 60, [77] * 70])

        with pytest.raises(SemanticGenerationError, match="after 2 attempts"):
            _call(inferencer)

    assert decoder.penalties == [1.35, 1.5]


def test_ref_free_zero_index_keeps_the_complete_candidate():
    with _guard_setting("1"):
        good = [index % 23 for index in range(70)]
        inferencer, decoder = _inferencer([good], reported_counts=[0])

        candidate, count, attempts = _call(inferencer)

    assert attempts == 1
    assert count == len(good)
    assert candidate.reshape(-1).tolist() == good
    assert decoder.penalties == [1.35]


def test_live_decoder_explicitly_receives_aqua_patch(monkeypatch):
    class FakeDecoder:
        pass

    decoder = FakeDecoder()
    calls = []
    monkeypatch.setattr(inferencer_module, "upstream_t2s_model_path", lambda: None)
    import aquatts.modeling.t2s_streaming as streaming

    monkeypatch.setattr(
        streaming,
        "apply_cuda_graph_patch",
        lambda value: calls.append(value) or value,
    )
    inferencer = inferencer_module.TTSInferencer.__new__(
        inferencer_module.TTSInferencer
    )
    inferencer.t2s_model = SimpleNamespace(model=decoder)

    inferencer._apply_aqua_t2s_patch()

    assert calls == [decoder]
