from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("aquatts.inferencer")
from aquatts.voice_registry import Voice  # noqa: E402
from tests import test_reference_runtime as reference_helpers  # noqa: E402

reference_inferencer = reference_helpers.reference_inferencer

SEMANTIC = ("prompt", "phones1", "bert1")
ACOUSTIC = ("refer_spec", "prompt_fea_ref", "prompt_ge", "mel2_norm")


@pytest.fixture
def mixed(reference_inferencer):
    value, path = reference_inferencer
    other = path.with_name("semantic.wav")
    other.write_bytes(b"other fixture")
    first = value._normalize_reference_prompt(path, "Acoustic", "en")
    second = value._normalize_semantic_reference(
        Voice("other", str(other), "意味参照", "日文"),
        prompt_text="Acoustic", prompt_language="英文", ref_free=False)
    def features(reference):
        marker = 1 if reference.audio_path == str(path) else 2
        return {key: torch.full((1, 2, 3), float(marker)) for key in (*SEMANTIC, *ACOUSTIC)
                if key != "phones1"} | {"phones1": [marker, marker]}
    value._build_reference_features = Mock(side_effect=features)
    return value, first, second


def test_source_fields_are_correct_and_request_mutation_isolated(mixed):
    value, first, second = mixed
    session = value._reference_session_for_request(first, second)
    for key in SEMANTIC:
        assert session[key] == [2, 2] if key == "phones1" else torch.all(session[key] == 2)
    for key in ACOUSTIC:
        assert torch.all(session[key] == 1)
    for key, item in session.items():
        if key == "phones1":
            item[0] = 99
        else:
            item.fill_(99)
    again = value._reference_session_for_request(first, second)
    assert again["phones1"] == [2, 2] and torch.all(again["prompt_ge"] == 1)
    ordinary = value._reference_session_for_request(first)
    assert ordinary["phones1"] == [1, 1] and torch.all(ordinary["prompt"] == 1)
    reverse = value._reference_session_for_request(second, first)
    assert reverse["phones1"] == [1, 1] and torch.all(reverse["prompt_ge"] == 2)
    assert value._build_reference_features.call_count == 2


def test_languages_normalize_independently_and_aliases_hit(mixed):
    value, first, second = mixed
    assert first.language == "en" and second.language == "all_ja"
    value._reference_session_for_request(first, second)
    alias = value._normalize_semantic_reference(
        {"ref_audio_path": second.audio_path, "prompt_text": "意味参照。", "prompt_language": "all_ja"},
        prompt_text="Acoustic", prompt_language="en", ref_free=False)
    value._reference_session_for_request(first, alias)
    assert value._build_reference_features.call_count == 2


def test_same_reference_and_small_capacity_are_supported(mixed, monkeypatch):
    value, first, second = mixed
    same = value._reference_session_for_request(first, first)
    assert same["phones1"] == [1, 1] and value._build_reference_features.call_count == 1
    monkeypatch.setenv("AQUA_SESSION_CACHE_MAX", "1")
    value._init_reference_cache()
    session = value._reference_session_for_request(first, second)
    assert session["phones1"] == [2, 2] and torch.all(session["mel2_norm"] == 1)
    assert len(value._reference_cache) == 1
    with pytest.raises(ValueError, match="capacity"):
        value._warm_reference_list([
            Voice("a", first.audio_path, first.text, first.language),
            Voice("b", second.audio_path, second.text, second.language)], strict=True)


def test_missing_and_corrupted_semantic_features_raise_without_fallback(mixed):
    value, first, second = mixed
    value._reference_session_for_request(first, second)
    entry = value._get_reference_entry(second)[0]
    entry.pop("bert1")
    with pytest.raises(RuntimeError, match="bert1"):
        value._reference_session_for_request(first, second)
    value.clear_reference_cache()
    value._build_reference_features.side_effect = RuntimeError("cannot encode semantic input")
    with pytest.raises(RuntimeError, match="cannot encode"):
        value._reference_session_for_request(first, second)


@pytest.mark.parametrize("reference,error", [
    ({}, "ref_audio_path"), ({"ref_audio_path": "audio.wav"}, "prompt_text"),
    ({"ref_audio_path": "audio.wav", "prompt_text": "  "}, "prompt_text"),
    ({"ref_audio_path": "audio.wav", "prompt_text": "Text", "prompt_language": "unknown"}, "language"),
])
def test_incomplete_reference_fails_explicitly(reference_inferencer, reference, error):
    value, _ = reference_inferencer
    with pytest.raises(ValueError, match=error):
        value._normalize_semantic_reference(reference, prompt_text="Acoustic", prompt_language="en", ref_free=False)


@pytest.mark.parametrize("api", ["infer", "infer_stream"])
@pytest.mark.parametrize("version", ["v2", "v2Pro", "v2ProPlus"])
def test_both_public_apis_reject_non_v3_before_loading(reference_inferencer, api, version):
    value, path = reference_inferencer
    value.model_version = version
    with pytest.raises(ValueError, match="only for V3"):
        result = getattr(value, api)("Text", str(path), "Reference", semantic_reference=Voice("s", str(path), "Text"))
        if api == "infer_stream":
            next(result)
    value._build_reference_features.assert_not_called()


@pytest.mark.parametrize("prompt,language,ref_free,message", [
    ("", "en", False, "acoustic prompt_text"), ("Text", "en", True, "ref_free"),
    ("Text", "bad", False, "acoustic reference language"),
])
def test_mixed_references_do_not_use_silent_fallback(reference_inferencer, prompt, language, ref_free, message):
    value, path = reference_inferencer
    with pytest.raises(ValueError, match=message):
        value._normalize_semantic_reference(Voice("s", str(path), "Text"),
                                             prompt_text=prompt, prompt_language=language, ref_free=ref_free)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cross_stream_clones_record_lifetime_when_cache_evicts(mixed, monkeypatch):
    value, first, second = mixed
    value.device = "cuda:0"
    parameter = torch.ones(1, device=value.device)
    value.vq_model.parameters = lambda: iter([parameter])
    builder = value._build_reference_features.side_effect
    value._build_reference_features.side_effect = lambda ref: {
        key: item.to(value.device) if torch.is_tensor(item) else item for key, item in builder(ref).items()}
    monkeypatch.setenv("AQUA_SESSION_CACHE_MAX", "1")
    value._init_reference_cache()
    source_stream, request_stream = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(source_stream):
        value._get_reference_entry(first)
    records = []
    record_stream = torch.Tensor.record_stream
    def record(tensor, stream):
        records.append(stream)
        record_stream(tensor, stream)
    monkeypatch.setattr(torch.Tensor, "record_stream", record)
    with torch.cuda.stream(request_stream):
        session = value._reference_session_for_request(first, second)
    request_stream.synchronize()
    assert len(records) == 6 and all(stream == request_stream for stream in records)
    assert torch.all(session["prompt"] == 2) and torch.all(session["prompt_ge"] == 1)
