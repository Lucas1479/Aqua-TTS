"""Public reference warmup and inference share one validated cache entry."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")
module = pytest.importorskip("aquatts.inferencer")
from tests import test_lazy_models as lazy_helpers  # noqa: E402
from aquatts.voice_registry import Voice  # noqa: E402

construct = lazy_helpers.construct


def features():
    return {key: torch.ones(1, 2, 3) for key in (
        "refer_spec", "prompt", "bert1", "sv_embedding", "mel2_norm", "prompt_fea_ref", "prompt_ge")
    } | {"phones1": [1, 2, 3]}


@pytest.fixture
def reference_inferencer(tmp_path):
    value = module.TTSInferencer.__new__(module.TTSInferencer)
    value.device, value.is_half = "cpu", False
    value.model_version, value.is_v2pro = "v3", False
    value.dict_language = {"英文": "en", "日文": "all_ja", "中文": "all_zh"}
    value.splits = {".", "。", ",", "，"}
    value.vq_model = SimpleNamespace(parameters=lambda: iter([torch.zeros(1)]))
    value._init_reference_cache()
    value._build_reference_features = Mock(side_effect=lambda _: features())
    path = tmp_path / "reference.wav"
    path.write_bytes(b"reference fixture")
    return value, path


@pytest.mark.parametrize("label,code,text,punct", [
    ("英文", "en", "Reference", "."), ("日文", "all_ja", "参照", "。"),
    ("中文", "all_zh", "参考", "。"),
])
def test_warmup_and_request_share_normalized_entry(reference_inferencer, monkeypatch, label, code, text, punct):
    value, path = reference_inferencer
    monkeypatch.chdir(path.parent)
    first = value.warm_reference_cache(path.name, text, label)
    assert first["status"] == "built" and first["ready"]
    entry = value._build_session_cache(str(path), text + punct, code)
    assert "prompt_ge" in entry
    assert value.warm_reference_cache(path, text + punct, code)["status"] == "hit"
    assert value._build_reference_features.call_count == 1


def test_failed_or_incomplete_warmup_never_reports_ready(reference_inferencer):
    value, path = reference_inferencer
    value._build_reference_features.side_effect = [features() | {"prompt_ge": None}, {}, features()]
    with pytest.raises(RuntimeError, match="prompt_ge"):
        value.warm_reference_cache(path, "Reference", "en")
    assert len(value._reference_cache) == 0
    result = value.warm_reference_cache(path, "Reference", "en", strict=False)
    assert result["status"] == "failed" and not result["ready"]
    assert len(value._reference_cache) == 0
    assert value.warm_reference_cache(path, "Reference", "en")["ready"]


@pytest.mark.parametrize("version", ["v2", "v3"])
def test_empty_prompt_warmup_matches_the_real_fallback(reference_inferencer, version):
    value, path = reference_inferencer
    value.model_version = version
    warmed = value.warm_reference_cache(path, "  ", "英文")
    assert warmed["ready"]
    assert value._build_session_cache(str(path), None, "all_ja")
    assert value._build_reference_features.call_count == 1
    normalized = value._build_reference_features.call_args.args[0]
    assert normalized.ref_free is (version == "v2")
    assert bool(normalized.text) is (version == "v3")


def test_replacement_dtype_and_explicit_clear_rebuild(reference_inferencer):
    value, path = reference_inferencer
    first = value.warm_reference_cache(path, "Reference", "en")
    path.write_bytes(b"changed audio fixture")
    second = value.warm_reference_cache(path, "Reference", "en")
    assert first["key"] != second["key"]
    value.vq_model.parameters = lambda: iter([torch.zeros(1, dtype=torch.float64)])
    assert value.warm_reference_cache(path, "Reference", "en")["status"] == "built"
    generation = value._reference_generation
    assert value.clear_reference_cache() == 3
    assert value._reference_generation == generation + 1
    assert value.warm_reference_cache(path, "Reference", "en")["status"] == "built"
    assert value._build_reference_features.call_count == 4


def test_file_changed_while_building_is_not_published(reference_inferencer):
    value, path = reference_inferencer
    def mutate(_):
        path.write_bytes(b"file replaced while encoding")
        return features()
    value._build_reference_features.side_effect = mutate
    with pytest.raises(RuntimeError, match="changed during"):
        value.warm_reference_cache(path, "Reference", "en")
    assert len(value._reference_cache) == 0


def test_constructor_only_warms_requested_references_and_checks_capacity(construct, monkeypatch, tmp_path):
    builder = Mock(side_effect=lambda _: features())
    monkeypatch.setattr(module.TTSInferencer, "_build_reference_features", builder)
    path = tmp_path / "reference.wav"
    path.write_bytes(b"fixture")
    assert construct().reference_warmup_results == []
    builder.assert_not_called()
    voice = Voice("voice", str(path), "Reference", "英文")
    value = construct(warmup_references=[voice, voice.to_dict()])
    assert [item["status"] for item in value.reference_warmup_results] == ["built", "hit"]
    assert builder.call_count == 1
    monkeypatch.setenv("AQUA_SESSION_CACHE_MAX", "1")
    builder.reset_mock()
    with pytest.raises(ValueError, match="capacity"):
        construct(warmup_references=[voice, Voice("other", str(path), "Different", "英文")])
    builder.assert_not_called()
    builder.side_effect = lambda _: {}
    value = construct(warmup_references=[voice], warmup_strict=False)
    assert value.reference_warmup_results[0]["status"] == "failed"


def test_ref_free_v2_does_not_encode_unused_semantic_reference(reference_inferencer, monkeypatch):
    value, path = reference_inferencer
    value.model_version = "v2"
    value.get_spepc = Mock(return_value=torch.ones(1, 2, 3))
    value.ssl_model = SimpleNamespace(model=Mock(side_effect=AssertionError("unused SSL")))
    value.get_phones_and_bert = Mock(side_effect=AssertionError("unused phones/BERT"))
    del value._build_reference_features
    assert value.warm_reference_cache(path, "", "en")["ready"]
    value.get_spepc.assert_called_once()
    value.get_phones_and_bert.assert_not_called()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_reference_hit_orders_cuda_streams_without_device_sync(reference_inferencer, monkeypatch):
    value, path = reference_inferencer
    value.device = "cuda:0"
    parameter = torch.ones(1, device=value.device)
    value.vq_model.parameters = lambda: iter([parameter])
    first, second = torch.cuda.Stream(device=0), torch.cuda.Stream(device=0)
    tensor = torch.zeros(1, device=value.device)
    def build(_):
        torch.cuda._sleep(2_000_000)
        tensor.fill_(17)
        return features() | {"refer_spec": tensor}
    value._build_reference_features.side_effect = build
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: pytest.fail("normal hit synchronized device"))
    with torch.cuda.stream(first):
        value._build_session_cache(str(path), "Reference", "en")
    with torch.cuda.stream(second):
        entry = value._build_session_cache(str(path), "Reference", "en")
        observed = entry["refer_spec"].clone()
    second.synchronize()
    assert observed.item() == 17 and value._build_reference_features.call_count == 1
