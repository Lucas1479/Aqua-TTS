"""Optional component lifecycle contracts with small model substitutes."""
from concurrent.futures import ThreadPoolExecutor
import builtins
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

torch = pytest.importorskip("torch")
module = pytest.importorskip("aquatts.inferencer")


class TinyBert(torch.nn.Module):
    def forward(self, input_ids, **_kwargs):
        value = torch.ones(1, input_ids.shape[1], 1024)
        return {"hidden_states": [value] * 4}


@pytest.fixture
def loaders(monkeypatch):
    import transformers
    tokenizer = Mock(side_effect=lambda text, **_: {
        "input_ids": torch.zeros(1, len(text) + 2, dtype=torch.long)})
    tokenizer_factory = Mock(return_value=tokenizer)
    model_factory = Mock(side_effect=lambda *_args, **_kwargs: TinyBert())
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer_factory)
    monkeypatch.setattr(transformers.AutoModelForMaskedLM, "from_pretrained", model_factory)
    return tokenizer_factory, model_factory


@pytest.fixture
def construct(monkeypatch, tmp_path):
    cls = module.TTSInferencer
    monkeypatch.setattr(cls, "_configure_fast_langdetect", lambda self: None)
    monkeypatch.setattr(cls, "_init_ssl_model", lambda self: None)
    monkeypatch.setattr(cls, "_load_models", lambda self: None)
    monkeypatch.setattr(module, "I18nAuto", lambda **_: lambda value: value)
    def detect(self):
        self.sovits_version = "v2"
        self.is_v2pro = False
        return "v2"
    monkeypatch.setattr(cls, "_detect_model_version", detect)
    def create(**kwargs):
        value = cls(device="cpu", bert_path=str(tmp_path), cuda_graph_preset="off", **kwargs)
        value.clean_text_inf = lambda text, _language: (
            list(range(len(text))), [1] * len(text), text)
        return value
    return create


def test_constructor_is_lazy_and_explicit_preload_is_idempotent(construct, loaders):
    tokenizer, model = loaders
    value = construct()
    assert not value.bert_loaded and value.bert_load_seconds is None
    tokenizer.assert_not_called()
    model.assert_not_called()
    duration = value.preload_bert_model()
    assert value.bert_loaded and duration >= 0
    assert not value.bert_model.training
    assert value.preload_bert_model() == duration
    assert tokenizer.call_count == model.call_count == 1
    second = construct(preload_bert=True)
    assert second.bert_loaded and tokenizer.call_count == model.call_count == 2


@pytest.mark.parametrize("language", ["en", "all_ja", "all_ko", "all_yue"])
def test_non_chinese_frontends_do_not_load_bert(construct, loaders, language):
    value = construct()
    phones, features, _ = value.get_phones_and_bert("abcdef。", language)
    assert features.shape == (1024, len(phones)) and not features.any()
    assert not value.bert_loaded
    for loader in loaders:
        loader.assert_not_called()


@pytest.mark.parametrize("language", ["all_zh", "zh", "auto"])
def test_chinese_and_mixed_segments_load_once(construct, loaders, monkeypatch, language):
    value = construct()
    monkeypatch.setattr(module.LangSegmenter, "getTexts", lambda _: [
        {"text": "English", "lang": "en"}, {"text": "中文测试文本。", "lang": "zh"}])
    for _ in range(2):
        phones, features, _ = value.get_phones_and_bert("中文测试文本。", language)
        assert features.shape == (1024, len(phones)) and features.any()
    assert value.bert_loaded
    assert all(loader.call_count == 1 for loader in loaders)


def test_chinese_reference_loads_bert_before_english_target(construct, loaders, monkeypatch, tmp_path):
    value = construct()
    monkeypatch.setattr(module.librosa, "load", lambda *_args, **_kwargs: (np.ones(8000), 16000))
    value.ssl_model = SimpleNamespace(model=lambda _: {"last_hidden_state": torch.ones(1, 8, 4)})
    value.vq_model = SimpleNamespace(extract_latent=lambda _: torch.ones(1, 1, 4, dtype=torch.long))
    value.get_spepc = lambda _: torch.ones(1, 4, 8)
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"fixture")
    entry = value._build_session_cache(str(reference), "中文参考文本。", "all_zh")
    assert entry["bert1"].any() and value.bert_loaded
    value.get_phones_and_bert("English text.", "en")
    assert all(loader.call_count == 1 for loader in loaders)


def test_concurrent_first_chinese_use_publishes_one_complete_model(construct, loaders):
    value = construct()
    entered, release = threading.Event(), threading.Event()
    def load(*_args, **_kwargs):
        entered.set()
        assert release.wait(5)
        return TinyBert()
    loaders[1].side_effect = load
    with ThreadPoolExecutor(max_workers=4) as pool:
        calls = [pool.submit(value.get_bert_feature, "中文", [1, 1]) for _ in range(4)]
        assert entered.wait(5)
        assert value.bert_model is None and value.tokenizer is None
        release.set()
        results = [call.result(timeout=5) for call in calls]
    assert all(loader.call_count == 1 for loader in loaders)
    assert all(item.shape == (1024, 2) and item.eq(1).all() for item in results)


@pytest.mark.parametrize("stage", ["tokenizer", "weights", "device"])
def test_failed_first_load_leaves_no_partial_state_and_can_retry(construct, loaders, stage):
    value = construct()
    error = RuntimeError("injected load failure")
    if stage == "tokenizer":
        loaders[0].side_effect = [error, loaders[0].return_value]
    elif stage == "weights":
        loaders[1].side_effect = [error, TinyBert()]
    else:
        broken = TinyBert()
        broken.to = Mock(side_effect=error)
        loaders[1].side_effect = [broken, TinyBert()]
    with pytest.raises(RuntimeError, match="injected"):
        value.preload_bert_model()
    assert value.tokenizer is None and value.bert_model is None
    assert value.bert_load_seconds is None
    value.preload_bert_model()
    assert value.bert_loaded


def test_super_resolution_import_is_optional_and_failure_preserves_audio(construct, monkeypatch):
    original = builtins.__import__
    imports = []
    def blocked(name, *args, **kwargs):
        if name == "tools.audio_sr":
            imports.append(name)
            raise ImportError("optional dependency unavailable")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", blocked)
    value = construct()
    audio = torch.arange(16, dtype=torch.float32) / 32
    value.model_version = "v3"
    np.testing.assert_array_equal(value._finalize_stream_chunk(audio, 24000), audio.numpy())
    assert imports == []
    result, rate = value._audio_sr(audio, 24000)
    np.testing.assert_array_equal(result, audio.numpy())
    assert rate == 24000 and value.sr_model is None and len(imports) == 1
    np.testing.assert_array_equal(
        value._finalize_stream_chunk(audio, 24000, if_sr=True), audio.numpy())


def test_super_resolution_load_retries_then_reuses_complete_instance(construct, monkeypatch):
    value = construct()
    model = Mock(side_effect=lambda audio, sr: (audio.cpu().numpy(), sr))
    loader = Mock(side_effect=[FileNotFoundError("missing weights"), model])
    monkeypatch.setitem(sys.modules, "tools.audio_sr", SimpleNamespace(AP_BWE=loader))
    audio = torch.zeros(16)
    value._audio_sr(audio, 24000)
    assert value.sr_model is None
    value._audio_sr(audio, 24000)
    value._audio_sr(audio, 24000)
    assert value.sr_model is model and loader.call_count == 2 and model.call_count == 2
