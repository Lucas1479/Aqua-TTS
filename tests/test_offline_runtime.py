from concurrent.futures import ThreadPoolExecutor
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")
module = pytest.importorskip("aquatts.inferencer")
from aquatts.inference.language_detector import bind_segmenter  # noqa: E402
from aquatts.inference.offline import OfflineResourceError, check_frontend  # noqa: E402


def model_files(path):
    for name in ("config.json", "vocab.txt", "pytorch_model.bin", "preprocessor_config.json"):
        (path / name).write_text("{}")


def test_private_segmenters_keep_global_detector_and_each_other(monkeypatch):
    import fast_langdetect
    before = fast_langdetect.infer._default_detector
    detect_zh = Mock(return_value=[{"lang": "zh", "score": 1.}])
    detect_en = Mock(return_value=[{"lang": "en", "score": 1.}])
    one = bind_segmenter(module.LangSegmenter.getTexts, SimpleNamespace(detect=detect_zh))
    two = bind_segmenter(module.LangSegmenter.getTexts, SimpleNamespace(detect=detect_en))
    monkeypatch.setattr(fast_langdetect, "detect", Mock(side_effect=AssertionError("global detector used")))
    expected_one = one("你好世界")
    assert detect_zh.call_count and not detect_en.call_count
    zh_calls = detect_zh.call_count
    expected_two = two("你好世界")
    assert detect_en.call_count and detect_zh.call_count == zh_calls
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda fn: fn("你好世界"), [one, two] * 4))
    # Upstream can override implausible detector labels using script rules.
    assert results == [expected_one, expected_two] * 4
    assert detect_zh.call_count and detect_en.call_count
    assert fast_langdetect.infer._default_detector is before


def test_local_detector_uses_supported_custom_path_and_never_changes_global(tmp_path):
    import fast_langdetect
    before = fast_langdetect.infer._default_detector
    value = module.TTSInferencer.__new__(module.TTSInferencer)
    value.offline_only = True
    value.fast_langdetect_path = str(tmp_path)
    with pytest.raises(OfflineResourceError, match="lid.176.bin"):
        value._configure_fast_langdetect()
    (tmp_path / "lid.176.bin").touch()
    value._configure_fast_langdetect()
    assert value._language_detector.config.custom_model_path == str(tmp_path / "lid.176.bin")
    assert fast_langdetect.infer._default_detector is before


@pytest.mark.parametrize("offline", [False, True])
def test_bert_calls_are_local_only_per_instance(tmp_path, monkeypatch, offline):
    import transformers
    model_files(tmp_path)
    tokenizer = Mock(return_value=object())
    model = Mock(return_value=torch.nn.Linear(1, 1))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer)
    monkeypatch.setattr(transformers.AutoModelForMaskedLM, "from_pretrained", model)
    value = module.TTSInferencer.__new__(module.TTSInferencer)
    value.offline_only, value.bert_path, value.is_half, value.device = offline, str(tmp_path), False, "cpu"
    before = dict(os.environ)
    value._init_bert_model()
    for loader in (tokenizer, model):
        assert loader.call_args.kwargs == ({"local_files_only": True} if offline else {})
    assert dict(os.environ) == before


def test_cnhubert_uses_instance_path_without_mutating_default(tmp_path, monkeypatch):
    model_files(tmp_path)
    factory = Mock(return_value=torch.nn.Linear(1, 1))
    monkeypatch.setattr(module.cnhubert, "CNHubert", factory)
    previous = module.cnhubert.cnhubert_base_path
    value = module.TTSInferencer.__new__(module.TTSInferencer)
    value.offline_only, value.cnhubert_path, value.is_half, value.device = True, str(tmp_path), False, "cpu"
    value._init_ssl_model()
    factory.assert_called_once_with(str(tmp_path))
    assert module.cnhubert.cnhubert_base_path == previous


def test_missing_chinese_resources_fail_before_import_or_download(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(OfflineResourceError, match="Chinese G2PW"):
        check_frontend("zh", "v2")


def test_missing_nltk_resources_do_not_start_download(monkeypatch):
    import nltk
    monkeypatch.setattr(nltk.data, "path", [])
    download = Mock(side_effect=AssertionError("network path reached"))
    monkeypatch.setattr(nltk, "download", download)
    with pytest.raises(OfflineResourceError, match="English NLTK"):
        check_frontend("en", "v2")
    download.assert_not_called()


def test_missing_japanese_dictionary_does_not_download(tmp_path, monkeypatch):
    import pyopenjtalk
    monkeypatch.setattr(pyopenjtalk, "OPEN_JTALK_DICT_DIR", str(tmp_path).encode())
    # pyopenjtalk-plus bundles its dictionary and has no _extract_dic.
    # Guard network access in either distribution, and the legacy downloader
    # when present, without requiring that private implementation detail.
    download = Mock(side_effect=AssertionError("download"))
    monkeypatch.setattr("socket.socket.connect", download)
    if hasattr(pyopenjtalk, "_extract_dic"):
        monkeypatch.setattr(pyopenjtalk, "_extract_dic", download)
    with pytest.raises(OfflineResourceError, match="Japanese OpenJTalk"):
        check_frontend("ja", "v2")
    download.assert_not_called()


@pytest.mark.parametrize("language", ["ko", "yue", "unknown"])
def test_unverified_frontends_fail_explicitly(language):
    with pytest.raises(OfflineResourceError, match="not verified"):
        check_frontend(language, "v2")
