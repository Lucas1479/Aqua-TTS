import importlib.util
from pathlib import Path
import sys
import threading
from types import ModuleType
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")
module = pytest.importorskip("aquatts.inferencer")


def test_ap_bwe_checks_and_loads_tool_relative_checkpoint(tmp_path, monkeypatch):
    source = tmp_path / "upstream" / "tools" / "audio_sr.py"
    source.parent.mkdir(parents=True)
    source.write_text("# upstream fixture")
    resources = source.parent / "AP_BWE_main" / "24kto48k"
    resources.mkdir(parents=True)
    for name in ("config.json", "g_24kto48k.zip"):
        (resources / name).touch()
    fake = ModuleType("tools.audio_sr")
    fake.__file__ = str(source)
    fake.__spec__ = importlib.util.spec_from_file_location("tools.audio_sr", source)
    fake.AP_BWE = Mock(return_value=object())
    monkeypatch.setitem(sys.modules, "tools.audio_sr", fake)
    unrelated = tmp_path / "caller"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    value = module.TTSInferencer.__new__(module.TTSInferencer)
    value.device, value.offline_only, value.sr_model = "cpu", True, None
    value._sr_load_lock = threading.Lock()
    assert value._get_audio_sr_model() is fake.AP_BWE.return_value
    assert Path(fake.AP_BWE.call_args.kwargs["checkpoint_file"]) == resources / "g_24kto48k.zip"
    assert value._get_audio_sr_model() is value.sr_model and fake.AP_BWE.call_count == 1


def test_online_mode_does_not_require_private_splitter_adapter(tmp_path, monkeypatch):
    import fast_langdetect
    from aquatts.inference import language_detector
    original = fast_langdetect.infer._default_detector
    monkeypatch.setattr(fast_langdetect.infer, "_default_detector", original)
    bind = Mock(side_effect=RuntimeError("split-lang internals changed"))
    monkeypatch.setattr(language_detector, "bind_segmenter", bind)
    value = module.TTSInferencer.__new__(module.TTSInferencer)
    value.offline_only, value.fast_langdetect_path = False, str(tmp_path)
    value._configure_fast_langdetect()
    bind.assert_not_called()
    assert value._segmenter is None
    assert fast_langdetect.infer._default_detector is value._language_detector


def test_offline_adapter_failure_does_not_change_global_detector(tmp_path, monkeypatch):
    import fast_langdetect
    from aquatts.inference import language_detector
    original = fast_langdetect.infer._default_detector
    (tmp_path / "lid.176.bin").touch()
    monkeypatch.setattr(language_detector, "bind_segmenter", Mock(side_effect=RuntimeError("unsupported splitter")))
    value = module.TTSInferencer.__new__(module.TTSInferencer)
    value.offline_only, value.fast_langdetect_path = True, str(tmp_path)
    with pytest.raises(RuntimeError, match="unsupported splitter"):
        value._configure_fast_langdetect()
    assert fast_langdetect.infer._default_detector is original
