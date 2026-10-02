"""Runtime contract tests; real weight smoke tests live in benchmarks/model_smoke.py."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

torch = pytest.importorskip("torch")
module = pytest.importorskip("aquatts.inferencer")


@pytest.fixture
def inferencer():
    value = module.TTSInferencer.__new__(module.TTSInferencer)
    value.device = "cpu"
    value.is_half = False
    value.is_v2pro = True
    value.model_version = "v2Pro"
    value.sv_model = SimpleNamespace(encode=Mock(return_value=torch.ones(1, 20480)))
    value.get_spepc = Mock(return_value=torch.ones(1, 1025, 8))
    value.vq_model = SimpleNamespace(
        decode=Mock(return_value=torch.ones(1, 1, 3200) * 0.25),
        extract_latent=Mock(return_value=torch.zeros(1, 1, 10, dtype=torch.long)),
    )
    return value


def test_primary_reference_uses_cached_speaker_condition(inferencer):
    spectrum, speaker = torch.zeros(1, 1025, 8), torch.zeros(1, 20480)
    session = {"refer_spec": spectrum, "sv_embedding": speaker}
    audio = inferencer._decode_v2(torch.zeros(1, 1, 10), [1, 2], "main.wav", session, None, 1.2)
    inferencer.sv_model.encode.assert_not_called()
    inferencer.get_spepc.assert_not_called()
    args, kwargs = inferencer.vq_model.decode.call_args
    assert args[2][0] is spectrum
    assert kwargs["sv_emb"][0] is speaker
    assert kwargs["speed"] == 1.2
    assert len(audio) == 3200


def test_failed_extra_speaker_does_not_desynchronize_references(inferencer):
    inferencer.get_spepc.side_effect = [torch.full((1, 2, 3), 1.), torch.full((1, 2, 3), 2.)]
    inferencer.sv_model.encode.side_effect = [ValueError("bad reference"), torch.tensor([[2.]])]
    inferencer._decode_v2(torch.zeros(1, 1, 10), [1], "main.wav", {}, ["bad.wav", "good.wav"], 1.)
    args, kwargs = inferencer.vq_model.decode.call_args
    assert len(args[2]) == len(kwargs["sv_emb"]) == 1
    assert args[2][0].flatten()[0] == kwargs["sv_emb"][0].item() == 2.


def test_all_failed_extra_references_use_primary_condition(inferencer):
    inferencer.sv_model.encode.side_effect = [ValueError("bad"), torch.tensor([[3.]])]
    inferencer._decode_v2(torch.zeros(1, 1, 10), [1], "main.wav", {}, ["bad.wav"], 1.)
    assert [call.args[0] for call in inferencer.sv_model.encode.call_args_list] == ["bad.wav", "main.wav"]
    assert inferencer.vq_model.decode.call_args.kwargs["sv_emb"][0].item() == 3.


@pytest.mark.parametrize("model", ["v1", "v2"])
def test_legacy_decoder_does_not_receive_speaker_keyword(inferencer, model):
    inferencer.is_v2pro = False
    inferencer.model_version = model
    inferencer._decode_v2(torch.zeros(1, 1, 10), [1], "main.wav", {}, None, 1.)
    assert "sv_emb" not in inferencer.vq_model.decode.call_args.kwargs
    inferencer.sv_model.encode.assert_not_called()


@pytest.mark.parametrize("model", ["v2", "v2Pro", "v2ProPlus"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_public_inference_paths_forward_condition_and_actual_rate(inferencer, model, stream, fail):
    inferencer.model_version = model
    inferencer.is_v2pro = model != "v2"
    inferencer.dict_language = {"英文": "en"}
    inferencer.splits = {"."}
    inferencer.max_sec = 10
    inferencer.hps = SimpleNamespace(data=SimpleNamespace(sampling_rate=32000))
    session = {"prompt": torch.ones(1, 10), "phones1": [1], "bert1": torch.zeros(1024, 1),
               "refer_spec": torch.ones(1, 1025, 8), "sv_embedding": torch.ones(1, 20480)}
    inferencer._build_session_cache = Mock(return_value=session)
    inferencer.get_phones_and_bert = Mock(return_value=([1, 2], torch.zeros(1024, 2), "Hello."))
    inferencer._infer_semantic_with_guard = Mock(return_value=(torch.ones(1, 1, 10), 10, 1))
    kwargs = dict(text="Hello.", ref_audio_path="main.wav", prompt_text="Reference.",
                  text_language="英文", prompt_language="英文", pause_second=0)
    if fail:
        inferencer.vq_model.decode.side_effect = ValueError("invalid speaker condition")
        with pytest.raises(ValueError, match="invalid speaker condition"):
            if stream:
                list(inferencer.infer_stream(**kwargs))
            else:
                inferencer.infer(**kwargs)
        return
    if stream:
        chunks = list(inferencer.infer_stream(**kwargs, chunk_size_seconds=0.025))
        assert chunks[0] == (32000, None, "")
        assert all(sr == 32000 for sr, _, _ in chunks)
        audio = np.concatenate([chunk for _, chunk, _ in chunks if chunk is not None])
    else:
        rate, audio = inferencer.infer(**kwargs)
        assert rate == 32000
    assert audio.shape == (3200,)
    assert np.isfinite(audio).all() and np.any(audio)
    assert ("sv_emb" in inferencer.vq_model.decode.call_args.kwargs) == inferencer.is_v2pro


@pytest.mark.parametrize("stream", [False, True])
def test_generated_lead_is_trimmed_per_item_without_removing_requested_pauses(inferencer, stream):
    rate = 32000
    raw = np.concatenate([np.zeros(6400), np.full(3200, .2), np.zeros(1280),
                          np.full(3200, .1), np.zeros(2560)]).astype(np.float32)
    inferencer.vq_model.decode.return_value = torch.from_numpy(raw).reshape(1, 1, -1)
    inferencer.dict_language = {"英文": "en"}
    inferencer.splits = {"."}
    inferencer.max_sec = 10
    inferencer.hps = SimpleNamespace(data=SimpleNamespace(sampling_rate=rate))
    inferencer._build_session_cache = Mock(return_value={
        "prompt": torch.ones(1, 10), "phones1": [1], "bert1": torch.zeros(1024, 1),
        "refer_spec": torch.ones(1, 1025, 8), "sv_embedding": torch.ones(1, 20480),
    })
    inferencer.get_phones_and_bert = Mock(return_value=([1, 2], torch.zeros(1024, 2), "Text."))
    inferencer._infer_semantic_with_guard = Mock(return_value=(torch.ones(1, 1, 10), 10, 1))
    kwargs = dict(text="First.\nSecond.", ref_audio_path="ref.wav", prompt_text="Reference.",
                  text_language="英文", prompt_language="英文", how_to_cut="不切", pause_second=.07)
    if stream:
        chunks = list(inferencer.infer_stream(**kwargs, chunk_size_seconds=.037))
        audio = np.concatenate([piece for _, piece, _ in chunks if piece is not None])
        assert [text for _, _, text in chunks if text] == ["First.", "Second."]
        expected_item = inferencer._apply_fade_out(raw, rate)[4800:]
    else:
        returned_rate, audio = inferencer.infer(**kwargs)
        assert returned_rate == rate
        expected_item = raw[4800:]
    expected_item = np.concatenate([expected_item, np.zeros(2240, dtype=np.float32)])
    np.testing.assert_array_equal(audio, np.tile(expected_item, 2))


def test_session_caches_each_reference_speaker_once(inferencer, monkeypatch):
    inferencer._session_cache = {}
    inferencer._session_cache_max = 8
    inferencer.ssl_model = SimpleNamespace(model=Mock(return_value={"last_hidden_state": torch.zeros(1, 10, 768)}))
    inferencer.get_phones_and_bert = Mock(return_value=([1], torch.zeros(1024, 1), "Reference."))
    monkeypatch.setattr(module.librosa, "load", lambda *a, **kw: (np.zeros(16000, dtype=np.float32), 16000))
    first = inferencer._build_session_cache("one.wav", "Reference.", "en")
    assert "sv_embedding" in first
    assert inferencer._build_session_cache("one.wav", "Reference.", "en") is first
    second = inferencer._build_session_cache("two.wav", "Reference.", "en")
    assert second is not first
    assert [call.args[0] for call in inferencer.sv_model.encode.call_args_list] == ["one.wav", "two.wav"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("cut", ["不切", "按标点符号切"])
@pytest.mark.parametrize("ending", ["、", ",", "，", "。", "？", "！", ""])
def test_public_paths_preserve_existing_prompt_and_segment_punctuation(inferencer, stream, cut, ending):
    inferencer.i18n = lambda key: key
    inferencer.sovits_version = "v2"
    inferencer._detect_model_version = lambda: "v2Pro"
    inferencer._init_language_dict()
    inferencer.max_sec = 10
    inferencer.hps = SimpleNamespace(data=SimpleNamespace(sampling_rate=32000))
    inferencer._build_session_cache = Mock(return_value={
        "prompt": torch.ones(1, 10), "phones1": [1], "bert1": torch.zeros(1024, 1),
        "refer_spec": torch.ones(1, 1025, 8), "sv_embedding": torch.ones(1, 20480),
    })
    inferencer.get_phones_and_bert = Mock(return_value=([1, 2], torch.zeros(1024, 2), "音声。"))
    inferencer._infer_semantic_with_guard = Mock(return_value=(torch.ones(1, 1, 10), 10, 1))
    first = "説明を始めます" + ending
    prompt = "参照音声" + ending
    expected = [first if ending else first + "。"]
    text = first
    if cut == "按标点符号切" and ending:
        text += "次の項目を確認します。"
        expected.append("次の項目を確認します。")
    kwargs = dict(text=text, ref_audio_path="main.wav", prompt_text=prompt,
                  text_language="日文", prompt_language="日文", how_to_cut=cut, pause_second=0)
    if stream:
        list(inferencer.infer_stream(**kwargs))
    else:
        inferencer.infer(**kwargs)
    assert [call.args[0] for call in inferencer.get_phones_and_bert.call_args_list] == expected
    assert inferencer._build_session_cache.call_args.args[1] == (prompt if ending else prompt + "。")


@pytest.mark.parametrize("header,model", [(b"00", "v1"), (b"01", "v2"), (b"02", "v3"),
                                         (b"03", "v3"), (b"05", "v2Pro"), (b"06", "v2ProPlus")])
def test_real_upstream_detector_accepts_renamed_header_checkpoint(tmp_path, header, model):
    path = tmp_path / "renamed.pth"
    path.write_bytes(header + b"fixture")
    from aquatts.inference.checkpoints import detect_sovits_version
    assert detect_sovits_version(path)[1] == model


@pytest.mark.parametrize("version", ["v2Pro", "v2ProPlus"])
@pytest.mark.parametrize("missing_speaker", [False, True])
def test_loader_preserves_architecture_and_requires_conditioning_weights(monkeypatch, version, missing_speaker):
    class Decoder(torch.nn.Module):
        def __init__(self, *args, version, **kwargs):
            super().__init__()
            self.version = version
            self.sv_emb = torch.nn.Linear(2, 2)
            self.ge_to512 = torch.nn.Linear(2, 2)
            self.prelu = torch.nn.PReLU()
            self.enc_q = torch.nn.Linear(2, 2)

        def decode(self, codes, phones, references, sv_emb=None):
            pass

    weights = Decoder(version=version).state_dict()
    if missing_speaker:
        del weights["sv_emb.weight"]
    checkpoint = {"weight": weights, "config": {
        "data": {"filter_length": 2048, "hop_length": 640, "n_speakers": 1},
        "train": {"segment_size": 20480}, "model": {"version": "v2"},
    }}
    monkeypatch.setattr(module, "load_sovits_new", lambda _: checkpoint)
    monkeypatch.setattr(module, "SynthesizerTrn", Decoder)
    from aquatts.inference import speaker
    encoder = Mock()
    monkeypatch.setattr(speaker, "SpeakerEncoder", encoder)
    value = module.TTSInferencer.__new__(module.TTSInferencer)
    value.sovits_path = "renamed.pth"
    value.model_version = version
    value.is_v2pro = True
    value.is_half = False
    value.device = "cpu"
    value.sv_model_path = "speaker.ckpt"
    if missing_speaker:
        with pytest.raises(RuntimeError, match="sv_emb.weight"):
            value._load_sovits_model()
        encoder.assert_not_called()
    else:
        value._load_sovits_model()
        assert value.vq_model.version == version
        assert value.sovits_version == "v2"
        encoder.assert_called_once_with("speaker.ckpt", "cpu", False)


@pytest.mark.parametrize("lora", [False, True])
def test_v3_loader_separates_acoustic_version_from_text_symbols(monkeypatch, tmp_path, lora):
    weights = {} if lora else {"enc_p.text_embedding.weight": SimpleNamespace(shape=(732, 1))}
    checkpoint = {"weight": weights, "config": {
        "data": {"filter_length": 2048, "hop_length": 640, "n_speakers": 1},
        "train": {"segment_size": 20480}, "model": {"version": "v2"},
    }}
    decoder = Mock()
    decoder.to.return_value = decoder
    decoder.cfm.merge_and_unload.return_value = decoder.cfm
    constructor = Mock(return_value=decoder)
    monkeypatch.setattr(module, "load_sovits_new", lambda _: checkpoint)
    monkeypatch.setattr(module, "SynthesizerTrnV3", constructor)
    monkeypatch.setattr(module, "get_peft_model", lambda model, config: model)
    pretrain = tmp_path / "pretrained.pth"
    pretrain.touch()
    value = module.TTSInferencer.__new__(module.TTSInferencer)
    value.sovits_path = "renamed.pth"
    value.sovits_pretrain_path = str(pretrain)
    value.model_version = "v3"
    value.is_v2pro = False
    value._detected_lora = lora
    value.is_half = False
    value.device = "cpu"

    value._load_sovits_model()

    assert constructor.call_args.kwargs["version"] == "v3"
    assert value.hps.model.version == "v3"
    assert value.sovits_version == "v2"
    assert value.if_lora_v3 is lora


def test_missing_speaker_weight_is_reported_before_importing_encoder(tmp_path):
    from aquatts.inference.speaker import SpeakerEncoder
    with pytest.raises(FileNotFoundError, match="speaker encoder weight"):
        SpeakerEncoder(tmp_path / "missing.ckpt", "cpu", False)


@pytest.mark.parametrize("version", ["v2", "v2Pro", "v2ProPlus", "v3"])
def test_public_language_labels_are_independent_of_host_ui_locale(monkeypatch, version):
    value = module.TTSInferencer.__new__(module.TTSInferencer)
    value.sovits_path = "renamed.pth"
    value.i18n = lambda key: "translated:" + key
    monkeypatch.setattr(module, "detect_sovits_version", lambda _: ("v2", version, False))
    value._init_language_dict()
    assert value.dict_language["英文"] == value.dict_language["translated:英文"] == "en"
    assert value.dict_language["日文"] == "all_ja"
    assert value.dict_language["中文"] == "all_zh"
    assert value.dict_language["韩文"] == "all_ko"
