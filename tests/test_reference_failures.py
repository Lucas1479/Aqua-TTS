from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

torch = pytest.importorskip("torch")
module = pytest.importorskip("aquatts.inferencer")
from tests import test_stream_concurrency as stream_helpers  # noqa: E402

tiny_v3 = stream_helpers.tiny_v3


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("version", ["v2", "v2Pro", "v3"])
def test_live_reference_failure_is_preserved_before_semantic_generation(tiny_v3, tmp_path, stream, version):
    value = tiny_v3
    value.model_version = version
    value.is_v2pro = version == "v2Pro"
    value.vq_model.parameters = lambda: iter([torch.zeros(1)])
    del value._build_session_cache
    error = ImportError("reference decoder is unavailable")
    value._build_reference_features = Mock(side_effect=error)
    path = tmp_path / "ref.wav"
    path.write_bytes(b"reference fixture")
    with pytest.raises(ImportError) as failure:
        result = (value.infer_stream if stream else value.infer)("Text", str(path), "Reference", how_to_cut="不切")
        if stream:
            next(result)
    assert failure.value is error
    value._infer_semantic_with_guard.assert_not_called()
    assert len(value._reference_cache) == 0


def test_prompt_only_zero_step_result_is_not_generated_audio():
    value = module.TTSInferencer.__new__(module.TTSInferencer)
    value.hz = 50
    value._semantic_guard_enabled = lambda: False
    value.t2s_model = SimpleNamespace(model=SimpleNamespace(infer_panel=lambda *_args, **_kwargs: (torch.tensor([[7, 8]]), 0)))
    candidate, count, _ = value._infer_semantic_with_guard(
        text_item="Text", target_phone_count=4, all_phoneme_ids=torch.ones(1, 4), all_phoneme_len=torch.tensor([4]),
        prompt=torch.tensor([[7, 8]]), bert=torch.zeros(1, 1024, 4), top_k=5, top_p=1., temperature=.6,
        effective_max_sec=3., enable_cuda_graph=False, enable_static_kv=True)
    assert count == 0 and candidate.shape == (1, 1, 0)


def test_soundfile_reference_decode_downmixes_and_resamples_without_torchcodec(tiny_v3, tmp_path, monkeypatch):
    import soundfile as sf
    import torchaudio
    monkeypatch.setattr(torchaudio, "load", Mock(side_effect=ImportError("TorchCodec unavailable")))
    path = tmp_path / "stereo.wav"
    samples = np.stack([np.full(16000, .2), np.full(16000, .4)], axis=1).astype(np.float32)
    sf.write(path, samples, 32000, subtype="FLOAT")
    audio = tiny_v3._load_reference_audio_24k(str(path))
    assert audio.shape == (1, 12000) and audio.dtype == torch.float32
    assert torch.isfinite(audio).all()
    torch.testing.assert_close(audio[:, 100:-100], torch.full((1, 11800), .3), rtol=.002, atol=.002)
    torchaudio.load.assert_not_called()



@pytest.mark.parametrize("endpoint", ["/tts", "/tts/file"])
@pytest.mark.parametrize("version", ["v2", "v2Pro", "v3"])
@pytest.mark.parametrize("cache_hit", [False, True])
def test_http_reference_failure_precedes_success_headers(tiny_v3, tmp_path, endpoint, version, cache_hit):
    import asyncio
    import httpx
    from aquatts.inference.cuda_execution import cuda_execution
    from aquatts.server import _create_app
    from aquatts.voice_registry import VoiceRegistry

    value = tiny_v3
    value.model_version, value.is_v2pro = version, version == "v2Pro"
    entry = dict(value._build_session_cache())
    entry["sv_embedding"] = torch.ones(1, 4)
    del value._build_session_cache
    path = tmp_path / "ref.wav"
    path.write_bytes(b"reference fixture")
    if cache_hit:
        value._build_reference_features = Mock(return_value=entry)
        value.warm_reference_cache(str(path), "Reference", "英文")
        entry["prompt"] = None
        expected = "Incomplete reference features: prompt"
    else:
        expected = "reference decoder is unavailable"
        value._build_reference_features = Mock(side_effect=ImportError(expected))
    app = _create_app(value, VoiceRegistry())

    async def run():
        # Model a real client: an exception after response.start cannot change 200.
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(endpoint, params={
                "text": "Text", "ref_audio_path": str(path), "prompt_text": "Reference",
                "text_language": "英文", "prompt_language": "英文",
            })
        assert response.status_code == 500, (response.status_code, response.content)
        assert "x-sample-rate" not in response.headers
        assert response.json()["detail"] == expected

    asyncio.run(run())
    value._infer_semantic_with_guard.assert_not_called()
    assert not cuda_execution._owners
    assert value._sovits_decode_lock.acquire(blocking=False)
    value._sovits_decode_lock.release()
