from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

import aquatts
from aquatts.upstream import GPTSoVITSConfigurationError, configure_gpt_sovits


def _fake_upstream(tmp_path: Path) -> Path:
    marker = tmp_path / "GPT_SoVITS" / "AR" / "models" / "t2s_model.py"
    marker.parent.mkdir(parents=True)
    marker.write_text("# structural fixture\n", encoding="utf-8")
    return tmp_path


def test_configure_upstream_accepts_repo_or_package_root(tmp_path, monkeypatch):
    repo = _fake_upstream(tmp_path)
    original_path = list(sys.path)
    monkeypatch.setattr(sys, "path", list(original_path))
    monkeypatch.setenv("GPT_SOVITS_HOME", "")

    assert configure_gpt_sovits(repo, require=True) == repo.resolve()
    assert os.environ["GPT_SOVITS_HOME"] == str(repo.resolve())
    assert str(repo.resolve()) in sys.path
    assert str((repo / "GPT_SoVITS").resolve()) in sys.path

    assert configure_gpt_sovits(repo / "GPT_SoVITS", require=True) == repo.resolve()


def test_configure_upstream_fails_closed_for_wrong_layout(tmp_path):
    with pytest.raises(GPTSoVITSConfigurationError, match="t2s_model.py"):
        configure_gpt_sovits(tmp_path, require=True)


def test_upstream_decoder_patch_preserves_native_naive_generator(monkeypatch):
    home = os.environ.get("GPT_SOVITS_HOME", "").strip()
    if not home:
        pytest.skip("upstream GPT-SoVITS checkout is not configured")

    configure_gpt_sovits(home, require=True)
    from AR.models.t2s_model import Text2SemanticDecoder
    from aquatts.modeling.t2s_streaming import apply_cuda_graph_patch

    config = {
        "model": {
            "hidden_dim": 8,
            "embedding_dim": 8,
            "head": 2,
            "n_layer": 1,
            "vocab_size": 16,
            "phoneme_vocab_size": 32,
            "dropout": 0.0,
            "EOS": 15,
        }
    }
    decoder = Text2SemanticDecoder(config)
    original_naive = decoder.infer_panel_naive
    original_panel = decoder.infer_panel
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    assert apply_cuda_graph_patch(decoder) is decoder
    assert decoder.infer_panel_naive == original_naive
    assert decoder.infer_panel != original_panel
    assert decoder._aquatts_original_infer_panel == original_panel
    assert decoder.use_static_kv_cache is False
    assert apply_cuda_graph_patch(decoder) is decoder

    import torch

    tokens, steps = decoder.infer_panel(
        torch.tensor([[1, 2, 3]]),
        torch.tensor([3]),
        torch.tensor([[4, 5]]),
        torch.randn(1, 1024, 3),
        top_k=5,
        top_p=1.0,
        early_stop_num=1,
        temperature=0.6,
    )
    assert tokens.ndim == 2
    assert steps >= 1


def test_speaker_sources_resolve_outside_upstream_cwd(tmp_path, monkeypatch):
    from aquatts.upstream import configure_speaker_encoder

    repo = _fake_upstream(tmp_path / "upstream")
    directory = repo / "GPT_SoVITS" / "eres2net"
    directory.mkdir()
    for name in ("ERes2NetV2.py", "fusion.py", "pooling_layers.py", "kaldi.py"):
        (directory / name).write_text("# fixture\n")
    monkeypatch.setenv("GPT_SOVITS_HOME", str(repo))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.chdir(tmp_path)
    assert configure_speaker_encoder() == directory
    assert Path.cwd() == tmp_path
    assert str(directory) in sys.path


def test_legacy_checkout_only_requires_speaker_sources_for_pro(tmp_path, monkeypatch):
    from aquatts.upstream import configure_speaker_encoder

    repo = _fake_upstream(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("GPT_SOVITS_HOME", str(repo))
    assert configure_gpt_sovits(repo, require=True) == repo
    with pytest.raises(GPTSoVITSConfigurationError, match="Pro speaker encoder"):
        configure_speaker_encoder()


def test_selected_t2s_source_is_not_bundled_fork():
    home = os.environ.get("GPT_SOVITS_HOME", "").strip()
    if not home:
        pytest.skip("upstream GPT-SoVITS checkout is not configured")

    configure_gpt_sovits(home, require=True)
    import AR.models.t2s_model as model

    assert Path(model.__file__).resolve() == aquatts.upstream_t2s_model_path()
    assert "_vendor" not in model.__file__
