import json

import pytest

from aquatts.inference.offline import OfflineResourceError, require_files, require_transformer


def assets(root, **extra):
    for name, content in {"config.json": "{}", "vocab.txt": "vocab", "pytorch_model.bin": "weights", **extra}.items():
        (root / name).write_text(content, encoding="utf-8")


@pytest.mark.parametrize("missing", ["config.json", "vocab.txt", "pytorch_model.bin"])
def test_missing_transformer_component_names_directory_and_file(tmp_path, missing):
    assets(tmp_path)
    (tmp_path / missing).unlink()
    with pytest.raises(OfflineResourceError) as error:
        require_transformer("BERT", tmp_path, tokenizer=True)
    assert "BERT" in str(error.value) and str(tmp_path) in str(error.value) and missing in str(error.value)


def test_local_shards_are_checked(tmp_path):
    assets(tmp_path)
    (tmp_path / "pytorch_model.bin").unlink()
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "part.safetensors"}}))
    with pytest.raises(OfflineResourceError, match="part.safetensors"):
        require_transformer("BERT", tmp_path)
    (tmp_path / "part.safetensors").touch()
    assert require_transformer("BERT", tmp_path) == tmp_path


def test_tokenizer_only_does_not_require_weights(tmp_path):
    assets(tmp_path)
    (tmp_path / "pytorch_model.bin").unlink()
    require_transformer("G2PW tokenizer", tmp_path, tokenizer=True, weights=False)


def test_fast_tokenizer_json_does_not_require_legacy_vocab(tmp_path):
    assets(tmp_path)
    (tmp_path / "vocab.txt").rename(tmp_path / "tokenizer.json")
    require_transformer("BERT", tmp_path, tokenizer=True)


def test_custom_model_code_is_outside_verified_boundary(tmp_path):
    assets(tmp_path, **{"config.json": '{"auto_map": {"AutoConfig": "remote.Config"}}'})
    with pytest.raises(OfflineResourceError, match="auto_map"):
        require_transformer("BERT", tmp_path)


def test_directory_is_not_a_file(tmp_path):
    (tmp_path / "lid.176.bin").mkdir()
    with pytest.raises(OfflineResourceError, match="fast-langdetect"):
        require_files("fast-langdetect", tmp_path, ["lid.176.bin"])
