import pytest

from aquatts.inference.checkpoints import detect_sovits_version


@pytest.mark.parametrize("model,symbols,lora", [
    ("v1", "v1", False), ("v2", "v2", False),
    ("v2Pro", "v2", False), ("v2ProPlus", "v2", False),
    ("v3", "v2", False), ("v3", "v2", True),
])
def test_metadata_wins_over_misleading_filename(model, symbols, lora):
    assert detect_sovits_version(
        "directory_v3/voice_v1.pth", reader=lambda _: [symbols, model, lora]
    ) == (symbols, model, lora)


def test_unsupported_architecture_is_not_treated_as_v2():
    with pytest.raises(ValueError, match="Unsupported.*v4"):
        detect_sovits_version("voice.pth", reader=lambda _: ["v2", "v4", True])


def test_unknown_header_is_not_guessed_from_path():
    def reader(_):
        raise KeyError(b"99")
    with pytest.raises(ValueError, match="Unrecognized"):
        detect_sovits_version("v2.pth", reader=reader)


def test_missing_checkpoint_is_not_hidden():
    def reader(_):
        raise FileNotFoundError("missing.pth")
    with pytest.raises(FileNotFoundError):
        detect_sovits_version("v2Pro.pth", reader=reader)
