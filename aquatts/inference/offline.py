"""Local resource checks; no process-wide Hub flags or network monkeypatches."""
from __future__ import annotations

import json
import os
from pathlib import Path


class OfflineResourceError(FileNotFoundError):
    """A requested component cannot be used with the local resource contract."""


def require_files(component, directory, names):
    directory = Path(directory).expanduser().resolve()
    missing = [name for name in names if not (directory / name).is_file()]
    if missing:
        raise OfflineResourceError(
            f"offline_only: {component}: expected directory {directory}; missing {', '.join(missing)}")
    return directory


def require_checkpoint(component, path):
    path = Path(path).expanduser().resolve()
    require_files(component, path.parent, [path.name])


def audio_sr_resource_dir():
    """Locate resources by the selected upstream tools.audio_sr module."""
    from importlib.util import find_spec
    spec = find_spec("tools.audio_sr")
    if spec is None or spec.origin is None:
        raise ImportError("AP-BWE requires the upstream tools.audio_sr module")
    return Path(spec.origin).resolve().parent / "AP_BWE_main" / "24kto48k"


def require_transformer(component, directory, *, tokenizer=False, weights=True, processor=False):
    names = ["config.json"]
    if processor:
        names += ["preprocessor_config.json"]
    root = require_files(component, directory, names)
    if tokenizer and not any((root / name).is_file() for name in ("vocab.txt", "tokenizer.json")):
        raise OfflineResourceError(
            f"offline_only: {component}: expected directory {root}; missing vocab.txt or tokenizer.json")
    for name in ("config.json", "tokenizer_config.json"):
        if (root / name).is_file():
            config = json.loads((root / name).read_text(encoding="utf-8"))
            if config.get("auto_map"):
                raise OfflineResourceError(f"offline_only: {component}: custom auto_map unsupported in {root / name}")
    if weights:
        if not any((root / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
            indices = [root / name for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json")
                       if (root / name).is_file()]
            if not indices:
                raise OfflineResourceError(
                    f"offline_only: {component}: expected directory {root}; missing model.safetensors or pytorch_model.bin (or shard index)")
            shards = json.loads(indices[0].read_text(encoding="utf-8")).get("weight_map", {})
            if not shards:
                raise OfflineResourceError(f"offline_only: {component}: empty shard index {indices[0]}")
            require_files(component, root, sorted(set(shards.values())))
    return root


def check_frontend(language, version):
    """Check before upstream lazy imports can download G2PW/NLTK/JTalk resources.

    G2PW still uses its upstream process-level paths. Configure those before
    importing that frontend; this function never changes cwd or environment.
    """
    if language == "en":
        import nltk
        from importlib.metadata import version as package_version
        resources = ["taggers/averaged_perceptron_tagger.zip", "corpora/cmudict.zip"]
        if tuple(int(x) for x in package_version("nltk").split(".")[:2]) >= (3, 9):
            resources += ["taggers/averaged_perceptron_tagger_eng/"]
        for resource in resources:
            try:
                nltk.data.find(resource)
            except LookupError as error:
                raise OfflineResourceError(
                    f"offline_only: English NLTK: missing {resource}; searched {nltk.data.path}") from error
    elif language == "ja":
        import pyopenjtalk
        directory = os.fsdecode(pyopenjtalk.OPEN_JTALK_DICT_DIR)
        require_files("Japanese OpenJTalk", directory, ["sys.dic", "matrix.bin", "char.bin", "unk.dic"])
    elif language == "zh":
        if version == "v1":
            return
        root = Path("GPT_SoVITS/text/G2PWModel").resolve()
        require_files("Chinese G2PW", root, ["config.py", "POLYPHONIC_CHARS.txt", "MONOPHONIC_CHARS.txt",
                                             "bopomofo_to_pinyin_wo_tune_dict.json", "char_bopomofo_dict.json"])
        if not any((root / name).is_file() for name in ("g2pW.onnx", "g2pw.onnx")):
            require_files("Chinese G2PW", root, ["g2pW.onnx"])
        source = os.environ.get("bert_path", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large")
        require_transformer("Chinese G2PW tokenizer", source, tokenizer=True, weights=False)
    else:
        raise OfflineResourceError(
            f"offline_only: frontend {language!r} is not verified; supported languages are en/ja/zh")
