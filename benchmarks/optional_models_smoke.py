"""Measure lazy/preloaded BERT with real local V3 weights in fresh processes.

Run once normally and once with --preload-bert. AP-BWE imports are deliberately
blocked; no super-resolution is requested. Voice assets remain local.
"""
from __future__ import annotations

import argparse
import builtins
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preload-bert", action="store_true")
    parser.add_argument("--frontend-cwd", type=Path,
                        help="Existing upstream resource root for cwd-relative G2PW assets")
    args = parser.parse_args()
    args.config = args.config.resolve()
    args.output = args.output.resolve()
    if args.frontend_cwd:
        if not (args.frontend_cwd / "GPT_SoVITS/text/G2PWModel/g2pW.onnx").is_file():
            parser.error("frontend-cwd must contain existing GPT_SoVITS/text/G2PWModel/g2pW.onnx")
        os.chdir(args.frontend_cwd)
    original_import = builtins.__import__
    def block_audio_sr(name, *positional, **kwargs):
        if name == "tools.audio_sr":
            raise AssertionError("AP-BWE imported despite super-resolution being disabled")
        return original_import(name, *positional, **kwargs)
    builtins.__import__ = block_audio_sr
    from aquatts import TTSInferencer
    import numpy as np
    import psutil
    import soundfile as sf
    import torch

    config = json.loads(args.config.read_text(encoding="utf-8"))
    loads = []
    original_load = TTSInferencer._init_bert_model
    def record_load(self):
        original_load(self)
        loads.append(self.bert_load_seconds)
    TTSInferencer._init_bert_model = record_load
    start = time.perf_counter()
    inferencer = TTSInferencer(**config["inferencer"], preload_bert=args.preload_bert,
                              cuda_graph_preset="lazy", use_flash_attn=True,
                              kv_cache_layout="nested")
    device = torch.device(inferencer.device)
    torch.cuda.synchronize(device)
    startup_seconds = time.perf_counter() - start
    def snapshot():
        return {"bert_loaded": inferencer.bert_loaded, "bert_loads": len(loads),
                "cuda_allocated": torch.cuda.memory_allocated(device),
                "cuda_reserved": torch.cuda.memory_reserved(device),
                "rss": psutil.Process().memory_info().rss}
    startup = snapshot()
    assert startup["bert_loads"] == int(args.preload_bert)
    rows = []
    args.output.mkdir(parents=True, exist_ok=True)
    examples = [("english", "This is a short test of voice synthesis.", "英文"),
                ("japanese", "今日はいい天気ですね。", "日文"),
                ("chinese", "你好，今天我们一起测试中文语音。", "中文")]
    for name, text, language in examples:
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        torch.cuda.manual_seed_all(7)
        chunks, first = [], None
        started = time.perf_counter()
        for rate, chunk, _ in inferencer.infer_stream(
                **config["reference"], text=text, text_language=language,
                top_k=5, top_p=1., temperature=.6, pause_second=0,
                how_to_cut="不切", sample_steps=4, chunk_size_seconds=.25):
            if chunk is not None and chunk.size:
                first = first if first is not None else time.perf_counter() - started
                chunks.append(chunk)
        total = time.perf_counter() - started
        audio = np.concatenate(chunks)
        assert rate == 24000 and audio.dtype == np.float32
        assert np.isfinite(audio).all() and np.max(np.abs(audio)) > 1e-4
        expected = 1 if name == "chinese" else int(args.preload_bert)
        assert len(loads) == expected, (name, len(loads))
        sf.write(args.output / f"{name}.wav", audio, rate)
        rows.append({"language": name, "first_nonempty_seconds": first,
                     "total_seconds": total, "samples": audio.size,
                     "pcm_sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
                     **snapshot()})
    assert "tools.audio_sr" not in sys.modules
    report = {"preload_bert": args.preload_bert, "gpu": torch.cuda.get_device_name(device),
              "torch": torch.__version__, "startup_seconds": startup_seconds,
              "startup": startup, "bert_load_seconds": loads, "trials": rows,
              "ap_bwe_import_blocked": True,
              "explicit_frontend_cwd": args.frontend_cwd is not None,
              "source_sha256_lf": hashlib.sha256((ROOT / "aquatts/inferencer.py")
                                                 .read_text(encoding="utf-8").encode()).hexdigest(),
              "scope": "One fresh-process sample per mode; functional/resource check, not a latency distribution or language-quality evaluation."}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
