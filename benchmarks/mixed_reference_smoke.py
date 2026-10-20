"""Validate V3 semantic/acoustic composition and repeated reference switching."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--semantic-audio", type=Path, required=True)
    parser.add_argument("--semantic-text", required=True)
    parser.add_argument("--semantic-language", default="英文")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import soundfile as sf
    import torch
    from aquatts import TTSInferencer, Voice

    config = json.loads(args.config.read_text(encoding="utf-8"))
    acoustic = Voice("acoustic", **config["reference"])
    semantic = Voice("semantic", str(args.semantic_audio.resolve()), args.semantic_text, args.semantic_language)
    tts = TTSInferencer(**config["inferencer"], cuda_graph_preset="lazy", use_flash_attn=True)
    counts = []
    build = tts._build_reference_features
    def counted(reference):
        counts.append(reference.audio_path)
        return build(reference)
    tts._build_reference_features = counted
    for voice in (acoustic, semantic):
        assert tts.warm_reference_cache(voice.ref_audio_path, voice.prompt_text, voice.prompt_language)["ready"]
    assert len(counts) == 2
    normalized = [tts._normalize_reference_prompt(v.ref_audio_path, v.prompt_text, v.prompt_language)
                  for v in (acoustic, semantic)]
    entries = [tts._get_reference_entry(ref)[0] for ref in normalized]
    snapshots = [{key: item.clone() if torch.is_tensor(item) else list(item)
                  for key, item in entry.items()} for entry in entries]
    sessions = []
    original_session = tts._reference_session_for_request
    def capture(acoustic_ref, semantic_ref=None):
        result = original_session(acoustic_ref, semantic_ref)
        sessions.append((acoustic_ref, semantic_ref, result))
        return result
    tts._reference_session_for_request = capture
    def equal(left, right):
        return torch.equal(left, right) if torch.is_tensor(left) else left == right

    common = {"text": "This is a short test of voice synthesis.", "text_language": "英文",
              "top_k": 5, "top_p": 1., "temperature": .6, "pause_second": 0,
              "how_to_cut": "不切", "sample_steps": 4}
    # Control for first-request Graph capture before comparing reference modes.
    # Retain the cold output separately; this is not a cold latency benchmark.
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    _, cold_audio = tts.infer(**common, **config["reference"])
    rows, outputs = [], {}
    for mode, ordinary, explicit in (("baseline", acoustic, None), ("same", acoustic, acoustic),
                                      ("mixed", acoustic, semantic), ("reverse", semantic, acoustic),
                                      ("restored", acoustic, None)):
        for streaming in (False, True):
            random.seed(7)
            np.random.seed(7)
            torch.manual_seed(7)
            kwargs = {key: value for key, value in ordinary.to_dict().items() if key != "name"}
            start = time.perf_counter()
            first = None
            if streaming:
                chunks = []
                for rate, chunk, _ in tts.infer_stream(**common, **kwargs, semantic_reference=explicit,
                                                       chunk_size_seconds=.25):
                    if chunk is not None and chunk.size:
                        first = first if first is not None else time.perf_counter() - start
                        chunks.append(chunk)
                audio = np.concatenate(chunks)
            else:
                rate, audio = tts.infer(**common, **kwargs, semantic_reference=explicit)
                first = time.perf_counter() - start
            total = time.perf_counter() - start
            assert rate == 24000 and np.isfinite(audio).all() and np.max(np.abs(audio)) > 1e-4
            outputs[mode, streaming] = audio
            if mode in {"same", "restored"}:
                assert np.array_equal(audio, outputs["baseline", streaming])
            actual_acoustic, actual_semantic, session = sessions[-1]
            a = normalized.index(actual_acoustic)
            s = a if actual_semantic is None else normalized.index(actual_semantic)
            for key in ("prompt", "phones1", "bert1"):
                assert equal(session[key], snapshots[s][key]), key
            for key in ("refer_spec", "prompt_fea_ref", "prompt_ge", "mel2_norm"):
                assert equal(session[key], snapshots[a][key]), key
            assert len(counts) == 2
            sf.write(args.output / f"{mode}-{'stream' if streaming else 'ordinary'}.wav", audio, rate)
            rows.append({"mode": mode, "streaming": streaming, "samples": int(audio.size),
                         "first_nonempty_seconds": first, "total_seconds": total,
                         "rtf": total / (audio.size / rate), "field_sources_verified": True,
                         "pcm_sha256": hashlib.sha256(audio.tobytes()).hexdigest()})
    for entry, snapshot in zip(entries, snapshots):
        assert all(equal(entry[key], snapshot[key]) for key in snapshot)
    report = {"gpu": torch.cuda.get_device_name(tts.device), "torch": torch.__version__,
              "reference_builds": len(counts), "resident_references": len(tts._reference_cache),
              "same_reference_and_restored_pcm_identical": True, "shared_cache_unchanged": True,
              "cold_same_path_control": {"samples": int(cold_audio.size),
                                         "warmed_samples": int(outputs["baseline", False].size),
                                         "pcm_identical": bool(np.array_equal(cold_audio, outputs["baseline", False]))},
              "trials": rows, "scope": "One warmed functional sample per mode/API after an explicit ordinary Graph-capture control; no speaker/emotion quality or performance claim. Second reference source is supplied by the caller.",
              "source_sha256_lf": hashlib.sha256((ROOT / "aquatts/inferencer.py").read_text(encoding="utf-8").encode()).hexdigest()}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
