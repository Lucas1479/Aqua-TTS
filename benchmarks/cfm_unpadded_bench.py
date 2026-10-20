"""Same-noise real-checkpoint CFM experiment; local voice assets are not published."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Fixed before candidate measurement. These are engineering bounds, not listening
# or perceptual equivalence claims. Baseline repeat noise is reported separately.
LIMITS = {
    "float16": {"mel_max": .02, "mel_rms": .003, "pcm_max": .1, "pcm_rms": .01},
    "float32": {"mel_max": 2e-4, "mel_rms": 3e-5, "pcm_max": .005, "pcm_rms": .0005},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import soundfile as sf
    import torch
    from aquatts import TTSInferencer
    from aquatts.inference.cfm import UnpaddedCFM
    from aquatts.inferencer import _denormalize_v3_spec

    config = json.loads(args.config.read_text(encoding="utf-8"))
    tts = TTSInferencer(**config["inferencer"], cuda_graph_preset="lazy", use_flash_attn=True)
    cfm = tts.vq_model.cfm
    original_forward = cfm.estimator.forward
    captured = []
    run_cfm = tts._run_cfm
    def capture(fea, lengths, prompt, steps):
        if not captured:
            captured.append((fea.clone(), prompt.clone()))
        return run_cfm(fea, lengths, prompt, steps)
    tts._run_cfm = capture
    common = {**config["reference"], "text": "This is a short test of voice synthesis.",
              "text_language": "英文", "top_k": 5, "top_p": 1., "temperature": .6,
              "pause_second": 0, "how_to_cut": "不切", "sample_steps": 4}
    def seed():
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
    seed()
    list(tts.infer_stream(**common, chunk_size_seconds=.25))
    tts._run_cfm = run_cfm
    adapter = UnpaddedCFM(cfm)
    tts._cfm_adapter = adapter
    adapted_forward = cfm.estimator.forward
    e2e = []
    for chunk in (None, .1, .25, .5):
        outputs = {}
        for repeat in range(3):
            for fast in ((False, True) if repeat % 2 == 0 else (True, False)):
                tts.cfm_unpadded = fast
                cfm.estimator.forward = adapted_forward if fast else original_forward
                seed()
                started = time.perf_counter()
                first = audible = None
                pieces = []
                if chunk is None:
                    rate, audio = tts.infer(**common)
                    first = time.perf_counter() - started
                    audible = first if np.max(np.abs(audio)) > .01 else None
                else:
                    for rate, piece, _ in tts.infer_stream(**common, chunk_size_seconds=chunk):
                        if piece is not None and piece.size:
                            first = first if first is not None else time.perf_counter() - started
                            if audible is None and np.max(np.abs(piece)) > .01:
                                audible = time.perf_counter() - started
                            pieces.append(piece)
                    audio = np.concatenate(pieces)
                total = time.perf_counter() - started
                assert rate == 24000 and np.isfinite(audio).all() and np.max(np.abs(audio)) > 1e-4
                outputs[fast] = audio
                sf.write(args.output / f"{'fast' if fast else 'baseline'}-{chunk}.wav", audio, rate)
                e2e.append({"chunk_seconds": chunk, "repeat": repeat, "fast": fast,
                            "samples": int(audio.size), "first_nonempty_seconds": first,
                            "first_audible_seconds": audible, "total_seconds": total,
                            "rtf": total / (audio.size / rate)})
            assert outputs[False].shape == outputs[True].shape
            diff = outputs[True].astype(np.float64) - outputs[False]
            e2e[-1]["paired_pcm_max"] = float(np.max(np.abs(diff)))
            e2e[-1]["paired_pcm_rms"] = float(np.sqrt(np.mean(diff ** 2)))

    mu_real, prompt_real = captured[0]
    ref_length = prompt_real.size(-1)
    cases = [("short_tail", torch.cat((mu_real[:, :16], mu_real[:, ref_length:ref_length + 1]), dim=1), prompt_real[..., :16]),
             ("real_chunk", mu_real, prompt_real)]
    # Stretch the captured reference to the caller's maximum and use real target
    # feature frames cyclically. This is a numerical stress case, not a voice sample.
    long_ref = mu_real[:, :ref_length].repeat(1, (468 + ref_length - 1) // ref_length, 1)[:, :468]
    long_prompt = prompt_real.repeat(1, 1, (468 + ref_length - 1) // ref_length)[..., :468]
    target = mu_real[:, ref_length:]
    target = target.repeat(1, (64 + target.size(1) - 1) // target.size(1), 1)[:, :64]
    cases.append(("long_reference", torch.cat((long_ref, target), dim=1), long_prompt))
    rows = []
    report = {"gpu": torch.cuda.get_device_name(tts.device), "torch": torch.__version__,
              "thresholds_fixed_before_measurement": LIMITS, "e2e": e2e, "cases": rows,
              "scope": "One warmup and three timed samples per CFM case; three paired warmed E2E requests per mode. Provisional single-GPU experiment, no P95 or perceptual-quality claim. FP32 computes with the same loaded weights converted from FP16."}

    def save():
        report["source_sha256_lf"] = {name: hashlib.sha256((ROOT / name).read_text(encoding="utf-8").encode()).hexdigest()
                                      for name in ("aquatts/inference/cfm.py", "aquatts/inferencer.py")}
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def errors(left, right):
        diff = (left.float() - right.float()).double()
        return {"max": diff.abs().max().item(), "rms": diff.square().mean().sqrt().item()}

    with torch.inference_mode():
        for dtype in (torch.float16, torch.float32):
            precision = str(dtype).split(".")[-1]
            cfm.to(dtype=dtype)
            for name, mu_input, ref_input in cases:
                mu, prompt = mu_input.to(dtype), ref_input.to(dtype)
                lengths = torch.tensor([mu.size(1)], device=mu.device)
                for steps in (4, 16, 32):
                    for cfg in (0., 1.):
                        def run(fast):
                            cfm.estimator.forward = adapted_forward if fast else original_forward
                            seed()
                            torch.cuda.synchronize(tts.device)
                            started = time.perf_counter()
                            output = (adapter(mu, lengths, prompt, steps, unpadded_length=mu.size(1), inference_cfg_rate=cfg)
                                      if fast else cfm.inference(mu, lengths, prompt, steps, inference_cfg_rate=cfg))
                            torch.cuda.synchronize(tts.device)
                            return output, time.perf_counter() - started
                        baseline, _ = run(False)
                        repeated, _ = run(False)
                        candidate, _ = run(True)
                        repeat_error = errors(baseline, repeated)
                        mel_error = errors(baseline, candidate)
                        finite = bool(torch.isfinite(candidate).all())
                        def vocode(mel):
                            # Very short tails need the vocoder's reflect-pad context.
                            mel = mel[..., prompt.size(-1):]
                            if mel.size(-1) < 8:
                                mel = mel.repeat(1, 1, (8 + mel.size(-1) - 1) // mel.size(-1))
                            return tts._run_bigvgan(_denormalize_v3_spec(mel).half())
                        pcm_error = errors(vocode(baseline), vocode(candidate))
                        times = {False: [], True: []}
                        for repeat in range(3):
                            for fast in ((False, True) if repeat % 2 == 0 else (True, False)):
                                _, seconds = run(fast)
                                times[fast].append(seconds)
                        limit = LIMITS[precision]
                        passed = (finite and mel_error["max"] <= limit["mel_max"] and mel_error["rms"] <= limit["mel_rms"]
                                  and pcm_error["max"] <= limit["pcm_max"] and pcm_error["rms"] <= limit["pcm_rms"])
                        rows.append({"precision": precision, "shape": name, "frames": mu.size(1), "reference_frames": prompt.size(-1),
                                     "steps": steps, "cfg": cfg, "baseline_repeat_error": repeat_error,
                                     "mel_error": mel_error, "pcm_error": pcm_error, "finite": finite, "passed": passed,
                                     "baseline_seconds": times[False], "fast_seconds": times[True],
                                     "median_ratio_fast_over_baseline": statistics.median(times[True]) / statistics.median(times[False])})
                        save()
                        print(json.dumps(rows[-1]), flush=True)
    report["all_numeric_checks_passed"] = all(row["passed"] for row in rows)
    report["e2e_pcm_checks_passed"] = all(row.get("paired_pcm_max", 0) <= LIMITS["float16"]["pcm_max"]
                                           and row.get("paired_pcm_rms", 0) <= LIMITS["float16"]["pcm_rms"] for row in e2e)
    save()
    if not report["all_numeric_checks_passed"] or not report["e2e_pcm_checks_passed"]:
        raise SystemExit("Numeric gate failed; retained complete measurements")


if __name__ == "__main__":
    main()
