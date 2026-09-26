# Aqua benchmark suite

Reproducible comparisons of Aqua and the selected upstream GPT-SoVITS
checkout. Benchmark commands require `GPT_SOVITS_HOME` or an explicit
`--upstream-home`; Aqua no longer benchmarks a vendored T2S model fork.

## T2S AR throughput

Run the Aqua CUDA Graph + SDPA fallback baseline:

```bash
python benchmarks/t2s_speed_bench.py \
  --upstream-home /path/to/GPT-SoVITS \
  --gpt-model /path/to/xxx-e15.ckpt
```

Run the preferred FlashAttention2 path explicitly (the product runtime selects
this automatically when FA2 is importable; the benchmark keeps variants explicit):

```bash
python benchmarks/t2s_speed_bench.py \
  --upstream-home /path/to/GPT-SoVITS \
  --gpt-model /path/to/xxx-e15.ckpt \
  --flash-attn valid
```

Compare current upstream, Aqua static KV, Aqua CUDA Graph, and optionally FA2
in isolated subprocesses:

```bash
python benchmarks/t2s_comparison_bench.py \
  --upstream-home /path/to/GPT-SoVITS \
  --gpt-model /path/to/xxx-e15.ckpt \
  --flash-ab
```

The authoritative number is `sync`, not tqdm's smoothed display rate. The
benchmark disables progress rendering, synchronizes CUDA before and after each
call, and also reports caller-visible `wall` throughput as a diagnostic.

### Fixed cases

Run length never depends on a sampled EOS token. Each shape has a deterministic
early-stop boundary and maps to a known static-KV bucket.

| Case | Initial KV length | Bucket | Reported AR steps | Purpose |
|---|---:|---:|---:|---|
| cold/conversation | 377 | 512 | 123 | First host-loop use after graph capture |
| short | 308 | 448 | 51 | Short conversational response |
| conversation | 377 | 512 | 123 | Reproduces the Amadeus 500+ it/s trace shape |
| long | 448 | 768 | 180 | Exposes long-bucket attention cost |

### Measurement protocol

1. Load the same fp16 GPT checkpoint for every engine.
2. Pre-capture only buckets 448, 512 and 768 for the Aqua Graph engines.
3. Measure one cold conversation-shape call.
4. Run 15 conversation-shape warmups. Five warmups were not sufficient to
   stabilize the RTX 4070 Ti SUPER's Windows P-state in repeated testing.
5. Measure seven repeats per steady-state shape and report the median.
6. Keep `CUDA_GRAPH_REPLAY_SYNC` unset. A device-wide replay sync is available
   only for graph diagnostics and is not part of production behavior.
7. Run comparison engines in isolated subprocesses to prevent `sys.modules`
   contamination between upstream and Aqua overlays.

JSON evidence can be saved with `--json-output result.json`.

## Current T2S result

RTX 4070 Ti SUPER, Windows 11, Python 3.12, PyTorch 2.5.1+cu124,
upstream GPT-SoVITS `08d627c`, seven-repeat median:

| Engine | Cold | Short / 448 | Conversation / 512 | Long / 768 |
|---|---:|---:|---:|---:|
| Current upstream | 106.8 | 145.5 | 153.5 | 156.0 |
| Aqua Graph + SDPA fallback | 346.7 | 490.0 | 519.3 | 476.9 |
| Aqua Graph + FA2 `valid` (runtime default when available) | 398.6 | 568.5 | 627.4 | 644.9 |

See the [raw methodology and interpretation](results/4070ti-super-win-py312-torch251-cu124-upstream-overlay.md).

The old June 2026 claim that FlashAttention2 had no short-case benefit and only
about 8% long-case benefit predates the removal of the per-token CUDA Graph
replay synchronization. It is historical, not the current recommendation. The
runtime now prefers FA2 when it is installed and falls back to SDPA otherwise;
the benchmark still requires an explicit variant so A/B evidence stays clear.

Absolute rates are sensitive to Windows GPU P-state and desktop scheduling.
Use the JSON trial range and reproduce on the deployment GPU before treating a
percentage as portable.

### RTX 4070 Laptop GPU (8 GB)

The same Aqua Graph + FA2 `valid` command and deterministic protocol were run
on a render-free RTX 4070 Laptop GPU in the same Windows 11 dual-GPU host. The
software environment, checkpoint, fixed shapes, 15 warmups, and seven
synchronized repeats were unchanged.

| Engine | Cold | Short / 448 | Conversation / 512 | Long / 768 |
|---|---:|---:|---:|---:|
| Aqua Graph + FA2 `valid` | 313.4 | 542.1 | 585.3 | 568.1 |

This is isolated T2S AR throughput rather than TTFP. The result shows that the
optimized decoder remains comfortably real-time on an 8 GB, lower-power mobile
Ada GPU; full-pipeline latency still depends on the SoVITS/CFM/BigVGAN workload
and local audio scheduling.

## TTFP (time to first playable audio)

```bash
python benchmarks/aqua_ttfp.py \
  --gpt-model /path/to/xxx-e15.ckpt \
  --sovits-model /path/to/xxx_e2_s174_l32.pth \
  --sovits-pretrain /path/to/s2Gv3.pth \
  --bert-model /path/to/chinese-roberta-wwm-ext-large \
  --cnhubert-model /path/to/chinese-hubert-base \
  --bigvgan-model /path/to/models--nvidia--bigvgan_v2_24khz_100band_256x \
  --fast-langdetect-model /path/to/fast_langdetect \
  --ref-audio /path/to/ref_audio.wav \
  --ref-text "reference transcript"
```

TTFP measures text-ready to first non-empty audio returned by `infer_stream()`.
Use two or more warmup texts, five repeats per text, and report the median. Do
not place `torch.cuda.empty_cache()` in the hot path. TTFP intentionally does
not add CUDA synchronization inside its timing window because it measures the
latency visible to the caller.

Current warm result with a cached matching BigVGAN CUDA extension,
0.25-second chunks, two warmup utterances and five repeats. All columns share
the same Aqua text/SoVITS/BigVGAN pipeline and vary only T2S execution:

| T2S execution | Short (3 chars) | Medium (19 chars) | Long (64 chars) |
|---|---:|---:|---:|
| Current upstream dynamic path | 416.8 ms | 692.7 ms | 1135.2 ms |
| Aqua Graph + SDPA | 250.5 ms | 305.0 ms | 394.2 ms |
| Aqua Graph + FA2 `valid` | **233.1 ms** | **287.7 ms** | **348.3 ms** |

The first long-text repeat was 3059.3 ms because it triggered one-time frontend
initialization; the following four were 342.1–369.1 ms. Preserve cold-shape
outliers in raw evidence, but do not substitute one cold sample for the warm
five-repeat median.

## BigVGAN raw kernel timing

```bash
python benchmarks/bigvgan_raw_bench.py
```

The raw benchmark measures only the fp16 BigVGAN forward pass. Warm each mel
shape first, synchronize immediately around measured calls, and report medians
for mel lengths representative of first chunks and full utterances.

## Real-weight model compatibility

`model_smoke.py` checks actual checkpoint loading, non-streaming output, and
18 streaming trials (three Japanese/English texts × two seeds × dynamic/static/
CUDA Graph). It rejects silent/error fallback, checks output rate and finite
PCM, verifies actual graph replay, and verifies that Pro's main speaker reference
is encoded only once. `--flash` additionally requires active FlashAttention.
Use one process per model; weights and references are not included.

Create a local JSON config (absolute paths recommended):

```json
{
  "expected_version": "v2ProPlus",
  "inferencer": {
    "device": "cuda",
    "gpt_path": "/models/voice.ckpt",
    "sovits_path": "/models/voice.pth",
    "sv_model_path": "/models/pretrained_eres2netv2w24s4ep4.ckpt",
    "bert_path": "/models/chinese-roberta-wwm-ext-large",
    "cnhubert_path": "/models/chinese-hubert-base",
    "fast_langdetect_path": "/models/fast_langdetect"
  },
  "reference": {
    "ref_audio_path": "/voices/reference.wav",
    "prompt_text": "Matching reference transcript.",
    "prompt_language": "日文"
  }
}
```

```bash
GPT_SOVITS_HOME=/path/to/GPT-SoVITS python benchmarks/model_smoke.py \
  --config /path/to/local-config.json --output /path/to/local-results
```

The optional `texts` config field accepts a list of `text`/`text_language`
objects, for example to test Chinese using already-installed upstream G2PW
assets. Documented language labels are accepted independently of the UI locale.
V3 needs its BigVGAN directory and, for LoRA weights, its base SoVITS checkpoint.
Reports include checkpoint SHA-256 identities and timing observations, not a
controlled performance or perceptual-quality comparison. Keep audio and configs
local. See [the recorded validation](results/v2-model-support.md).

## Additional v2-family latency benchmarks

`model_latency.py` uses the same local config as `model_smoke.py` and the existing
short/medium/long TTFP texts. Run each engine in a fresh process:

```powershell
python benchmarks/model_latency.py --config F:\local\v2ProPlus.json --engine upstream --output F:\local\upstream.json
python benchmarks/model_latency.py --config F:\local\v2ProPlus.json --engine sdpa --output F:\local\sdpa.json
python benchmarks/model_latency.py --config F:\local\v2ProPlus.json --engine flash --output F:\local\flash.json
```

The upstream variant leaves native T2S unpatched but shares Aqua's frontend,
reference cache, semantic guard and SoVITS decoder. The other variants use CUDA
Graph with SDPA or FA2 `valid`; the FA2 benchmark fails on kernel fallback.
Two warmup utterances precede five seeded repeats per text. Results retain all
repeats, including first-shape initialization, and report median first audio,
total generation time, audio duration and RTF. The configured segmentation is
`按标点符号切`, output chunks are 0.25 s, and inter-segment pauses are 0.3 s.
This is model-side latency, excluding sound-device playback.

The [additional v2-family results](results/v2-model-latency.md) supplement the
existing v3 and mobile-GPU measurements above. Model/voice weights differ across
families; comparisons should be made between engines within one checkpoint pair.
Floating-point backend differences can change sampled tokens even with paired
seeds, so the report also includes semantic hashes and per-trial audio durations.
