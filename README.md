






<div align="center">

<img src="https://raw.githubusercontent.com/Lucas1479/Aqua-TTS/main/assets/aqua.png" width="720"/>

<h1>🌊 Aqua-TTS: <a href="https://github.com/RVC-Boss/GPT-SoVITS">GPT-SoVITS</a> Low-Latency Inference Runtime on GPU</h1>

<p>Built for low-latency voice conversation with your LoRA characters</p>

<p>
  <a href="https://github.com/Lucas1479/Aqua-TTS/blob/main/README.zh.md">中文</a> | English
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-blue" alt="Python"/>
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License"/>
  <img src="https://img.shields.io/badge/CUDA-11.8%2B-brightgreen" alt="CUDA"/>
</p>

</div>

---

Aqua-TTS is a GPU-optimized inference runtime purpose-built for **real-time voice conversation** — specifically, low-latency streaming TTS with your own [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) v3 LoRA character voices. It does not replace model weights — it replaces the execution strategy: static KV cache buffers, bucketed CUDA Graph capture/replay, FlashAttention2 when compatible, and an ABI-keyed BigVGAN CUDA extension cache. On an RTX 4070 Ti SUPER, the current deterministic benchmark reaches **568–645 synchronized it/s** with FA2 and **477–519 it/s** through the automatic SDPA fallback. An additional 8 GB RTX 4070 Laptop GPU validation sustained **542–585 synchronized it/s**, showing that the optimized AR path remains fast on a lower-power mobile GPU. Warm model-side first-audio medians are **233 / 288 / 348 ms** for the short, medium, and long cases below. In the full streaming player pipeline, practical first-audio latency is usually **0.4–0.7 s** depending on chunk length, audio device startup, cache state, and scheduling overhead.

**V2-family support:** Aqua-TTS also loads **v2, v2Pro, and v2ProPlus** checkpoints,
including the Pro/Plus speaker encoder. See the [v2-family benchmarks](#additional-v2-family-benchmarks)
and [checkpoint setup](#v2-v2-pro-and-v2-pro-plus-checkpoints) below. The original
v3 and RTX 4070 Laptop measurements are preserved in Highlights.

## Highlights

The timings below are historical first-PCM measurements, not first spoken sound;
they predate the [generated-lead trimming and onset fields](#speaker-demo).

<sub>**Latency definitions:** TTFP benchmark = model-side first audio latency under warm-cache (table below). E2E first-audio = full pipeline including audio buffer and playback startup, typically **0.4–0.7 s** in practice. Cold start = init + model load + first inference, dominated by BigVGAN CUDA kernel compilation (~2 min on first run, then cached).</sub>

The timing rows retain the published measurements from before the KV-layout
update. Mechanism rows describe the current runtime; new storage measurements
are documented in [the KV layout probe](benchmarks/README.md#graph-kv-storage-layouts).

| | Upstream T2S execution* | Aqua Graph + SDPA | Aqua default (FA2 `valid`) |
|---|---|---|---|
| T2S short / bucket 448 | 145.5 it/s | **490.0 it/s** | **568.5 it/s** |
| T2S conversation / bucket 512 | 153.5 it/s | **519.3 it/s** | **627.4 it/s** |
| T2S long / bucket 768 | 156.0 it/s | **476.9 it/s** | **644.9 it/s** |
| TTFP short (3 chars) | 416.8 ms | 250.5 ms | **233.1 ms** |
| TTFP medium (19 chars) | 692.7 ms | 305.0 ms | **287.7 ms** |
| TTFP long (64 chars) | 1135.2 ms | 394.2 ms | **348.3 ms** |
| Model definition | Direct upstream module | **Validated upstream + in-memory overlay** | **Validated upstream + in-memory overlay** |
| Decode attention | Native PyTorch, dynamic KV | SDPA over static bucket | **FA2 over true KV length** |
| KV-cache writes | Per-token `torch.cat` | **In-place `scatter_`** | **FA2 KV-cache update** |
| KV allocation | Grows per token | **Pre-allocated and bounded per bucket** | **Pre-allocated and bounded per bucket** |
| CUDA Graph | None in the standard entry point | **One per prepared bucket, up to 6 + lazy capture** | **One per prepared bucket, up to 6 + lazy capture** |
| Replay synchronization | Eager launches | **Stream-ordered; no per-token device sync** | **Stream-ordered; no per-token device sync** |
| EOS host read | Native condition checks | **Greedy + sampled EOS combined once** | **Greedy + sampled EOS combined once** |
| Graph concurrency | N/A | **Full AR lease per bucket, or shared nested root** | **Full AR lease per bucket, or shared nested root** |
| Failure path | Dynamic decoder | **Graph → static KV → dynamic** | **FA2 → SDPA; Graph → static KV → dynamic** |
| BigVGAN activation | Runtime extension/JIT path | **ABI-keyed CUDA cache → PyTorch fallback** | **ABI-keyed CUDA cache → PyTorch fallback** |
| Streaming contract | Native upstream generators | **Generators preserved; only direct `infer_panel()` patched** | **Generators preserved; only direct `infer_panel()` patched** |

*Benchmark environment: RTX 4070 Ti SUPER (16 GB), PyTorch 2.5.1+cu124, fp16, upstream `08d627c`. T2S throughput uses 15 warmups and seven synchronized repeats per fixed shape. TTFP uses the same Aqua text/SoVITS/BigVGAN pipeline for all three columns so the T2S execution mode is isolated; it uses two warmup utterances, five repeats, a matching cached BigVGAN CUDA extension, and 0.25 s chunks. The published v0.2.0 Aqua TTFP was ~257 / 301 / 404 ms; current FA2 medians are ~9% / 4% / 14% lower. FA2 is attempted automatically when importable and falls back to SDPA; set `AQUATTS_T2S_FLASH_ATTN=0` to force SDPA. One-time long-text frontend initialization produced a ~3 s first repeat, retained in the raw evidence but excluded by the median. See [benchmarks/README.md](https://github.com/Lucas1479/Aqua-TTS/blob/main/benchmarks/README.md) for commands and methodology.*

### 8 GB mobile GPU validation

The same deterministic FA2 `valid` protocol was also run on an **RTX 4070 Laptop GPU (8 GB)** in the same Windows 11 host. The mobile GPU was not driving a display; Python, PyTorch, FlashAttention2, checkpoint, fixed shapes, 15 warmups, and seven synchronized repeats were unchanged.

| GPU | Cold conversation | Short / 448 | Conversation / 512 | Long / 768 |
|---|---:|---:|---:|---:|
| RTX 4070 Laptop GPU (8 GB) | 313.4 it/s | **542.1 it/s** | **585.3 it/s** | **568.1 it/s** |

These are isolated T2S AR throughput measurements, not full-pipeline TTFP. They demonstrate that Aqua's graph + FA2 execution path can retain strong conversational throughput on an 8 GB, lower-power mobile Ada GPU.

https://github.com/user-attachments/assets/581cef5f-f8ce-4570-81ae-a6c092698223

## Features

- **Static KV cache** — pre-allocated scatter buffers eliminate per-step `torch.cat` overhead
- **Semantic stability guard** — rejects repeated-token collapse before the vocoder and retries once
- **Bucketed CUDA Graph** — one graph per prepared bucket across six configured capacities, with optional nested KV storage and lazy capture
- **Cached BigVGAN CUDA extension** — NVIDIA kernel cached by GPU/Python/Torch/CUDA ABI, with torch fallback
- **Adaptive FlashAttention2 KV cache** — `flash_attn_with_kvcache` `valid` mode is preferred automatically, with SDPA fallback
- **Streaming API** — generator-based `infer_stream()` with early first-audio yield
- **Built-in presets** — fast / balanced / quality generation presets; full / minimal / lazy / off CUDA Graph presets
- **Voice registry** — map voice names to reference audio + prompt, with JSON persistence
- **HTTP server** — lightweight FastAPI server with streaming TTS endpoint, voice management, and health check
- **PyPI install** — `pip install "aqua-tts[runtime]"` → `from aquatts import TTSInferencer`

> **Scope notice** — Aqua-TTS is an optimization layer for upstream GPT-SoVITS **v2, v2Pro, v2ProPlus, and v3** (with the existing v1 path retained). It loads the selected upstream `Text2SemanticDecoder`, validates the required contract, and patches only the direct `infer_panel()` path. Upstream batching and streaming entry points remain intact; incompatible upstream changes fail closed. GPT-SoVITS v4 is not currently supported.

> **Known limitations** — Windows + CUDA is the primary tested path. Linux passes unit tests but GPU-dependent paths (CUDA Graph, BigVGAN kernel) have not been validated on Linux hardware. macOS is not supported. TTFP varies with GPU model, audio device, chunk size, and model weights — the headline TTFP numbers in this README were measured on an RTX 4070 Ti SUPER with specific v3 LoRA weights, while the RTX 4070 Laptop result covers isolated T2S AR throughput only. Neither should be treated as universal. V2-family checkpoints use their own SoVITS decoder; the v3 BigVGAN latency figures do not describe v2-family performance.

## Additional v2-family benchmarks

**Latest measurement: 2026-09-27.** RTX 4070 Ti SUPER (16 GB), Windows,
PyTorch 2.5.1+cu121, fp16, upstream `08d627c`. The table reports **warm model-side
first-PCM latency**, excluding model loading and audio playback. Short, medium,
and long inputs contain 3, 19, and 64 characters respectively.

Each model/engine/text combination uses five warmups followed by twenty requests,
resetting seed `20260926` each time: **540 measured requests** in total. First use
and synchronized stage diagnostics are recorded separately; no new CUDA Graphs
were captured during measured requests. All three engines share the same Aqua
frontend, reference cache, and SoVITS path for each checkpoint pair.

**First-audio latency (TTFP, median / p50; lower is better)**

| Model / input | Upstream T2S execution | Aqua Graph + SDPA | Aqua default (FA2 `valid`) |
|---|---:|---:|---:|
| v2 · short (3 chars) | 255.7 ms | 107.5 ms | **93.9 ms** |
| v2 · medium (19 chars) | 519.5 ms | 177.1 ms | **142.6 ms** |
| v2 · long (64 chars) | 1005.1 ms | 306.0 ms | **231.0 ms** |
| v2Pro · short (3 chars) | 272.7 ms | 110.9 ms | **96.4 ms** |
| v2Pro · medium (19 chars) | 547.4 ms | 183.4 ms | **143.1 ms** |
| v2Pro · long (64 chars) | 965.4 ms | 293.0 ms | **221.4 ms** |
| v2ProPlus · short (3 chars) | 252.9 ms | 107.0 ms | **100.7 ms** |
| v2ProPlus · medium (19 chars) | 537.0 ms | 180.2 ms | **144.7 ms** |
| v2ProPlus · long (64 chars) | 927.2 ms | 291.9 ms | **219.6 ms** |

**Short-input first-audio latency (p95; lower is better)**

| Model / input | Upstream T2S execution | Aqua Graph + SDPA | Aqua default (FA2 `valid`) |
|---|---:|---:|---:|
| v2 · short (3 chars) | 293.6 ms | 116.4 ms | **103.3 ms** |
| v2Pro · short (3 chars) | 303.7 ms | 122.5 ms | **110.3 ms** |
| v2ProPlus · short (3 chars) | 283.3 ms | 119.8 ms | **107.4 ms** |

V2 Pro/FA2 short-text minimum was **93.1 ms** (median **96.4 ms**, p95 **110.3 ms**).
v2 uses official base weights; Pro/Plus use the tested Kurisu fine-tunes. Compare
engines within one checkpoint pair. FA2 can produce different sampled tokens,
so these are observed same-input latencies, not fixed-token speedup claims.
V2-family output is 32 kHz and is chunked after each text segment is decoded.
Desktop GPU activity remained present during the measurements.

The [full retest report](benchmarks/results/v2-model-retest-20260927.md),
[raw trials](benchmarks/results/v2-model-retest-20260927.json), and
[GPU telemetry](benchmarks/results/v2-model-retest-20260927-gpu.csv) include
first-use timings, total time/RTF, checkpoint identities, and separate diagnostics.
The [earlier paired-seed report](benchmarks/results/v2-model-latency.md) retains
the original v2 measurements. Its different protocol prevents treating the
difference as an implementation speedup. Existing v3 and RTX 4070 Laptop figures
above remain unchanged and use their own documented workloads and environments.

## V2, V2 Pro, and V2 Pro Plus checkpoints

Use a matching GPT/SoVITS checkpoint pair. Aqua reads the SoVITS header or
upstream model identity, so renaming a checkpoint does not select its architecture.
Pro and Plus additionally require the shared ERes2Net speaker encoder:

```python
from aquatts import TTSInferencer

tts = TTSInferencer(
    gpt_path="/models/voice.ckpt",
    sovits_path="/models/voice.pth",
    sv_model_path="/models/pretrained_eres2netv2w24s4ep4.ckpt",  # Pro/Plus only
    language="zh_CN",  # optional UI locale; inference language labels stay stable
)
sr, audio = tts.infer(
    text="今日はいい天気ですね。",
    ref_audio_path="/voices/reference.wav",
    prompt_text="Matching reference transcript.",
    text_language="日文", prompt_language="日文",
)
```

`GPT_SOVITS_HOME` must select an upstream checkout with Pro model definitions
and `GPT_SoVITS/eres2net/` (validated against `08d627c`). If `sv_model_path` is
omitted, Aqua uses `GPT_SOVITS_HOME/GPT_SoVITS/pretrained_models/sv/`
`pretrained_eres2netv2w24s4ep4.ckpt`. The speaker weight remains external; Aqua
loads it only for Pro/Plus. Plain v2 does not need it. V2-family inference does
not load a v3 base model or BigVGAN. CNHuBERT and the relevant text frontend
assets are required; BERT assets are needed when Chinese features are requested.

Both `infer()` and `infer_stream()` accept these models. The main reference's
speaker embedding is cached per session; additional references receive matched
spectra and speaker embeddings. V2-family streaming decodes each text segment
before splitting its audio into output chunks. `chunk_size_seconds` controls
transport chunks, not incremental semantic or acoustic decoding. T2S static KV,
CUDA Graph, and optional FlashAttention remain available; v3 CFM/BigVGAN options
such as `sample_steps` do not accelerate v2-family decoding.

For the server, add `--sv-model /models/pretrained_eres2netv2w24s4ep4.ckpt` to
`python -m aquatts.server --gpt-model ... --sovits-model ...`. `/tts` returns mono
float32 PCM with the actual `X-Sample-Rate`; `/tts/file` records that same rate in
the WAV header (normally 32 kHz for v2-family weights, 24 kHz for v3). Streaming
failures propagate instead of being replaced with a silent audio block.

See [the model validation report](benchmarks/results/v2-model-support.md) for
exact tested weights and limits. This change adds no MPS or ROCm support.

### Optional model loading

BERT is lazy by default: the first Chinese target or reference segment loads its
tokenizer and model. English/Japanese-only targets **and references** leave BERT
unloaded. Mixed-language input loads it if the frontend produces a Chinese segment.
The first Chinese request pays the loading cost. Set `preload_bert=True` in the
constructor, or call `tts.preload_bert_model()` during application preparation.
`tts.bert_loaded` and `tts.bert_load_seconds` expose the current state and the first
successful load duration. Concurrent first use loads once; a failed load can retry.

AP-BWE's module and model load only when super-resolution is requested. Missing
optional dependencies or weights retain the existing audio fallback and warning.

### Reference readiness

Prepare a reference through the same normalization and cache used by both APIs:

```python
result = tts.warm_reference_cache("/voices/reference.wav", "こんにちは。", "日文")
assert result["ready"]  # status is "built" or "hit"
```

The constructor also accepts `warmup_references=[voice, ...]`, where entries are
`Voice` objects or dictionaries with `ref_audio_path`, `prompt_text` and
`prompt_language`. Only supplied references are prepared. Results are available
as `tts.reference_warmup_results`. Strict warmup raises on failure; use
`warmup_strict=False` for constructor best effort, or `strict=False` on the public
method, to receive an explicit `failed` result.

Keys include the canonical path, file mtime/size, normalized text/language, model
generation and actual device/dtype. File replacement causes a miss; explicit
`tts.clear_reference_cache()` invalidates entries. V3 empty-reference fallback
and ordinary inference share the same key. Non-V3 reference-free requests cache
acoustic features without encoding an unused semantic prompt. Concurrent builds
publish complete entries once, and CUDA events order reuse across streams.
Explicit warmup synchronizes its selected device before reporting readiness.

`AQUA_SESSION_CACHE_MAX` is a positive capacity (default 8), with FIFO eviction.
A constructor list with more distinct normalized references than capacity is
rejected. A successful warmup result also reports its key, resident count and any
evicted key. Device/dtype changes require a new inferencer when CUDA Graphs are
captured. Omitting warmup keeps on-demand reference preparation.

### CUDA device and vocoder preparation

`device="cuda"` resolves the current CUDA device during construction; `cuda:N`
selects a visible process index. Unavailable CUDA/index selections fail before
model loading. `CUDA_VISIBLE_DEVICES` remapping is respected. BigVGAN extension
cache selection and calls follow the selected CUDA context.

BigVGAN keeps the default 40/70/128 mel-frame warmup, five calls each. Override
`bigvgan_warmup_shapes=[9, 23, 47]` and `bigvgan_warmup_repeats=2`, or pass an empty
shape list to disable it. Omitted arguments read `AQUATTS_BIGVGAN_WARMUP_SHAPES`
(comma-separated; empty disables) and `AQUATTS_BIGVGAN_WARMUP_REPEATS`; explicit
arguments win. Shapes are deduplicated and must be positive integers. Limits are
16 distinct shapes, 4096 frames per shape, 20 repeats and 65536 total warmed frames.

Warmup uses the model's dtype/device/mel channels in inference mode and preserves
CPU and selected-device RNG state. `tts.vocoder_warmup_stats` reports completion,
disablement or failure; failures are logged. `tts.vocoder_shape_counts` records
actual request mel lengths. With 24 kHz V3 output, 0.1/0.25/0.5-second chunks can
use 9/23/47 mel frames plus shorter tails, so defaults do not cover every workload.
Normal vocoder calls add no device-wide synchronization; diagnostic timing remains
opt-in. `load_bigvgan(..., device="cuda:N")` also accepts an explicit device.

## Supported Languages

Aqua-TTS uses the selected GPT-SoVITS checkpoint's text frontend. Pass the code to `text_language` / `prompt_language` — reference audio and target text can use different languages.

| Language | Code |
|---|---|
| Japanese | `日文` |
| Chinese | `中文` |
| English | `英文` |

## How it works

Aqua-TTS loads model definitions and text/runtime modules from the checkout selected by `GPT_SOVITS_HOME`. After the checkpoint is loaded it applies the static-KV/CUDA-Graph optimization in place. The small namespace bridge under `_vendor/` redirects only BigVGAN's CUDA activation to Aqua's canonical extension loader; it does not contain a forked `t2s_model.py`:

```
aqua-tts/
├── aquatts/                       # Pure Python package (pip-installable)
│   ├── __init__.py                # sys.path configuration + lazy exports
│   ├── upstream.py                # Upstream checkout validation and routing
│   ├── inferencer.py              # TTSInferencer — main entry point
│   ├── server.py                  # FastAPI HTTP server
│   ├── voice_registry.py          # Voice name → audio path mapping
│   ├── modeling/
│   │   ├── t2s_streaming.py       # T2SBlockWithStaticCache, CUDA Graph patch
│   │   └── t2s_flash_attn.py      # Adaptive FlashAttention2 KV-cache patch
│   ├── bigvgan/
│   │   ├── cuda/                  # Standalone CUDA kernel loader + sources
│   │   └── torch/                 # Pure-PyTorch fallback (resample, filter, act)
│   ├── inference/
│   │   ├── streaming.py           # Audio post-processing (fade, chunk finalization)
│   │   ├── semantic_stability.py  # Pre-vocoder semantic collapse admission checks
│   │   ├── params.py              # SoVITS parameter presets
│   │   └── presets.py             # Named presets (generation + CUDA Graph)
│   └── _vendor/
│       └── GPT_SoVITS/            # Namespace bridge; no model fork
│           └── BigVGAN/alias_free_activation/cuda/  # Thin Aqua loader adapters
├── benchmarks/                    # TTFP, T2S comparison, BigVGAN raw benchmarks
├── examples/                      # basic_usage.py, streaming_inference.py
└── tests/                         # Unit tests
```

See **[TECHNICAL.md](https://github.com/Lucas1479/Aqua-TTS/blob/main/TECHNICAL.md)** for deep technical documentation.

## Requirements

| | Entry | Recommended |
|---|---|---|
| GPU | RTX 3060 6 GB | RTX 4060 / 4070 8 GB+ |
| CUDA | 11.8+ | 12.x |
| RAM | 8 GB | 16 GB+ |
| Python | 3.10 | 3.11 / 3.12 |
| OS | Windows 10+ | Windows 11 |

Linux CI (Ubuntu) passes for all unit tests. GPU-dependent paths (CUDA Graph, BigVGAN kernel) have not been tested on Linux hardware. macOS is not supported — Aqua-TTS requires CUDA.

> On 6 GB cards, use `cuda_graph_preset="lazy"` or `"off"` to reduce VRAM pressure from pre-captured graphs.

## Installation

### 1. Install GPT-SoVITS

You need a working GPT-SoVITS v3 installation. Aqua-TTS imports from it.

```bash
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
cd GPT-SoVITS
pip install -r requirements.txt
```

Aqua-TTS has been tested against GPT-SoVITS v3 (2025-04-01 release).

### 2. Install PyTorch

Install PyTorch matching your CUDA version **before** installing Aqua-TTS — it is not included in the package dependencies so you can pick the right CUDA wheel:

```bash
# Example for CUDA 12.1 — see https://pytorch.org/get-started/locally/ for other versions
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Aqua-TTS has been tested with PyTorch 2.5.1+cu121 on an RTX 4070 Ti SUPER.

### 3. Install Aqua-TTS

```bash
# Core + runtime dependencies from PyPI
pip install "aqua-tts[runtime]"

# Or with HTTP server support
pip install "aqua-tts[server]"

# Or with local playback support
pip install "aqua-tts[playback]"
```

For development or unreleased changes, install from source instead:

```bash
git clone https://github.com/Lucas1479/Aqua-TTS.git
cd Aqua-TTS
pip install -e ".[runtime]"
```

Extras:
- `[runtime]` — soundfile, librosa, peft (needed by `TTSInferencer`)
- `[server]` — FastAPI + uvicorn (includes `[runtime]` automatically)
- `[playback]` — PyAudio (includes `[runtime]` automatically)

### 4. Configure GPT-SoVITS path

Set the `GPT_SOVITS_HOME` environment variable to your GPT-SoVITS repo root:

```bash
# Windows (PowerShell)
$env:GPT_SOVITS_HOME = "C:\path\to\GPT-SoVITS"

# Linux / macOS
export GPT_SOVITS_HOME=/path/to/GPT-SoVITS
```

### 5. Download pretrained models

Aqua-TTS needs BigVGAN v2 pretrained weights from [Hugging Face](https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x). Download them directly into your GPT-SoVITS repo with `huggingface_hub`:

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='nvidia/bigvgan_v2_24khz_100band_256x',
    local_dir='GPT_SoVITS/pretrained_models/models--nvidia--bigvgan_v2_24khz_100band_256x',
)
"
```

The `models--nvidia--bigvgan_v2_24khz_100band_256x` directory name is the HuggingFace Hub cache format — use it exactly as shown.

### 6. Prepare model weights

- **GPT weights (T2S)**: A `Text2SemanticLightningModule` checkpoint (e.g., `s1v3.ckpt` or self-trained `xxx-e15.ckpt`).
- **SoVITS weights (vocoder)**: A `SynthesizerTrn` checkpoint with optional LoRA (e.g., `xxx_e2_s174_l32.pth`).
- **Reference audio**: A 3-10 second 24 kHz WAV file with known transcript.

### Quick sanity check

Once `GPT_SOVITS_HOME` is set and models are in place, verify the package loads correctly:

```bash
python -c "from aquatts import TTSInferencer; print('aquatts OK')"
```

If this prints `aquatts OK`, the install is good. If it errors, check that `GPT_SOVITS_HOME` points to the right directory and all pretrained models are downloaded.

## Usage

### Python API

```python
import os
os.environ["GPT_SOVITS_HOME"] = "/path/to/GPT-SoVITS"
os.environ["ENABLE_CUDA_GRAPH"] = "1"

from aquatts import TTSInferencer

tts = TTSInferencer(
    device="cuda",
    gpt_path="GPT_weights_v3/s1v3.ckpt",
    sovits_path="SoVITS_weights_v3/your_model.pth",
    cuda_graph_preset="full",  # "full", "minimal", "lazy", or "off"
)

# Streaming generation — yields (sample_rate, audio_chunk, text) tuples
for sr, chunk, text in tts.infer_stream(
    text="こんにちは、世界！",
    ref_audio_path="reference audio/ref_audio.wav",
    prompt_text="こんにちは。今日はいい天気ですね。",
    text_language="日文",
    prompt_language="日文",
    preset="fast",  # "fast", "balanced", or "quality"
):
    # chunk is float32 numpy array at `sr` Hz
    # text is the text segment being spoken
    pass
```

See `examples/basic_usage.py` and `examples/streaming_inference.py` for runnable examples.

### Presets

Two layers of presets control quality/speed trade-offs:

**Generation presets** (per-request, via `infer_stream(preset=...)`):

| Preset | Speed | Sample Steps | Top-K | Temperature |
|--------|-------|-------------|-------|-------------|
| `fast` | 1.3x | 4 | 3 | 0.6 |
| `balanced` | 1.1x | 4 | 5 | 0.6 |
| `quality` | 1.0x | 16 | 8 | 0.8 |

**CUDA Graph presets** (system-level, via `TTSInferencer(cuda_graph_preset=...)`):

| Preset | Pre-capture | Buckets | Description |
|--------|-------------|---------|-------------|
| `full` | Yes | 128, 256, 448, 512, 768, 1024 | All lengths covered |
| `minimal` | Yes | 256, 512, 1024 | Common lengths only |
| `lazy` | No | on-the-fly | Lower memory, slower TTFP |
| `off` | Disabled | none | Static KV only, no graphs |

`infer()` and `infer_stream()` inherit the instance's Graph setting when
`enable_cuda_graph` is omitted or `None`. Explicit `False` disables replay for
that request; `True` opts in when NVIDIA CUDA and static KV are available,
including lazy capture on an `off` instance. `enable_static_kv=False` always
disables Graph. `ENABLE_CUDA_GRAPH=0/1` overrides the preset at construction;
`ENABLE_CUDA_GRAPH_PRECAPTURE` and `CUDA_GRAPH_PRECAPTURE_BUCKETS` override startup
capture only. Later environment changes do not alter an existing instance.

Complete pre-capture for the buckets you intend to serve **before accepting
requests**. Capture requires exclusive admission across cooperating CUDA work
in the process; an active request (including a paused stream) defers it, and
the requesting decode falls back to eager execution. With continuous overlapping
requests, an unprepared bucket can remain eager indefinitely. This affects
`lazy`, buckets omitted by `minimal`, failed startup captures, and instances
constructed after serving begins. `full` avoids request-time capture for its
configured buckets only when startup capture succeeds. To prepare deferred
buckets, drain active streams and call
`tts.t2s_model.model.precapture_cuda_graph([128, 256, 448, 512, 768, 1024])`,
choosing the capacities within your memory budget. Check the returned per-bucket
booleans before reopening traffic. No background capture or fairness guarantee
forces an idle window. Other CUDA users in the same process must also participate
in the shared work admission.

Use `tts.t2s_model.model.cuda_graph_stats["graph_replay_steps"]` after a request
to verify execution; a successful capture alone does not demonstrate replay.

Requests sharing a decoder Graph workspace are serialized for their complete
T2S AR generation, including prompt installation and sampling. CUDA events order
handoff between streams; this does not add per-token device synchronization.
Different storage owners and separate decoder instances retain independent ownership.
Graph-disabled requests use private KV. Counters describe the last completed AR
request; they are not an aggregate across concurrent requests.

Each computation bucket now captures one Graph, reused across real prefix lengths.
`TTSInferencer(kv_cache_layout="bucket")` keeps independent KV per bucket and is
the default. Opt in to `kv_cache_layout="nested"` to share a maximum-capacity root
across shorter bucket views. This reduces multi-bucket KV residency while retaining
short attention shapes; all Graph requests on that root share one owner. An omitted
argument reads `AQUATTS_KV_CACHE_LAYOUT` once (default `bucket`). Explicit arguments
take priority. Captured layout/device/dtype/capacity changes require a new decoder.
Sparse single-bucket deployments can use less KV with `bucket`; `nested` is not
universally smaller. These are deliberate defaults for this release: `bucket`
keeps per-bucket ownership independent, capture presets bound startup memory,
and CFM unpadded remains an explicit experiment pending further hardware and
quality validation. Use `nested` when shared KV residency and prepared-bucket
promotion fit your workload. Full/minimal capture only their declared bucket sets; lazy/off
do not allocate a Graph root at startup. A full nested bucket promotes to the
smallest larger **already captured** bucket, retaining the live KV prefix in place
and clearing only the newly exposed tail. Promotion never captures or expands the
preparation set. Without a prepared target (or with bucket storage), decoding
preserves the full prefix and continues dynamically.

`decoder.cuda_graph_stats` describes the last completed request: initial/final KV,
start/final bucket, promotions, Graph/static/dynamic steps, capture attempts and
workspace wait seconds. Numeric request logs expose the same path without text.

**Adaptive FlashAttention2 T2S path** (preferred automatically when available):

Aqua-TTS can replace the q_len=1 T2S static-KV attention step with
`flash_attn_with_kvcache`. The `valid` mode attends over the true KV length
rather than the full zero-padded bucket and is now the default whenever a
compatible FlashAttention2 installation can be imported. If FA2 is absent or a
kernel rejects the active shape/device, Aqua falls back to SDPA.

No environment variable is required. Override the automatic choice when needed:

```bash
export AQUATTS_T2S_FLASH_ATTN=0  # force SDPA
# export AQUATTS_T2S_FLASH_ATTN=1  # explicitly request FA2
export AQUATTS_T2S_FLASH_ATTN_MODE=valid  # valid | bucket
```

or from Python with `use_flash_attn=False` / `True`:

```python
tts = TTSInferencer(..., use_flash_attn=True, flash_attn_mode="valid")
```

The old June 2026 guidance that FA2 had no short-case benefit and only about 8%
long-case benefit was measured before CUDA Graph replay synchronization became
diagnostic-only, so it is no longer the current performance conclusion.

Deterministic A/B after removing the per-token replay synchronization
(RTX 4070 Ti SUPER, PyTorch 2.5.1+cu124, flash-attn 2.7.0.post2,
15 warmups, seven synchronized repeats):

| Case | SDPA fallback | Default FA2 (`valid`) | Directional gain |
|---|---:|---:|---:|
| T2S short / bucket 448 | 490.0 it/s | 568.5 it/s | ~16% |
| T2S conversation / bucket 512 | 519.3 it/s | 627.4 it/s | ~21% |
| T2S long / bucket 768 | 476.9 it/s | 644.9 it/s | ~35% |

The shared device-wide synchronization previously hid part of the attention
kernel difference. With one EOS synchronization per step, FA2's valid-length
KV reads are much more visible, especially at bucket 768. Treat the percentages
as hardware-specific and reproduce them on the deployment GPU before changing
the automatic policy or requiring bit-for-bit continuity with the SDPA path.

```python
from aquatts import apply_preset, list_presets

print(list_presets())  # ["balanced", "fast", "quality"]
params = apply_preset("fast", overrides={"top_k": 5})
```

### HTTP Server

Start a lightweight inference server:

```bash
python -m aquatts.server \
    --gpt-model GPT_weights_v3/s1v3.ckpt \
    --sovits-model SoVITS_weights_v3/model.pth \
    --cuda-graph-preset full \
    --host 127.0.0.1 --port 8000

# With authentication (required when binding to a non-loopback address)
python -m aquatts.server ... --host 0.0.0.0 --api-key mysecrettoken
# or: export AQUA_API_KEY=mysecrettoken
```

Endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Model load status |
| `GET` | `/presets` | List available presets |
| `POST` | `/tts` | Streaming TTS (float32 PCM chunks) |
| `POST` | `/tts/file` | One-shot TTS (downloadable .wav) |
| `GET` | `/voices` | List registered voices |
| `POST` | `/voices/add` | Register a new voice |
| `DELETE` | `/voices/{name}` | Remove a registered voice |

`/tts` prepares and validates the reference before sending success headers.
Reference preparation failures return HTTP 500 with a JSON `detail`, without
`X-Sample-Rate`. A cold reference can therefore delay headers until preparation
finishes; cached references reuse the prepared features.

```bash
# Streaming TTS — returns raw float32 PCM (24kHz, mono, little-endian)
curl -X POST "http://127.0.0.1:8000/tts?text=Hello&voice=alice" --output - \
  | play -t raw -r 24k -e floating-point -b 32 -c 1 -

# Download WAV file
curl -X POST "http://127.0.0.1:8000/tts/file?text=Hello&voice=alice" -o output.wav
```

### Voice Registry

Manage multiple characters/voices without repeating paths:

```python
from aquatts import Voice, VoiceRegistry

registry = VoiceRegistry(json_path="./voices.json")

registry.add(Voice(
    name="alice",
    ref_audio_path="./voices/alice_ref.wav",
    prompt_text="こんにちは。今日はいい天気ですね。",
    prompt_language="日文",
))

voice = registry.get("alice")
for sr, chunk, text in tts.infer_stream(
    text="Hello world",
    ref_audio_path=voice.ref_audio_path,
    prompt_text=voice.prompt_text,
    prompt_language=voice.prompt_language,
):
    pass
```

On the HTTP server, pass `voice` instead of `ref_audio_path`:

```bash
curl -X POST "http://127.0.0.1:8000/tts?text=Hello&voice=alice"
```

For production deployments, pass an explicit registry path:

```bash
python -m aquatts.server ... --voice-registry /data/voices.json
# or
export AQUA_VOICE_JSON=/data/voices.json
```

Live reference preparation errors propagate before semantic generation for all
model versions. Only complete cached references are reused. Optional explicit
warmup may still use `strict=False` and report a failed status. V3 local reference
audio is decoded with SoundFile; TorchAudio remains a resampling dependency and
does not require TorchCodec for this path.

### Concurrent CUDA work and graph capture

Aqua inference, full streaming-generator lifetimes, constructors, reference/BERT
preload and vocoder preparation participate in one process-wide admission gate.
Requests may run concurrently. Capture is deferred when another work owner is
active; the current request uses eager decoding and can retry capture later.
New work waits while an admitted capture is in progress. Generator ownership
survives worker-thread migration, and HTTP cancellation closes its lease.

Other GPU clients in the same process should participate with a synchronous
`with cuda_execution.work():` scope, importing `cuda_execution` from
`aquatts.inference.cuda_execution`. Arbitrary external CUDA calls bypassing this
cooperative gate cannot be protected by Aqua. A suspended stream retains its
lease until exhausted or explicitly closed.

### V3 semantic and acoustic references

Both `infer()` and `infer_stream()` accept an optional `semantic_reference`,
using a `Voice` or a dictionary with `ref_audio_path`, `prompt_text` and
`prompt_language` (default Japanese). The ordinary reference supplies acoustic
conditions; the explicit reference supplies semantic conditioning:

| Fields | Source |
| --- | --- |
| `prompt`, `phones1`, `bert1` | Semantic reference |
| `refer_spec`, `prompt_fea_ref`, `prompt_ge`, `mel2_norm` | Ordinary acoustic reference |

```python
from aquatts import Voice

semantic = Voice("semantic", "semantic.wav", "Its actual transcript.", "英文")
audio = tts.infer(
    "Target sentence.", ref_audio_path="acoustic.wav",
    prompt_text="Acoustic reference transcript.", prompt_language="英文",
    text_language="英文", semantic_reference=semantic,
)
```

When mixing, both transcripts must be explicit and nonempty, languages must be
recognized, and `ref_free=True` is rejected. Non-V3 models reject the parameter.
Incomplete references fail without silently substituting the other source.
Each request combines independent copies; shared cache entries are not modified.
Warm both references with `warmup_references` or `warm_reference_cache`; keeping
both resident requires cache capacity of at least two distinct normalized entries.
Omitting `semantic_reference` preserves ordinary reference behavior. This API
defines conditioning sources; it does not guarantee a fixed emotion strength or
unchanged speaker identity. Voice quality and listening review remain separate.

### Experimental V3 CFM mask optimization

`TTSInferencer(cfm_unpadded=True)` opts into an experimental single-item path;
the default is `False`. Aqua already knows the complete input length when it
constructs each V3 CFM chunk. Only that known unpadded case omits the attention
mask. Unknown lengths, batches and padded inputs retain a mask, prepared once
outside the diffusion loop. No GPU length scalar is read back to the CPU.

The instance adapter changes only DiT's mask assignment in memory after checking
the upstream interface. The CFM diffusion loop, `infer=True` tuple protocol,
conditional/CFG text caches, timestep cache and training branch stay upstream.
It does not replace external checkout files or class/global functions. Unsupported
upstream implementations fail explicitly when enabled. Disable the option to
restore the ordinary masked path; non-V3 models reject it.

Use `benchmarks/cfm_unpadded_bench.py --config ... --output ...` for same-noise
4/16/32-step, CFG, FP16/FP32 comparisons, short tails, long references and paired
ordinary/streaming requests. Numerical bounds are fixed in the benchmark before
measurement; raw repeat noise and timings are retained. This remains disabled
by default pending broader hardware and listening review.

### Local resources and offline model loading

Pass `offline_only=True` to `TTSInferencer` (or `--offline-only` in
`examples/basic_usage.py`) to require preinstalled resources. The default is
`False`. BERT and both BigVGAN loaders pass `local_files_only=True`; CNHuBERT
already uses local-only Transformers loading. Missing files name their component
and expected directory. BERT checks remain lazy until Chinese use or explicit
`preload_bert_model()`. FastText uses an instance-owned detector and the local
`fast_langdetect_path/lid.176.bin`, with no download fallback. Online instances
keep upstream's global detector and download behavior, without requiring the
private splitter adapter. Offline detectors remain instance-owned.

Install frontend resources before their first import:

- English: NLTK `cmudict` and `averaged_perceptron_tagger` zip resources; NLTK
  3.9+ also needs `averaged_perceptron_tagger_eng`.
- Japanese: a complete OpenJTalk dictionary at `OPEN_JTALK_DICT_DIR` (set before
  importing pyopenjtalk), or its installed package default.
- Chinese: upstream G2PW still reads `GPT_SoVITS/text/G2PWModel` relative to the
  process working directory. It needs config, character tables, pronunciation
  JSON files and `g2pW.onnx`/`g2pw.onnx`. Its tokenizer uses upstream `bert_path`
  (set before importing `text.chinese2`), defaulting to
  `GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large`. This is separate
  from the constructor's `bert_path`; both must resolve locally. Configure the
  process resource layout once; Aqua does not change cwd during inference.
- Optional AP-BWE: `AP_BWE_main/24kto48k/{config.json,g_24kto48k.zip}` beside the
  selected upstream `tools/audio_sr.py`. Checks and loading use that same path,
  independently of cwd. Its existing missing-component fallback remains unchanged.

The checked frontend boundary covers English, Japanese, Chinese and mixtures
of these; offline Korean/Cantonese requests fail explicitly. This is a model
resource contract, not a network sandbox. It does not set Hub environment flags
or mutate Hub private state, and works after Hub/Transformers have been imported.
Upstream's initial LangSegmenter import still sets its own process default;
initialize that upstream module before other users of its global default, or use
explicit detector instances. Offline constructors bind private functions without
modifying the upstream functions or that default. Online constructors retain the
upstream global-detector configuration. Tested with upstream
`08d627c`, fast-langdetect 1.0.1, split-lang 2.1.1, g2p-en 2.1.0 and pyopenjtalk
0.4.1; incompatible private splitter interfaces fail only in offline mode.

`benchmarks/optional_models_smoke.py --offline-only --config ... --output ...
--frontend-cwd ...` blocks network connections in a fresh process and exercises
real local English/Japanese/Chinese/mixed inference. No weights are downloaded.

## Configuration

| Env variable | Default | Description |
|---|---|---|
| `GPT_SOVITS_HOME` | *(required)* | Path to GPT-SoVITS repo root |
| `BIGVGAN_CACHE_ROOT` | package cache | Optional writable/product-owned compiled extension cache root |
| `AQUA_API_KEY` | *(unset)* | Bearer token for all server endpoints; unset = no auth |
| `AQUA_VOICE_JSON` | `./voices.json` | Path to voice registry JSON file. **Always set this** — default is relative to process CWD and will be lost on directory change |
| `AQUA_SESSION_CACHE_MAX` | `8` | Max number of cached reference audio sessions |
| `ENABLE_CUDA_GRAPH` | `1` | Enable CUDA Graph replay |
| `ENABLE_CUDA_GRAPH_PRECAPTURE` | `1` | Pre-capture all bucket graphs at startup |
| `AQUATTS_T2S_FLASH_ATTN` | `auto` / unset | `0` forces SDPA; `1` explicitly requests FlashAttention2; `auto`, empty, or unset prefers FA2 when importable |
| `AQUATTS_T2S_FLASH_ATTN_MODE` | `valid` | FlashAttention mode: `valid` uses true KV length; `bucket` preserves zero-padded bucket length |
| `TTS_OUTPUT_LANGUAGE` | `日文` | Default output language. Change to `中文` or `英文` if not using Japanese |
| `TTS_REF_TEXT_JA` | `こんにちは。今日はいい天気ですね。` | Default Japanese reference text |
| `TTS_REF_TEXT_EN` | *(empty)* | Default English reference text |
| `TTS_STREAM_SYNC_TIMING` | `0` | Enable per-step CFM timing (adds GPU sync overhead) |

## Benchmarks

```bash
# T2S comparison (official vs official+CUDA Graph vs Aqua-TTS)
# Replace with your own GPT checkpoint (s1v3.ckpt or a self-trained .ckpt)
python benchmarks/t2s_comparison_bench.py \
  --upstream-home /path/to/GPT-SoVITS \
  --gpt-model /path/to/xxx-e15.ckpt \
  --flash-ab

# TTFP benchmark (streaming end-to-end)
python benchmarks/aqua_ttfp.py \
    --gpt-model GPT_weights_v3/xxx-e15.ckpt \
    --sovits-model SoVITS_weights_v3/xxx_e2_s174_l32.pth \
    --ref-audio "reference audio/ref_audio.wav" \
    --ref-text "transcript of reference audio"

# BigVGAN raw kernel timing
python benchmarks/bigvgan_raw_bench.py
```

See [benchmarks/README.md](https://github.com/Lucas1479/Aqua-TTS/blob/main/benchmarks/README.md) for full methodology and results.

## Speaker Demo

Generated leading silence is trimmed per synthesis item, retaining 50 ms before
the first 10 ms RMS window at or above -45 dBFS. Internal pauses, trailing audio,
and `pause_second` are retained; entirely quiet items are returned intact. The
streaming path waits for complete detection frames and carries text on the first
emitted PCM chunk. This amplitude rule is not a linguistic voice detector and
does not promise the same improvement for every checkpoint or reference.

Historical TTFP tables measure first nonempty PCM, not first spoken sound, and
predate this onset processing. The demo retains that metric and adds `lead_ms`
(PCM duration preceding the first active window) and `first_voiced_chunk_ms`
(when the containing chunk is ready for submission). Neither measures speaker
output latency. Buffering silent generated chunks can increase reported TTFP
while reducing the silence that playback submits; compare voiced PCM positions
and listening as well as packet timings.

```bash
pip install -e ".[playback]"
python examples/play_ete.py --gpt-sovits-home /path/to/GPT-SoVITS-v3lora
```

The demo plays three Kurisu-style Japanese utterances (short, medium, long) through PyAudio and prints TTFP plus live T2S throughput for each scripted line. Pass `--show-total` to also print audio duration, wall time, and RTF.
Playback demos default to the low-latency live profile (`--top-p 1 --speed 1.1 --sample-steps 4 --how-to-cut 按标点符号切`) with queue-based playback, chunk merging, short fades, and padding to reduce chunk-boundary artifacts. To inspect whole-utterance T2S throughput without punctuation splitting, pass `--how-to-cut 不切`.

For recording an interactive demo, load the voice once and type any Japanese text:

```bash
python examples/live_talk.py --gpt-sovits-home /path/to/GPT-SoVITS-v3lora
```

## Acknowledgements

Aqua-TTS was inspired by [GENIE-TTS](https://github.com/High-Logic/Genie-TTS), which demonstrated that a focused, self-contained inference runtime could meaningfully close the latency gap in GPT-SoVITS. That framing — optimise the runtime, not the model — shaped the direction of this project.

## License

MIT — see [LICENSE](https://github.com/Lucas1479/Aqua-TTS/blob/main/LICENSE).

Third-party code:
- **GPT-SoVITS**: supplied as a separate upstream checkout and consumed under its own MIT license; Aqua does not redistribute its T2S model source.
- **NVIDIA BigVGAN**: CUDA kernel sources under Apache 2.0 — see [NOTICE](https://github.com/Lucas1479/Aqua-TTS/blob/main/NOTICE).
- **alias-free-torch**: `aquatts/bigvgan/torch/` adapted under Apache 2.0 — see [NOTICE](https://github.com/Lucas1479/Aqua-TTS/blob/main/NOTICE).

## Development

```bash
git clone https://github.com/Lucas1479/Aqua-TTS.git
cd Aqua-TTS
pip install -e ".[runtime]"
pip install -r requirements-dev.txt

export GPT_SOVITS_HOME=/path/to/GPT-SoVITS
python -m pytest tests/ -v
```

See [CONTRIBUTING.md](https://github.com/Lucas1479/Aqua-TTS/blob/main/CONTRIBUTING.md) for contribution guidelines and [CHANGELOG.md](https://github.com/Lucas1479/Aqua-TTS/blob/main/CHANGELOG.md) for release history.
Release maintainers should follow [PUBLISHING.md](https://github.com/Lucas1479/Aqua-TTS/blob/main/PUBLISHING.md).
