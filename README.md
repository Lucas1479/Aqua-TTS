<div align="center">

<img src="assets/aqua.png" width="720"/>

<h1>🌊 Aqua-TTS: <a href="https://github.com/RVC-Boss/GPT-SoVITS">GPT-SoVITS</a> Real-Time Inference Runtime on GPU</h1>

<p>Built for low-latency voice conversation with your LoRA characters</p>

<p>
  <a href="README.zh.md">中文</a> | English
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-blue" alt="Python"/>
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License"/>
  <img src="https://img.shields.io/badge/CUDA-11.8%2B-brightgreen" alt="CUDA"/>
</p>

</div>

---

Aqua-TTS is a GPU-optimized inference runtime purpose-built for **real-time voice conversation** — specifically, low-latency streaming TTS with your own [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) v3 LoRA character voices. It does not replace model weights — it replaces the execution strategy: static KV cache buffers, bucketed CUDA Graph capture/replay, and pre-compiled BigVGAN CUDA kernels. On an RTX 4070 Ti SUPER, this reaches **440–470 it/s** T2S throughput and reduces TTFP from 1.0–3.6 s to **~0.26–0.41 s** on the tested utterances — see [Highlights](#highlights) for the full comparison.

> **Scope notice** — Aqua-TTS is a self-contained runtime for GPT-SoVITS **v3**. It is not a plugin and does not track upstream changes. The techniques here — static KV cache, bucketed CUDA Graph, pre-compiled BigVGAN kernel — are documented in [TECHNICAL.md](TECHNICAL.md) and designed to be portable. If you need v4 support, `aquatts/modeling/` and `aquatts/_vendor/` are the right starting points for adaptation.

## Highlights

| | GPT-SoVITS (official) | + CUDA Graph | Aqua-TTS |
|---|---|---|---|
| T2S throughput | ~80-90 it/s | ~230 it/s | **440-470 it/s** |
| TTFP (short, 3 chars) | 1061ms | 1016ms | **262ms** |
| TTFP (medium, 19 chars) | 1599ms | 1476ms | **318ms** |
| TTFP (long, 64 chars) | 3598ms | 2852ms | **409ms** |
| KV cache | Dynamic `torch.cat` | Static `scatter_` | **Static `scatter_` buffer** |
| CUDA Graph | None | Single lazy graph | **17 pre-captured graphs, 6 buckets** |
| BigVGAN vocoder | PyTorch JIT | PyTorch JIT | **Pre-compiled CUDA kernel** |
| KV-cache allocation | Unbounded growth | Unbounded growth | **Bounded per bucket config** |

*Measured on NVIDIA GeForce RTX 4070 Ti SUPER (16 GB), float16, same model weights (xxx-e15.ckpt + xxx_e2_s174_l32.pth). T2S measured at 500-token target; TTFP is time to the first playable audio returned to the caller. For non-streaming baselines this may coincide with full utterance completion; for Aqua-TTS it is the first streaming audio chunk with `chunk_size_seconds=0.25`. See [benchmarks/README.md](benchmarks/README.md) for full methodology and ablation results.*

## Features

- **Static KV cache** — pre-allocated scatter buffers eliminate per-step `torch.cat` overhead
- **Bucketed CUDA Graph** — 13 pre-captured graphs across 6 bucket sizes, no warmup jitter
- **Pre-compiled BigVGAN** — NVIDIA CUDA kernel auto-loaded from pre-built `.pyd`, with torch fallback
- **Streaming API** — generator-based `infer_stream()` with early first-audio yield
- **Built-in presets** — fast / balanced / quality generation presets; full / minimal / lazy / off CUDA Graph presets
- **Voice registry** — map voice names to reference audio + prompt, with JSON persistence
- **HTTP server** — lightweight FastAPI server with streaming TTS endpoint, voice management, and health check
- **Source install** — `pip install -e ".[runtime]"` → `from aquatts import TTSInferencer`

## Supported Languages

Aqua-TTS inherits GPT-SoVITS v3's language support. Pass the code to `text_language` / `prompt_language` — reference audio and target text can use different languages.

| Language | Code |
|---|---|
| Japanese | `日文` |
| Chinese | `中文` |
| English | `英文` |

## How it works

Aqua-TTS vendors overrides for two GPT-SoVITS files inside `aquatts/_vendor/` using Python namespace packages (`pkgutil.extend_path`). When you `import aquatts`, the package automatically configures `sys.path` so the vendored files take precedence over the main GPT-SoVITS repo. Set `GPT_SOVITS_HOME` to point at your GPT-SoVITS installation — Aqua-TTS handles the rest:

```
aqua-tts/
├── aquatts/                       # Pure Python package (pip-installable)
│   ├── __init__.py                # sys.path configuration + lazy exports
│   ├── inferencer.py              # TTSInferencer — main entry point
│   ├── server.py                  # FastAPI HTTP server
│   ├── voice_registry.py          # Voice name → audio path mapping
│   ├── modeling/
│   │   └── t2s_streaming.py       # T2SBlockWithStaticCache, CUDA Graph patch
│   ├── bigvgan/
│   │   ├── cuda/                  # Standalone CUDA kernel loader + sources
│   │   └── torch/                 # Pure-PyTorch fallback (resample, filter, act)
│   ├── inference/
│   │   ├── streaming.py           # Audio post-processing (fade, chunk finalization)
│   │   ├── params.py              # SoVITS parameter presets
│   │   └── presets.py             # Named presets (generation + CUDA Graph)
│   └── _vendor/
│       └── GPT_SoVITS/            # Vendored overrides (namespace packages)
│           ├── AR/models/
│           │   └── t2s_model.py   # Static KV + CUDA Graph T2S decoder
│           └── BigVGAN/alias_free_activation/cuda/
│               ├── load.py        # Pre-compiled kernel loader (MSVC auto-discovery)
│               ├── activation1d.py  # Fused anti-alias activation
│               └── *.cpp, *.cu, *.h  # NVIDIA BigVGAN CUDA kernel sources
├── benchmarks/                    # TTFP, T2S comparison, BigVGAN raw benchmarks
├── examples/                      # basic_usage.py, streaming_inference.py
└── tests/                         # Unit tests
```

See **[TECHNICAL.md](TECHNICAL.md)** for deep technical documentation.

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
git clone https://github.com/Lucas1479/Aqua-TTS.git
cd Aqua-TTS

# Core + runtime dependencies
pip install -e ".[runtime]"

# Or with HTTP server support
pip install -e ".[runtime,server]"
```

> PyPI package is planned but not yet published.

Extras:
- `[runtime]` — soundfile, librosa, peft (needed by `TTSInferencer`)
- `[server]` — FastAPI + uvicorn (includes `[runtime]` automatically)

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

## Configuration

| Env variable | Default | Description |
|---|---|---|
| `GPT_SOVITS_HOME` | *(required)* | Path to GPT-SoVITS repo root |
| `AQUA_API_KEY` | *(unset)* | Bearer token for all server endpoints; unset = no auth |
| `AQUA_VOICE_JSON` | `./voices.json` | Path to voice registry JSON file. **Always set this** — default is relative to process CWD and will be lost on directory change |
| `AQUA_SESSION_CACHE_MAX` | `8` | Max number of cached reference audio sessions |
| `ENABLE_CUDA_GRAPH` | `1` | Enable CUDA Graph replay |
| `ENABLE_CUDA_GRAPH_PRECAPTURE` | `1` | Pre-capture all bucket graphs at startup |
| `TTS_OUTPUT_LANGUAGE` | `日文` | Default output language. Change to `中文` or `英文` if not using Japanese |
| `TTS_REF_TEXT_JA` | `こんにちは。今日はいい天気ですね。` | Default Japanese reference text |
| `TTS_REF_TEXT_EN` | *(empty)* | Default English reference text |
| `TTS_STREAM_SYNC_TIMING` | `0` | Enable per-step CFM timing (adds GPU sync overhead) |

## Benchmarks

```bash
# T2S comparison (official vs official+CUDA Graph vs Aqua-TTS)
# Replace with your own GPT checkpoint (s1v3.ckpt or a self-trained .ckpt)
python benchmarks/t2s_comparison_bench.py --gpt-model GPT_weights_v3/xxx-e15.ckpt

# TTFP benchmark (streaming end-to-end)
python benchmarks/aqua_ttfp.py \
    --gpt-model GPT_weights_v3/xxx-e15.ckpt \
    --sovits-model SoVITS_weights_v3/xxx_e2_s174_l32.pth \
    --ref-audio "reference audio/ref_audio.wav" \
    --ref-text "transcript of reference audio"

# BigVGAN raw kernel timing
python benchmarks/bigvgan_raw_bench.py
```

See [benchmarks/README.md](benchmarks/README.md) for full methodology and results.

## Speaker Demo

```bash
pip install -e ".[playback]"
python examples/play_ete.py --gpt-sovits-home /path/to/GPT-SoVITS-v3lora
```

The demo plays three Kurisu-style Japanese utterances (short, medium, long) through PyAudio and prints first-audio latency and live T2S throughput for each line. Pass `--show-total` to also print audio duration, wall time, and RTF.
Playback demos use queue-based playback with chunk merging, short fades, and padding to reduce chunk-boundary artifacts. For benchmark-like latency settings, pass `--chunk-size-seconds 0.25`.

For recording an interactive demo, load the voice once and type any Japanese text:

```bash
python examples/live_talk.py --gpt-sovits-home /path/to/GPT-SoVITS-v3lora
```

## Acknowledgements

Aqua-TTS was inspired by [GENIE-TTS](https://github.com/w-okada/genie-tts), which demonstrated that a focused, self-contained inference runtime could meaningfully close the latency gap in GPT-SoVITS. That framing — optimise the runtime, not the model — shaped the direction of this project.

## License

MIT — see [LICENSE](LICENSE).

Third-party code:
- **GPT-SoVITS**: vendored `aquatts/_vendor/GPT_SoVITS/AR/models/t2s_model.py` is based on GPT-SoVITS (MIT) — see [NOTICE](NOTICE).
- **NVIDIA BigVGAN**: CUDA kernel sources under Apache 2.0 — see [NOTICE](NOTICE).
- **alias-free-torch**: `aquatts/bigvgan/torch/` adapted under Apache 2.0 — see [NOTICE](NOTICE).

## Development

```bash
git clone https://github.com/Lucas1479/Aqua-TTS.git
cd Aqua-TTS
pip install -e ".[runtime]"
pip install -r requirements-dev.txt

export GPT_SOVITS_HOME=/path/to/GPT-SoVITS
python -m pytest tests/ -v
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines and [CHANGELOG.md](CHANGELOG.md) for release history.
