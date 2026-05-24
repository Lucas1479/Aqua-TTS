# Spectralis-TTS

**GPU-optimized inference for GPT-SoVITS v3 — static KV cache + bucketed CUDA Graph + pre-compiled BigVGAN kernel.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-brightgreen)](https://developer.nvidia.com/cuda-toolkit)

Spectralis-TTS is a **GPU-optimized runtime service layer** for [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) v3. It does not replace model weights — it replaces the execution strategy: static KV cache buffers, bucketed CUDA Graph capture/replay, and pre-compiled BigVGAN CUDA kernels. The result is **5.5x faster** T2S decoding and **2-7x lower** time-to-first-packet.

## Highlights

| | GPT-SoVITS (official) | Spectralis-TTS |
|---|---|---|
| T2S throughput | ~80-90 it/s | **440-470 it/s** |
| TTFP (short, 3 chars) | 1061ms | **456ms** |
| TTFP (medium, 19 chars) | 1599ms | **484ms** |
| TTFP (long, 64 chars) | 3598ms | **499ms** |
| KV cache | Dynamic `torch.cat` | **Static `scatter_` buffer** |
| CUDA Graph | Single lazy graph | **13 pre-captured graphs, 6 buckets** |
| BigVGAN vocoder | PyTorch JIT | **Pre-compiled CUDA kernel** |
| GPU memory safety | OOM on long texts | **Bounded static buffers** |

*Measured on NVIDIA GeForce RTX 4070 Ti SUPER (16 GB), float16, same model weights. See [benchmarks/README.md](benchmarks/README.md) for full comparison methodology.*

## How it works

Spectralis-TTS vendors overrides for two GPT-SoVITS files inside `spectralis/_vendor/` using Python namespace packages (`pkgutil.extend_path`). When you `import spectralis`, the package automatically configures `sys.path` so the vendored files take precedence over the main GPT-SoVITS repo. Set `GPT_SOVITS_HOME` to point at your GPT-SoVITS installation — Spectralis handles the rest:

```
spectralis-tts/
├── spectralis/                    # Pure Python package (pip-installable)
│   ├── __init__.py                # sys.path configuration + TTSInferencer re-export
│   ├── inferencer.py              # TTSInferencer — main entry point
│   ├── modeling/
│   │   └── t2s_streaming.py       T2SBlockWithStaticCache, apply_cuda_graph_patch
│   ├── bigvgan/
│   │   ├── cuda/                  Standalone CUDA kernel loader + sources
│   │   └── torch/                 Pure-PyTorch fallback (resample, filter, act)
│   ├── inference/
│   │   ├── streaming.py           Audio post-processing (fade, chunk finalization)
│   │   └── params.py              SoVITS parameter presets
│   └── _vendor/
│       └── GPT_SoVITS/            # Vendored overrides (namespace packages)
│           ├── AR/
│           │   └── models/
│           │       └── t2s_model.py  Static KV + CUDA Graph T2S decoder
│           └── BigVGAN/
│               └── alias_free_activation/cuda/
│                   ├── load.py       Pre-compiled kernel loader (MSVC auto-discovery)
│                   ├── activation1d.py  Fused anti-alias activation
│                   └── *.cpp, *.cu, *.h  NVIDIA BigVGAN CUDA kernel sources
├── benchmarks/                    TTFP, T2S comparison, BigVGAN raw benchmarks
├── examples/                      basic_usage.py, streaming_inference.py
└── tests/                         Unit tests
```

See **[TECHNICAL.md](TECHNICAL.md)** for deep technical documentation.

## Installation

### 1. Install GPT-SoVITS

You need a working GPT-SoVITS v3 installation. Spectralis-TTS imports from it.

```bash
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
cd GPT-SoVITS
pip install -r requirements.txt
```

Spectralis-TTS has been tested against GPT-SoVITS v3 (2025-04-01 release).

### 2. Install Spectralis-TTS

```bash
git clone https://github.com/SiqiLiOcean/spectralis-tts.git
cd spectralis-tts
pip install -e .
```

### 3. Configure GPT-SoVITS path

Set the `GPT_SOVITS_HOME` environment variable to your GPT-SoVITS repo root:

```bash
# Windows (PowerShell)
$env:GPT_SOVITS_HOME = "C:\path\to\GPT-SoVITS"

# Windows (CMD)
set GPT_SOVITS_HOME=C:\path\to\GPT-SoVITS

# Linux / macOS
export GPT_SOVITS_HOME=/path/to/GPT-SoVITS
```

### 4. Download pretrained models

Spectralis needs BigVGAN v2 pretrained weights. Place them inside your GPT-SoVITS repo:

```
GPT_SoVITS/pretrained_models/models--nvidia--bigvgan_v2_24khz_100band_256x/
```

You can download them from [Hugging Face](https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x).

### 5. Prepare model weights

- **GPT weights (T2S)**: A `Text2SemanticLightningModule` checkpoint (e.g., `s1v3.ckpt` or self-trained `xxx-e15.ckpt`).
- **SoVITS weights (vocoder)**: A `SynthesizerTrn` checkpoint with optional LoRA (e.g., `xxx_e2_s174_l32.pth`).
- **Reference audio**: A 3-10 second 24 kHz WAV file with known transcript.

## Usage

```python
import os

# Set GPT_SOVITS_HOME to your GPT-SoVITS repo root before importing spectralis
os.environ["GPT_SOVITS_HOME"] = "/path/to/GPT-SoVITS"
os.environ["ENABLE_CUDA_GRAPH"] = "1"  # Enable CUDA Graph (on by default)

from spectralis import TTSInferencer

tts = TTSInferencer(
    device="cuda",
    gpt_path="GPT_weights_v3/s1v3.ckpt",
    sovits_path="SoVITS_weights_v3/your_model.pth",
)

# Streaming generation — yields (sample_rate, audio_chunk, text) tuples
for sr, chunk, text in tts.infer_stream(
    text="こんにちは、世界！",
    ref_audio_path="reference audio/ref_audio.wav",
    prompt_text="こんにちは。今日はいい天気ですね。",
    text_language="日文",
    prompt_language="日文",
):
    # chunk is float32 numpy array at `sr` Hz
    # text is the text segment being spoken
    pass
```

See `examples/basic_usage.py` and `examples/streaming_inference.py` for runnable examples.

## Configuration

| Env variable | Default | Description |
|---|---|---|
| `ENABLE_CUDA_GRAPH` | `1` | Enable CUDA Graph replay |
| `ENABLE_CUDA_GRAPH_PRECAPTURE` | `1` | Pre-capture all bucket graphs at startup |
| `CUDA_GRAPH_PRECAPTURE_BUCKETS` | (all) | Comma-separated bucket sizes to capture |
| `TTS_STREAM_SYNC_TIMING` | `0` | Enable per-step CFM timing (adds GPU sync overhead) |

## Benchmarks

```bash
# T2S comparison (official vs official CUDA Graph vs Spectralis)
python benchmarks/t2s_comparison_bench.py --gpt-model GPT_weights_v3/s1v3.ckpt

# TTFP benchmark (streaming end-to-end)
python benchmarks/spectralis_ttfp.py \
    --gpt-model GPT_weights_v3/s1v3.ckpt \
    --sovits-model SoVITS_weights_v3/your_model.pth \
    --ref-audio "reference audio/ref_audio.wav" \
    --ref-text "transcript of reference audio"

# BigVGAN raw kernel timing
python benchmarks/bigvgan_raw_bench.py
```

See [benchmarks/README.md](benchmarks/README.md) for full comparison methodology and results.

## License

MIT — see [LICENSE](LICENSE).

Third-party code:
- **GPT-SoVITS**: vendored `spectralis/_vendor/GPT_SoVITS/AR/models/t2s_model.py` is based on GPT-SoVITS (MIT) — see [NOTICE](NOTICE).
- **NVIDIA BigVGAN**: CUDA kernel sources under Apache 2.0 — see [NOTICE](NOTICE).
- **alias-free-torch**: `spectralis/bigvgan/torch/` adapted under Apache 2.0 — see [NOTICE](NOTICE).

## Development

```bash
git clone https://github.com/SiqiLiOcean/spectralis-tts.git
cd spectralis-tts
pip install -e ".[server]"
pip install -r requirements-dev.txt

# Set GPT_SOVITS_HOME for testing
export GPT_SOVITS_HOME=/path/to/GPT-SoVITS  # or setx on Windows

python -m pytest tests/ -v
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines and [CHANGELOG.md](CHANGELOG.md) for release history.
