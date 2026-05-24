# Spectralis-TTS

**GPU-optimized inference wrapper for GPT-SoVITS v3 with static KV cache and CUDA Graph acceleration.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-brightgreen)](https://developer.nvidia.com/cuda-toolkit)

Spectralis-TTS is a drop-in optimization layer for [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) that accelerates the AR Text-to-Semantic (T2S) decoder by **3.5-5x** and reduces time-to-first-packet by **2-4x** through static KV cache buffers and bucketed CUDA Graph capture.

## Highlights

| | Vanilla GPT-SoVITS | Spectralis-TTS |
|---|---|---|
| T2S throughput | ~90-120 it/s | **370-440 it/s** |
| TTFP (short) | 1061ms | **424ms** |
| TTFP (long) | 3598ms | **648ms** |
| KV cache | Dynamic `torch.cat` | **Static `scatter_` buffer** |
| CUDA Graph | Not available | **6 buckets, thread-safe** |
| BigVGAN vocoder | PyTorch JIT | **Pre-compiled CUDA kernel** |
| GPU memory safety | OOM on long texts | **Bounded static buffers** |

*Measured on RTX 4070 Ti SUPER, float16, same model weights.*

## What's Inside

```
spectralis/
├── modeling/
│   ├── t2s_streaming.py    Static KV cache blocks + CUDA Graph capture
│   └── __init__.py
├── bigvgan/
│   ├── cuda/
│   │   ├── load.py         Pre-compiled kernel loader (per-GPU cache)
│   │   ├── activation1d.py Fused anti-alias activation (CUDA + fallback)
│   │   └── *.cpp, *.cu, *.h  NVIDIA BigVGAN CUDA kernel sources
│   └── torch/
│       ├── resample.py     UpSample1d / DownSample1d (pure PyTorch)
│       ├── filter.py       Low-pass filter (Kaiser window)
│       └── act.py          Activation1d (non-fused fallback)
└── inference/
    ├── streaming.py        Audio post-processing + BigVGAN loader
    └── params.py           TTS parameter presets
```

See **[TECHNICAL.md](TECHNICAL.md)** for deep technical documentation.

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ with compatible PyTorch
- GPT-SoVITS v3 installed (this repo vendors the core files)

### Quick install

```bash
git clone https://github.com/SiqiLiOcean/spectralis-tts.git
cd spectralis-tts
pip install -e .
```

For the optional API server:

```bash
pip install -e ".[server]"
```

## Usage

```python
import os
os.environ["ENABLE_CUDA_GRAPH"] = "1"  # Enable CUDA Graph

from local_tts_infer import TTSInferencer

tts = TTSInferencer(
    device="cuda",
    gpt_path="GPT_weights_v3/xxx-e15.ckpt",
    sovits_path="SoVITS_weights_v3/xxx_e2_s174_l32.pth",
)

# Streaming generation
for sr, audio_chunk in tts.infer_stream(
    text="こんにちは、世界！",
    ref_audio_path="reference.wav",
    prompt_text="reference transcript",
    text_language="日文",
    prompt_language="日文",
):
    # audio_chunk is float32 numpy array at `sr` Hz
    pass
```

Or apply the patch manually:

```python
from spectralis.modeling import apply_cuda_graph_patch

t2s_model = ...  # Your Text2SemanticLightningModule
apply_cuda_graph_patch(t2s_model.model)
```

## Benchmarks

```bash
# T2S speed benchmark
python benchmarks/t2s_speed_bench.py --gpt-model GPT_weights_v3/xxx-e15.ckpt

# TTFP benchmark
python benchmarks/spectralis_ttfp.py \
    --gpt-model GPT_weights_v3/xxx-e15.ckpt \
    --sovits-model SoVITS_weights_v3/xxx_e2_s174_l32.pth \
    --ref-audio "reference audio/kurisu_reference2.wav" \
    --ref-text "reference transcript"
```

See [benchmarks/README.md](benchmarks/README.md) for full comparison results.

## Configuration

| Env variable | Default | Description |
|---|---|---|
| `ENABLE_CUDA_GRAPH` | `1` | Enable CUDA Graph replay |
| `ENABLE_CUDA_GRAPH_PRECAPTURE` | `1` | Pre-capture all buckets at startup |
| `CUDA_GRAPH_PRECAPTURE_BUCKETS` | (all) | Comma-separated bucket sizes |

## License

MIT — see [LICENSE](LICENSE).

The BigVGAN CUDA kernel (`spectralis/bigvgan/cuda/*.cpp, *.cu, *.h`) is from NVIDIA BigVGAN, licensed under Apache 2.0 — see [NOTICE](NOTICE).

This project builds on [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS), also MIT licensed.

## Development

```bash
git clone https://github.com/SiqiLiOcean/spectralis-tts.git
cd spectralis-tts
pip install -e ".[server]"
pip install -r requirements-dev.txt
python -m pytest tests/ -v
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines and [CHANGELOG.md](CHANGELOG.md) for release history.
