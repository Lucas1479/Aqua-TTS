<div align="center">

**[English](#english)** | **[中文](#chinese)**

</div>

# Aqua-TTS

**GPU-optimized inference for GPT-SoVITS v3 — static KV cache + bucketed CUDA Graph + pre-compiled BigVGAN kernel.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-brightgreen)](https://developer.nvidia.com/cuda-toolkit)

---

<a id="english"></a>

Spectralis-TTS is a **GPU-optimized runtime service layer** for [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) v3. It does not replace model weights — it replaces the execution strategy: static KV cache buffers, bucketed CUDA Graph capture/replay, and pre-compiled BigVGAN CUDA kernels. The result is **5.5x faster** T2S decoding and **2-7x lower** time-to-first-packet.

[中文版本](#chinese)

## Highlights

| | GPT-SoVITS (official) | + CUDA Graph | Aqua-TTS |
|---|---|---|---|
| T2S throughput | ~80-90 it/s | ~230 it/s | **440-470 it/s** |
| TTFP (short, 3 chars) | 1061ms | 1016ms | **456ms** |
| TTFP (medium, 19 chars) | 1599ms | 1476ms | **484ms** |
| TTFP (long, 64 chars) | 3598ms | 2852ms | **499ms** |
| KV cache | Dynamic `torch.cat` | Static `scatter_` | **Static `scatter_` buffer** |
| CUDA Graph | None | Single lazy graph | **13 pre-captured graphs, 6 buckets** |
| BigVGAN vocoder | PyTorch JIT | PyTorch JIT | **Pre-compiled CUDA kernel** |
| GPU memory safety | OOM on long texts | OOM on long texts | **Bounded static buffers** |

*Measured on NVIDIA GeForce RTX 4070 Ti SUPER (16 GB), float16, same model weights (xxx-e15.ckpt + xxx_e2_s174_l32.pth). T2S measured at 500-token target; TTFP measured on 3 test texts with 5 repeats each (median reported). See [benchmarks/README.md](benchmarks/README.md) for full methodology and ablation results.*

## Features

- **Static KV cache** — pre-allocated scatter buffers eliminate per-step `torch.cat` overhead
- **Bucketed CUDA Graph** — 13 pre-captured graphs across 6 bucket sizes, no warmup jitter
- **Pre-compiled BigVGAN** — NVIDIA CUDA kernel auto-loaded from pre-built `.pyd`, with torch fallback
- **Streaming API** — generator-based `infer_stream()` with bounded GPU memory
- **Built-in presets** — fast / balanced / quality generation presets; full / minimal / lazy / off CUDA Graph presets
- **Voice registry** — map voice names to reference audio + prompt, with JSON persistence
- **HTTP server** — lightweight FastAPI server with streaming TTS endpoint, voice management, and health check
- **pip-installable** — `pip install spectralis-tts` → `from spectralis import TTSInferencer`

## How it works

Spectralis-TTS vendors overrides for two GPT-SoVITS files inside `spectralis/_vendor/` using Python namespace packages (`pkgutil.extend_path`). When you `import spectralis`, the package automatically configures `sys.path` so the vendored files take precedence over the main GPT-SoVITS repo. Set `GPT_SOVITS_HOME` to point at your GPT-SoVITS installation — Spectralis handles the rest:

```
spectralis-tts/
├── spectralis/                    # Pure Python package (pip-installable)
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

## Installation

### 1. Install GPT-SoVITS

You need a working GPT-SoVITS v3 installation. Spectralis-TTS imports from it.

```bash
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
cd GPT-SoVITS
pip install -r requirements.txt
```

Spectralis-TTS has been tested against GPT-SoVITS v3 (2025-04-01 release).

### 2. Install Aqua-TTS

```bash
# Core + runtime dependencies
pip install "spectralis-tts[runtime]"

# Or with HTTP server support (includes runtime + fastapi + uvicorn)
pip install "spectralis-tts[runtime,server]"
```

For development:

```bash
git clone https://github.com/SiqiLiOcean/spectralis-tts.git
cd spectralis-tts
pip install -e ".[runtime,server]"
```

Extras:
- `[runtime]` — soundfile, librosa, peft (needed by `TTSInferencer`)
- `[server]` — FastAPI + uvicorn (includes `[runtime]` automatically)

### 3. Configure GPT-SoVITS path

Set the `GPT_SOVITS_HOME` environment variable to your GPT-SoVITS repo root:

```bash
# Windows (PowerShell)
$env:GPT_SOVITS_HOME = "C:\path\to\GPT-SoVITS"

# Linux / macOS
export GPT_SOVITS_HOME=/path/to/GPT-SoVITS
```

### 4. Download pretrained models

Spectralis needs BigVGAN v2 pretrained weights from [Hugging Face](https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x). Place them inside your GPT-SoVITS repo:

```
GPT_SoVITS/pretrained_models/models--nvidia--bigvgan_v2_24khz_100band_256x/
```

### 5. Prepare model weights

- **GPT weights (T2S)**: A `Text2SemanticLightningModule` checkpoint (e.g., `s1v3.ckpt` or self-trained `xxx-e15.ckpt`).
- **SoVITS weights (vocoder)**: A `SynthesizerTrn` checkpoint with optional LoRA (e.g., `xxx_e2_s174_l32.pth`).
- **Reference audio**: A 3-10 second 24 kHz WAV file with known transcript.

## Usage

### Python API

```python
import os
os.environ["GPT_SOVITS_HOME"] = "/path/to/GPT-SoVITS"
os.environ["ENABLE_CUDA_GRAPH"] = "1"

from spectralis import TTSInferencer

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
from spectralis import apply_preset, list_presets

print(list_presets())  # ["balanced", "fast", "quality"]
params = apply_preset("fast", overrides={"top_k": 5})
```

### HTTP Server

Start a lightweight inference server:

```bash
python -m spectralis.server \
    --gpt-model GPT_weights_v3/s1v3.ckpt \
    --sovits-model SoVITS_weights_v3/model.pth \
    --cuda-graph-preset full \
    --host 127.0.0.1 --port 8000
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
# Pipe to a player that supports raw floats, e.g.:
curl -X POST "http://127.0.0.1:8000/tts?text=Hello&voice=alice" --output - \
  | play -t raw -r 24k -e floating-point -b 32 -c 1 -

# Download WAV file (properly formatted, playable anywhere)
curl -X POST "http://127.0.0.1:8000/tts/file?text=Hello&voice=alice" -o output.wav
```

### Voice Registry

Manage multiple characters/voices without repeating paths:

```python
from spectralis import Voice, VoiceRegistry

registry = VoiceRegistry(json_path="./voices.json")

registry.add(Voice(
    name="alice",
    ref_audio_path="./voices/alice_ref.wav",
    prompt_text="こんにちは。今日はいい天気ですね。",
    prompt_language="日文",
))

# Use voice name in TTS
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

For production deployments, pass an explicit registry path to avoid writing to the process CWD:

```bash
python -m spectralis.server ... --voice-registry /data/voices.json
# or
export SPECTRALIS_VOICE_JSON=/data/voices.json
```

## Configuration

| Env variable | Default | Description |
|---|---|---|
| `GPT_SOVITS_HOME` | *(required)* | Path to GPT-SoVITS repo root |
| `SPECTRALIS_VOICE_JSON` | `./voices.json` | Path to voice registry JSON file |
| `ENABLE_CUDA_GRAPH` | `1` | Enable CUDA Graph replay |
| `ENABLE_CUDA_GRAPH_PRECAPTURE` | `1` | Pre-capture all bucket graphs at startup |
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
pip install -e ".[runtime]"
pip install -r requirements-dev.txt

# Set GPT_SOVITS_HOME for testing
export GPT_SOVITS_HOME=/path/to/GPT-SoVITS

python -m pytest tests/ -v
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines and [CHANGELOG.md](CHANGELOG.md) for release history.

---

<a id="chinese"></a>

# Spectralis-TTS · 中文

**GPT-SoVITS v3 的 GPU 优化推理运行时 — 静态 KV 缓存 + 分段 CUDA Graph + 预编译 BigVGAN 内核。**

[English version](#english)

## 亮点

| | GPT-SoVITS（官方） | + CUDA Graph | Aqua-TTS |
|---|---|---|---|
| T2S 吞吐量 | ~80-90 it/s | ~230 it/s | **440-470 it/s** |
| TTFP（短，3 字符） | 1061ms | 1016ms | **456ms** |
| TTFP（中，19 字符） | 1599ms | 1476ms | **484ms** |
| TTFP（长，64 字符） | 3598ms | 2852ms | **499ms** |
| KV 缓存 | 动态 `torch.cat` | 静态 `scatter_` | **静态 `scatter_` 缓冲区** |
| CUDA Graph | 无 | 单个延迟图 | **13 个预捕获图，6 个分段** |
| BigVGAN 声码器 | PyTorch JIT | PyTorch JIT | **预编译 CUDA 内核** |
| GPU 内存安全 | 长文本时 OOM | 长文本时 OOM | **有界静态缓冲区** |

*在 NVIDIA GeForce RTX 4070 Ti SUPER (16 GB)、float16、相同模型权重（xxx-e15.ckpt + xxx_e2_s174_l32.pth）下测量。T2S 以 500 token 为目标测量；TTFP 在 3 个测试文本上各 5 次重复取中位数。完整方法和消融结果见 [benchmarks/README.md](benchmarks/README.md)。*

## 特性

- **静态 KV 缓存** — 预分配的 scatter 缓冲区，消除每步 `torch.cat` 开销
- **分段 CUDA Graph** — 6 个桶大小共 13 个预捕获图，无预热抖动
- **预编译 BigVGAN** — NVIDIA CUDA 内核从预构建 `.pyd` 自动加载，支持 torch 回退
- **流式 API** — 基于生成器的 `infer_stream()`，GPU 内存有界
- **内置预设** — 快速 / 均衡 / 质量 三种生成预设；完整 / 最小 / 延迟 / 关闭 四种 CUDA Graph 预设
- **角色管理** — 将角色名映射到参考音频和文本提示，支持 JSON 持久化
- **HTTP 服务器** — 轻量 FastAPI 服务，支持流式 TTS 接口、角色管理和健康检查
- **pip 安装** — `pip install spectralis-tts` → `from spectralis import TTSInferencer`

## 工作原理

Spectralis-TTS 将两个 GPT-SoVITS 文件的修改版本置于 `spectralis/_vendor/` 内，通过 Python 命名空间包（`pkgutil.extend_path`）机制覆盖主仓库版本。当你 `import spectralis` 时，包会自动配置 `sys.path` 使 vendored 文件优先于主 GPT-SoVITS 仓库。设置 `GPT_SOVITS_HOME` 环境变量指向你的 GPT-SoVITS 安装目录即可：

```
spectralis-tts/
├── spectralis/                    # 纯 Python 包（pip 可安装）
│   ├── __init__.py                # sys.path 配置 + 延迟导出
│   ├── inferencer.py              # TTSInferencer — 主入口
│   ├── server.py                  # FastAPI HTTP 服务器
│   ├── voice_registry.py          # 角色名 → 音频路径映射
│   ├── modeling/
│   │   └── t2s_streaming.py       # T2SBlockWithStaticCache, CUDA Graph 补丁
│   ├── bigvgan/
│   │   ├── cuda/                  # 独立 CUDA 内核加载器与源码
│   │   └── torch/                 # 纯 PyTorch 回退
│   ├── inference/
│   │   ├── streaming.py           # 音频后处理
│   │   ├── params.py              # SoVITS 参数预设
│   │   └── presets.py             # 命名预设（生成 + CUDA Graph）
│   └── _vendor/
│       └── GPT_SoVITS/            # 覆盖文件（命名空间包）
│           ├── AR/models/
│           │   └── t2s_model.py   # 静态 KV + CUDA Graph T2S 解码器
│           └── BigVGAN/alias_free_activation/cuda/
│               ├── load.py        # 预编译内核加载器
│               ├── activation1d.py  # 融合抗混叠激活
│               └── *.cpp, *.cu, *.h  # NVIDIA BigVGAN CUDA 内核源码
├── benchmarks/                    # TTFP、T2S 对比、BigVGAN 原始基准测试
├── examples/                      # 示例脚本
└── tests/                         # 单元测试
```

详细技术文档见 **[TECHNICAL.md](TECHNICAL.md)**。

## 安装

### 1. 安装 GPT-SoVITS

需要一个可用的 GPT-SoVITS v3 安装。Spectralis-TTS 从其中导入模块。

```bash
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
cd GPT-SoVITS
pip install -r requirements.txt
```

Spectralis-TTS 已通过 GPT-SoVITS v3（2025-04-01 版本）测试。

### 2. 安装 Aqua-TTS

```bash
# 核心 + 运行时依赖
pip install "spectralis-tts[runtime]"

# 或含 HTTP 服务器支持（运行时 + FastAPI + Uvicorn）
pip install "spectralis-tts[runtime,server]"
```

开发模式：

```bash
git clone https://github.com/SiqiLiOcean/spectralis-tts.git
cd spectralis-tts
pip install -e ".[runtime,server]"
```

扩展说明：
- `[runtime]` — soundfile, librosa, peft（`TTSInferencer` 所需）
- `[server]` — FastAPI + uvicorn（自动包含 `[runtime]`）

### 3. 配置 GPT-SoVITS 路径

设置 `GPT_SOVITS_HOME` 环境变量指向你的 GPT-SoVITS 仓库根目录：

```bash
# Windows (PowerShell)
$env:GPT_SOVITS_HOME = "C:\path\to\GPT-SoVITS"

# Linux / macOS
export GPT_SOVITS_HOME=/path/to/GPT-SoVITS
```

### 4. 下载预训练模型

需要从 [Hugging Face](https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x) 下载 BigVGAN v2 预训练权重，放入 GPT-SoVITS 仓库：

```
GPT_SoVITS/pretrained_models/models--nvidia--bigvgan_v2_24khz_100band_256x/
```

### 5. 准备模型权重

- **GPT 权重 (T2S)**：`Text2SemanticLightningModule` 检查点（如 `s1v3.ckpt` 或自训练的 `xxx-e15.ckpt`）
- **SoVITS 权重（声码器）**：`SynthesizerTrn` 检查点，可选 LoRA（如 `xxx_e2_s174_l32.pth`）
- **参考音频**：3-10 秒的 24 kHz WAV 文件，文本内容已知

## 使用方式

### Python API

```python
import os
os.environ["GPT_SOVITS_HOME"] = "/path/to/GPT-SoVITS"
os.environ["ENABLE_CUDA_GRAPH"] = "1"

from spectralis import TTSInferencer

tts = TTSInferencer(
    device="cuda",
    gpt_path="GPT_weights_v3/s1v3.ckpt",
    sovits_path="SoVITS_weights_v3/your_model.pth",
    cuda_graph_preset="full",  # "full"、"minimal"、"lazy" 或 "off"
)

# 流式生成 — 产出 (sample_rate, audio_chunk, text) 元组
for sr, chunk, text in tts.infer_stream(
    text="こんにちは、世界！",
    ref_audio_path="reference audio/ref_audio.wav",
    prompt_text="こんにちは。今日はいい天気ですね。",
    text_language="日文",
    prompt_language="日文",
    preset="fast",  # "fast"、"balanced" 或 "quality"
):
    # chunk 是 float32 numpy 数组，采样率 sr Hz
    # text 是当前正在朗读的文本片段
    pass
```

可运行示例见 `examples/basic_usage.py` 和 `examples/streaming_inference.py`。

### 预设方案

两层预设控制质量和速度的权衡：

**生成预设**（每次请求，通过 `infer_stream(preset=...)` 指定）：

| 预设 | 速度 | 采样步数 | Top-K | 温度 |
|------|------|---------|-------|------|
| `fast` | 1.3x | 4 | 3 | 0.6 |
| `balanced` | 1.1x | 4 | 5 | 0.6 |
| `quality` | 1.0x | 16 | 8 | 0.8 |

**CUDA Graph 预设**（系统级别，通过 `TTSInferencer(cuda_graph_preset=...)` 指定）：

| 预设 | 预捕获 | 桶大小 | 说明 |
|------|--------|--------|------|
| `full` | 是 | 128, 256, 448, 512, 768, 1024 | 覆盖全部长度 |
| `minimal` | 是 | 256, 512, 1024 | 仅常见长度 |
| `lazy` | 否 | 即时生成 | 较低内存，TTFP 较慢 |
| `off` | 禁用 | 无 | 仅静态 KV，无图 |

```python
from spectralis import apply_preset, list_presets

print(list_presets())  # ["balanced", "fast", "quality"]
params = apply_preset("fast", overrides={"top_k": 5})
```

### HTTP 服务器

启动轻量推理服务：

```bash
python -m spectralis.server \
    --gpt-model GPT_weights_v3/s1v3.ckpt \
    --sovits-model SoVITS_weights_v3/model.pth \
    --cuda-graph-preset full \
    --host 127.0.0.1 --port 8000
```

接口：

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 模型加载状态 |
| `GET` | `/presets` | 列出可用预设 |
| `POST` | `/tts` | 流式 TTS（float32 PCM 分块） |
| `POST` | `/tts/file` | 一次性 TTS（可下载 .wav 文件） |
| `GET` | `/voices` | 列出已注册角色 |
| `POST` | `/voices/add` | 注册新角色 |
| `DELETE` | `/voices/{name}` | 移除已注册角色 |

```bash
# 流式 TTS — 返回 raw float32 PCM（24kHz, 单声道, little-endian）
# 用 SoX 播放原始浮点音频流：
curl -X POST "http://127.0.0.1:8000/tts?text=你好世界&voice=alice" --output - \
  | play -t raw -r 24k -e floating-point -b 32 -c 1 -

# 下载 WAV 文件（格式完整，任意播放器可用）
curl -X POST "http://127.0.0.1:8000/tts/file?text=你好世界&voice=alice" -o output.wav
```

### 角色管理

管理多个角色/音色，无需每次重复指定路径：

```python
from spectralis import Voice, VoiceRegistry

registry = VoiceRegistry(json_path="./voices.json")

registry.add(Voice(
    name="alice",
    ref_audio_path="./voices/alice_ref.wav",
    prompt_text="こんにちは。今日はいい天気ですね。",
    prompt_language="日文",
))

# 在 TTS 中使用角色名
voice = registry.get("alice")
for sr, chunk, text in tts.infer_stream(
    text="Hello world",
    ref_audio_path=voice.ref_audio_path,
    prompt_text=voice.prompt_text,
    prompt_language=voice.prompt_language,
):
    pass
```

在 HTTP 服务器上，传入 `voice` 参数替代 `ref_audio_path`：

```bash
curl -X POST "http://127.0.0.1:8000/tts?text=你好&voice=alice"
```

生产环境建议显式指定注册表路径，避免写入进程当前目录：

```bash
python -m spectralis.server ... --voice-registry /data/voices.json
# 或
export SPECTRALIS_VOICE_JSON=/data/voices.json
```

## 配置

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `GPT_SOVITS_HOME` | *(必填)* | GPT-SoVITS 仓库根目录路径 |
| `SPECTRALIS_VOICE_JSON` | `./voices.json` | 角色注册表 JSON 文件路径 |
| `ENABLE_CUDA_GRAPH` | `1` | 启用 CUDA Graph 回放 |
| `ENABLE_CUDA_GRAPH_PRECAPTURE` | `1` | 启动时预捕获所有桶图 |
| `TTS_STREAM_SYNC_TIMING` | `0` | 启用逐步 CFM 计时（增加 GPU 同步开销） |

## 基准测试

```bash
# T2S 对比（官方 vs 官方 CUDA Graph vs Spectralis）
python benchmarks/t2s_comparison_bench.py --gpt-model GPT_weights_v3/s1v3.ckpt

# TTFP 基准（端到端流式）
python benchmarks/spectralis_ttfp.py \
    --gpt-model GPT_weights_v3/s1v3.ckpt \
    --sovits-model SoVITS_weights_v3/your_model.pth \
    --ref-audio "reference audio/ref_audio.wav" \
    --ref-text "参考音频的文本内容"

# BigVGAN 原始内核计时
python benchmarks/bigvgan_raw_bench.py
```

完整对比方法和结果见 [benchmarks/README.md](benchmarks/README.md)。

## 许可证

MIT — 详见 [LICENSE](LICENSE)。

第三方代码：
- **GPT-SoVITS**：vendored `spectralis/_vendor/GPT_SoVITS/AR/models/t2s_model.py` 基于 GPT-SoVITS (MIT) — 详见 [NOTICE](NOTICE)。
- **NVIDIA BigVGAN**：CUDA 内核源码基于 Apache 2.0 — 详见 [NOTICE](NOTICE)。
- **alias-free-torch**：`spectralis/bigvgan/torch/` 基于 Apache 2.0 — 详见 [NOTICE](NOTICE)。

## 开发

```bash
git clone https://github.com/SiqiLiOcean/spectralis-tts.git
cd spectralis-tts
pip install -e ".[runtime]"
pip install -r requirements-dev.txt

# 设置 GPT_SOVITS_HOME 用于测试
export GPT_SOVITS_HOME=/path/to/GPT-SoVITS

python -m pytest tests/ -v
```

贡献指南见 [CONTRIBUTING.md](CONTRIBUTING.md)，更新日志见 [CHANGELOG.md](CHANGELOG.md)。
