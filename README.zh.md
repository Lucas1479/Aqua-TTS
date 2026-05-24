<div align="center">

<img src="assets/aqua.png" width="720"/>

<h1>🌊 Aqua-TTS: <a href="https://github.com/RVC-Boss/GPT-SoVITS">GPT-SoVITS</a> GPU 实时推理运行时</h1>

<p>为与 LoRA 角色实时语音对话而生</p>

<p>
  中文 | <a href="README.md">English</a>
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-blue" alt="Python"/>
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License"/>
  <img src="https://img.shields.io/badge/CUDA-11.8%2B-brightgreen" alt="CUDA"/>
</p>

</div>

---

Aqua-TTS 是专为**实时语音对话**设计的 GPU 优化推理运行时——核心场景是与你自己的 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) v3 LoRA 角色进行低延迟流式语音交互。它不替换模型权重，而是替换执行策略：静态 KV 缓存缓冲区、分段 CUDA Graph 捕获/回放以及预编译 BigVGAN CUDA 内核。在 RTX 4070 Ti SUPER 上，T2S 吞吐量可达 **440–470 it/s**，首声播放延迟从 1.0–3.6 s 降至 **~0.26–0.41 s**（测试文本下）——完整对比见 [亮点](#亮点)。

> **定位说明** — Aqua-TTS 是面向 GPT-SoVITS **v3** 的独立运行时，不是补丁插件，也不承诺跟进上游更新。本项目涉及的技术——静态 KV 缓存、分段 CUDA Graph、预编译 BigVGAN 内核——已在 [TECHNICAL.md](TECHNICAL.md) 中详细记录，具备可移植性。如需 v4 支持，`aquatts/modeling/` 和 `aquatts/_vendor/` 是适配的合理起点。

## 亮点

| | GPT-SoVITS（官方） | + CUDA Graph | Aqua-TTS |
|---|---|---|---|
| T2S 吞吐量 | ~80-90 it/s | ~230 it/s | **440-470 it/s** |
| TTFP（短，3 字符） | 1061ms | 1016ms | **262ms** |
| TTFP（中，19 字符） | 1599ms | 1476ms | **318ms** |
| TTFP（长，64 字符） | 3598ms | 2852ms | **409ms** |
| KV 缓存 | 动态 `torch.cat` | 静态 `scatter_` | **静态 `scatter_` 缓冲区** |
| CUDA Graph | 无 | 单个延迟图 | **17 个预捕获图，6 个分段** |
| BigVGAN 声码器 | PyTorch JIT | PyTorch JIT | **预编译 CUDA 内核** |
| KV 缓存分配 | 无界增长 | 无界增长 | **按桶配置有界分配** |

*在 NVIDIA GeForce RTX 4070 Ti SUPER (16 GB)、float16、相同模型权重（xxx-e15.ckpt + xxx_e2_s174_l32.pth）下测量。T2S 以 500 token 为目标测量；TTFP 统计调用方收到第一段可播放音频的时间。对非流式基线，这个时间可能等同于整句/整段生成完成；对 Aqua-TTS，则是 `chunk_size_seconds=0.25` 下的首个流式音频块返回时间。完整方法和消融结果见 [benchmarks/README.md](benchmarks/README.md)。*

## 特性

- **静态 KV 缓存** — 预分配的 scatter 缓冲区，消除每步 `torch.cat` 开销
- **分段 CUDA Graph** — 6 个桶大小共 13 个预捕获图，无预热抖动
- **预编译 BigVGAN** — NVIDIA CUDA 内核从预构建 `.pyd` 自动加载，支持 torch 回退
- **流式 API** — 基于生成器的 `infer_stream()`，首包尽早 yield
- **内置预设** — 快速 / 均衡 / 质量 三种生成预设；完整 / 最小 / 延迟 / 关闭 四种 CUDA Graph 预设
- **角色管理** — 将角色名映射到参考音频和文本提示，支持 JSON 持久化
- **HTTP 服务器** — 轻量 FastAPI 服务，支持流式 TTS 接口、角色管理和健康检查
- **源码安装** — `pip install -e ".[runtime]"` → `from aquatts import TTSInferencer`

## 语言支持

Aqua-TTS 继承 GPT-SoVITS v3 的语言能力。通过 `text_language` / `prompt_language` 参数传入语言代码，参考音频和目标文本可使用不同语言。

| 语言 | 代码 |
|---|---|
| 日语 | `日文` |
| 中文 | `中文` |
| 英语 | `英文` |

## 工作原理

Aqua-TTS 将两个 GPT-SoVITS 文件的修改版本置于 `aquatts/_vendor/` 内，通过 Python 命名空间包（`pkgutil.extend_path`）机制覆盖主仓库版本。当你 `import aquatts` 时，包会自动配置 `sys.path` 使 vendored 文件优先于主 GPT-SoVITS 仓库。设置 `GPT_SOVITS_HOME` 环境变量指向你的 GPT-SoVITS 安装目录即可：

```
aqua-tts/
├── aquatts/                       # 纯 Python 包（pip 可安装）
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

## 推荐配置

| | 入门 | 推荐 |
|---|---|---|
| GPU | RTX 3060 6 GB | RTX 4060 / 4070 8 GB+ |
| CUDA | 11.8+ | 12.x |
| 内存 | 8 GB | 16 GB+ |
| Python | 3.10 | 3.11 / 3.12 |
| 系统 | Windows 10+ | Windows 11 |

Linux CI（Ubuntu）的单元测试全部通过。GPU 相关路径（CUDA Graph、BigVGAN 内核）尚未在 Linux 硬件上测试。macOS 不支持——Aqua-TTS 依赖 CUDA。

> 6 GB 显存显卡建议使用 `cuda_graph_preset="lazy"` 或 `"off"` 以减少预捕获图的显存占用。

## 安装

### 1. 安装 GPT-SoVITS

需要一个可用的 GPT-SoVITS v3 安装。Aqua-TTS 从其中导入模块。

```bash
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
cd GPT-SoVITS
pip install -r requirements.txt
```

Aqua-TTS 已通过 GPT-SoVITS v3（2025-04-01 版本）测试。

### 2. 安装 PyTorch

在安装 Aqua-TTS **之前**，先按你的 CUDA 版本单独安装 PyTorch——该依赖不包含在包内，以允许用户选择对应的 CUDA wheel：

```bash
# 示例：CUDA 12.1 版本 — 其他版本见 https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Aqua-TTS 已在 RTX 4070 Ti SUPER 上使用 PyTorch 2.5.1+cu121 测试通过。

### 3. 安装 Aqua-TTS

```bash
git clone https://github.com/Lucas1479/Aqua-TTS.git
cd Aqua-TTS

# 核心 + 运行时依赖
pip install -e ".[runtime]"

# 或含 HTTP 服务器支持
pip install -e ".[runtime,server]"
```

> PyPI 包计划发布，目前尚未上线。

扩展说明：
- `[runtime]` — soundfile, librosa, peft（`TTSInferencer` 所需）
- `[server]` — FastAPI + uvicorn（自动包含 `[runtime]`）

### 4. 配置 GPT-SoVITS 路径

设置 `GPT_SOVITS_HOME` 环境变量指向你的 GPT-SoVITS 仓库根目录：

```bash
# Windows (PowerShell)
$env:GPT_SOVITS_HOME = "C:\path\to\GPT-SoVITS"

# Linux / macOS
export GPT_SOVITS_HOME=/path/to/GPT-SoVITS
```

### 5. 下载预训练模型

需要从 [Hugging Face](https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x) 下载 BigVGAN v2 预训练权重，直接下载到 GPT-SoVITS 仓库目录内：

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

`models--nvidia--bigvgan_v2_24khz_100band_256x` 是 HuggingFace Hub 的本地缓存目录格式，请严格按照上面的路径放置。

### 6. 准备模型权重

- **GPT 权重 (T2S)**：`Text2SemanticLightningModule` 检查点（如 `s1v3.ckpt` 或自训练的 `xxx-e15.ckpt`）
- **SoVITS 权重（声码器）**：`SynthesizerTrn` 检查点，可选 LoRA（如 `xxx_e2_s174_l32.pth`）
- **参考音频**：3-10 秒的 24 kHz WAV 文件，文本内容已知

## 使用方式

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
from aquatts import apply_preset, list_presets

print(list_presets())  # ["balanced", "fast", "quality"]
params = apply_preset("fast", overrides={"top_k": 5})
```

### HTTP 服务器

启动轻量推理服务：

```bash
python -m aquatts.server \
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
curl -X POST "http://127.0.0.1:8000/tts?text=你好世界&voice=alice" --output - \
  | play -t raw -r 24k -e floating-point -b 32 -c 1 -

# 下载 WAV 文件
curl -X POST "http://127.0.0.1:8000/tts/file?text=你好世界&voice=alice" -o output.wav
```

### 角色管理

管理多个角色/音色，无需每次重复指定路径：

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

在 HTTP 服务器上，传入 `voice` 参数替代 `ref_audio_path`：

```bash
curl -X POST "http://127.0.0.1:8000/tts?text=你好&voice=alice"
```

生产环境建议显式指定注册表路径：

```bash
python -m aquatts.server ... --voice-registry /data/voices.json
# 或
export AQUA_VOICE_JSON=/data/voices.json
```

## 配置

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `GPT_SOVITS_HOME` | *(必填)* | GPT-SoVITS 仓库根目录路径 |
| `AQUA_API_KEY` | *(未设置)* | 服务器所有端点的 Bearer 令牌；未设置则无鉴权 |
| `AQUA_VOICE_JSON` | `./voices.json` | 角色注册表 JSON 文件路径。**建议始终设置**——默认值相对于进程 CWD，目录切换后数据将丢失 |
| `AQUA_SESSION_CACHE_MAX` | `8` | 参考音频 session 最大缓存数量 |
| `ENABLE_CUDA_GRAPH` | `1` | 启用 CUDA Graph 回放 |
| `ENABLE_CUDA_GRAPH_PRECAPTURE` | `1` | 启动时预捕获所有桶图 |
| `TTS_OUTPUT_LANGUAGE` | `日文` | 默认输出语言。非日语用户请改为 `中文` 或 `英文` |
| `TTS_REF_TEXT_JA` | `こんにちは。今日はいい天気ですね。` | 默认日语参考文本 |
| `TTS_REF_TEXT_EN` | *(空)* | 默认英语参考文本 |
| `TTS_STREAM_SYNC_TIMING` | `0` | 启用逐步 CFM 计时（增加 GPU 同步开销） |

## 基准测试

```bash
# T2S 对比（官方 vs 官方+CUDA Graph vs Aqua-TTS）
python benchmarks/t2s_comparison_bench.py --gpt-model GPT_weights_v3/s1v3.ckpt

# TTFP 基准（端到端流式）
python benchmarks/aqua_ttfp.py \
    --gpt-model GPT_weights_v3/s1v3.ckpt \
    --sovits-model SoVITS_weights_v3/your_model.pth \
    --ref-audio "reference audio/ref_audio.wav" \
    --ref-text "参考音频的文本内容"

# BigVGAN 原始内核计时
python benchmarks/bigvgan_raw_bench.py
```

完整对比方法和结果见 [benchmarks/README.md](benchmarks/README.md)。

## 声卡播放 Demo

```bash
pip install -e ".[playback]"
python examples/play_ete.py --gpt-sovits-home /path/to/GPT-SoVITS-v3lora
```

该 demo 会通过 PyAudio 直接播放三句 Kurisu 风格日语文本（短、中、长），并为每句打印首声延迟、实时句子 T2S 吞吐和 RTF。
播放 demo 默认使用更稳的听感参数（`--sample-steps 8 --chunk-size-seconds 0.35`）。如果要切回接近 benchmark 的低延迟设置，可传入 `--sample-steps 4 --chunk-size-seconds 0.25`。

如果要录交互式 demo，可以只加载一次音色，然后输入任意日文文本播放：

```bash
python examples/live_talk.py --gpt-sovits-home /path/to/GPT-SoVITS-v3lora
```

## 致谢

Aqua-TTS 的灵感来源于 [GENIE-TTS](https://github.com/w-okada/genie-tts)——它证明了一个专注、自包含的推理运行时能够切实缩短 GPT-SoVITS 的延迟。"优化运行时而非模型"这一思路，奠定了本项目的方向。

## 许可证

MIT — 详见 [LICENSE](LICENSE)。

第三方代码：
- **GPT-SoVITS**：vendored `aquatts/_vendor/GPT_SoVITS/AR/models/t2s_model.py` 基于 GPT-SoVITS (MIT) — 详见 [NOTICE](NOTICE)。
- **NVIDIA BigVGAN**：CUDA 内核源码基于 Apache 2.0 — 详见 [NOTICE](NOTICE)。
- **alias-free-torch**：`aquatts/bigvgan/torch/` 基于 Apache 2.0 — 详见 [NOTICE](NOTICE)。

## 开发

```bash
git clone https://github.com/Lucas1479/Aqua-TTS.git
cd Aqua-TTS
pip install -e ".[runtime]"
pip install -r requirements-dev.txt

export GPT_SOVITS_HOME=/path/to/GPT-SoVITS
python -m pytest tests/ -v
```

贡献指南见 [CONTRIBUTING.md](CONTRIBUTING.md)，更新日志见 [CHANGELOG.md](CHANGELOG.md)。
