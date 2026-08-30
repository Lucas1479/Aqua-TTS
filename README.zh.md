<div align="center">

<img src="https://raw.githubusercontent.com/Lucas1479/Aqua-TTS/main/assets/aqua.png" width="720"/>

<h1>🌊 Aqua-TTS: <a href="https://github.com/RVC-Boss/GPT-SoVITS">GPT-SoVITS</a> GPU 实时推理运行时</h1>

<p>为与 LoRA 角色实时语音对话而生</p>

<p>
  中文 | <a href="https://github.com/Lucas1479/Aqua-TTS/blob/main/README.md">English</a>
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-blue" alt="Python"/>
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License"/>
  <img src="https://img.shields.io/badge/CUDA-11.8%2B-brightgreen" alt="CUDA"/>
</p>

</div>

---

Aqua-TTS 是专为**实时语音对话**设计的 GPU 优化推理运行时——核心场景是与你自己的 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) v3 LoRA 角色进行低延迟流式语音交互。它不替换模型权重，而是替换执行策略：静态 KV 缓存缓冲区、分段 CUDA Graph 捕获/回放以及预编译 BigVGAN CUDA 内核。在 RTX 4070 Ti SUPER 上，当前确定性基准在对话常用的 448/512 bucket 达到 **490–519 同步 it/s**；可选 FlashAttention2 在相同形状达到 **568–627 it/s**。测试环境下模型侧首音延迟通常约 **0.26–0.41 s**——完整对比见 [亮点](#亮点)。

## 亮点

<sub>**延迟定义：** TTFP 基准 = 预热缓存下模型侧首音延迟（下表）。端到端首音 = 完整管线含音频缓冲和播放启动耗时，实际通常 **0.4–0.7 s**。冷启动 = 初始化 + 模型加载 + 首次推理，主要被 BigVGAN CUDA 内核编译占据（首次约 2 分钟，之后缓存）。</sub>

| | 当前上游 | Aqua-TTS 默认 | Aqua + FA2 `valid`（可选） |
|---|---:|---:|---:|
| T2S 短形状 / bucket 448 | 145.5 it/s | **490.0 it/s** | **568.5 it/s** |
| T2S 对话形状 / bucket 512 | 153.5 it/s | **519.3 it/s** | **627.4 it/s** |
| T2S 长形状 / bucket 768 | 156.0 it/s | **476.9 it/s** | **644.9 it/s** |
| KV 缓存 | 动态 `torch.cat` | **静态 `scatter_` 缓冲区** | **按有效长度读取的 FA2 KV 缓存** |
| CUDA Graph | 无 | **分桶预捕获** | **分桶预捕获** |

*T2S 基准环境：NVIDIA GeForce RTX 4070 Ti SUPER (16 GB)、PyTorch 2.5.1+cu124、fp16、上游 `08d627c`；每个形状预热 15 次并进行 7 次 CUDA 同步测量，报告中位数。FlashAttention2 默认仍关闭。绝对吞吐会受 Windows GPU P-state 影响；cold 数据、每轮范围、复现命令和独立 TTFP 方法见 [benchmarks/README.md](https://github.com/Lucas1479/Aqua-TTS/blob/main/benchmarks/README.md)。*

https://github.com/user-attachments/assets/581cef5f-f8ce-4570-81ae-a6c092698223

## 特性

- **静态 KV 缓存** — 预分配的 scatter 缓冲区，消除每步 `torch.cat` 开销
- **分段 CUDA Graph** — 6 个桶大小共 13 个预捕获图，无预热抖动
- **预编译 BigVGAN** — NVIDIA CUDA 内核从预构建 `.pyd` 自动加载，支持 torch 回退
- **可选 FlashAttention2 KV 缓存** — 实验性的 `flash_attn_with_kvcache` T2S 路径，默认关闭
- **流式 API** — 基于生成器的 `infer_stream()`，首包尽早 yield
- **内置预设** — 快速 / 均衡 / 质量 三种生成预设；完整 / 最小 / 延迟 / 关闭 四种 CUDA Graph 预设
- **角色管理** — 将角色名映射到参考音频和文本提示，支持 JSON 持久化
- **HTTP 服务器** — 轻量 FastAPI 服务，支持流式 TTS 接口、角色管理和健康检查
- **PyPI 安装** — `pip install "aqua-tts[runtime]"` → `from aquatts import TTSInferencer`

> **定位说明** — Aqua-TTS 是上游 GPT-SoVITS **v3** 的优化层。它加载所选上游的 `Text2SemanticDecoder`，验证兼容契约，并且只替换直接 `infer_panel()` 路径；上游 batching/streaming 入口保持原样。不兼容的上游变更会明确失败。当前不支持 GPT-SoVITS v4。

## 语言支持

Aqua-TTS 继承 GPT-SoVITS v3 的语言能力。通过 `text_language` / `prompt_language` 参数传入语言代码，参考音频和目标文本可使用不同语言。

| 语言 | 代码 |
|---|---|
| 日语 | `日文` |
| 中文 | `中文` |
| 英语 | `英文` |

## 工作原理

Aqua-TTS 从 `GPT_SOVITS_HOME` 指定的上游 checkout 加载模型定义和文本/运行时模块，并在 checkpoint 加载后原位应用静态 KV/CUDA Graph 优化。`_vendor/` 只保留 BigVGAN CUDA 激活到 Aqua 统一加载器的命名空间桥接，不再包含分叉的 `t2s_model.py`：

```
aqua-tts/
├── aquatts/                       # 纯 Python 包（pip 可安装）
│   ├── __init__.py                # sys.path 配置 + 延迟导出
│   ├── upstream.py                # 上游 checkout 校验与路由
│   ├── inferencer.py              # TTSInferencer — 主入口
│   ├── server.py                  # FastAPI HTTP 服务器
│   ├── voice_registry.py          # 角色名 → 音频路径映射
│   ├── modeling/
│   │   ├── t2s_streaming.py       # T2SBlockWithStaticCache, CUDA Graph 补丁
│   │   └── t2s_flash_attn.py      # 可选 FlashAttention2 KV-cache 补丁
│   ├── bigvgan/
│   │   ├── cuda/                  # 独立 CUDA 内核加载器与源码
│   │   └── torch/                 # 纯 PyTorch 回退
│   ├── inference/
│   │   ├── streaming.py           # 音频后处理
│   │   ├── params.py              # SoVITS 参数预设
│   │   └── presets.py             # 命名预设（生成 + CUDA Graph）
│   └── _vendor/
│       └── GPT_SoVITS/            # 命名空间桥接，不含模型分叉
│           └── BigVGAN/alias_free_activation/cuda/  # Aqua 加载器薄适配
├── benchmarks/                    # TTFP、T2S 对比、BigVGAN 原始基准测试
├── examples/                      # 示例脚本
└── tests/                         # 单元测试
```

详细技术文档见 **[TECHNICAL.md](https://github.com/Lucas1479/Aqua-TTS/blob/main/TECHNICAL.md)**。

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
# 从 PyPI 安装核心 + 运行时依赖
pip install "aqua-tts[runtime]"

# 或含 HTTP 服务器支持
pip install "aqua-tts[server]"

# 或含本地播放支持
pip install "aqua-tts[playback]"
```

如需开发版或尚未发布的修改，可从源码安装：

```bash
git clone https://github.com/Lucas1479/Aqua-TTS.git
cd Aqua-TTS
pip install -e ".[runtime]"
```

扩展说明：
- `[runtime]` — soundfile, librosa, peft（`TTSInferencer` 所需）
- `[server]` — FastAPI + uvicorn（自动包含 `[runtime]`）
- `[playback]` — PyAudio（自动包含 `[runtime]`）

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

**可选 FlashAttention2 T2S 路径**（实验性，默认关闭）：

Aqua-TTS 可以用 `flash_attn_with_kvcache` 替换 q_len=1 的 T2S 静态 KV attention 步骤。它是 opt-in 路径，因为收益较明显的 `valid` 模式会按真实 KV 长度参与注意力，而不是默认 SDPA 路径里的完整零填充 bucket，因此输出可能和默认路径有轻微差异。

适合在“长首句”或长文本 T2S 吞吐实验中开启：

```bash
export AQUATTS_T2S_FLASH_ATTN=1
export AQUATTS_T2S_FLASH_ATTN_MODE=valid  # valid | bucket
```

也可以从 Python 显式开启：

```python
tts = TTSInferencer(..., use_flash_attn=True, flash_attn_mode="valid")
```

FlashAttention2 仍为可选功能，主要考虑额外依赖的可移植性，以及它需要独立的语义与音频回归覆盖。2026 年 6 月“短句无收益、长句约提升 8%”的结论是在 CUDA Graph replay 逐 token 强制同步仍存在时测得，已不代表当前性能。

取消逐 token replay 同步后的确定性 A/B
（RTX 4070 Ti SUPER，PyTorch 2.5.1+cu124，flash-attn 2.7.0.post2，预热 15 次，每个形状同步测量 7 次）：

| 场景 | Flash 关闭 | Flash 开启 (`valid`) | 方向性收益 |
|---|---:|---:|---:|
| T2S 短形状 / bucket 448 | 490.0 it/s | 568.5 it/s | 约 16% |
| T2S 对话形状 / bucket 512 | 519.3 it/s | 627.4 it/s | 约 21% |
| T2S 长形状 / bucket 768 | 476.9 it/s | 644.9 it/s | 约 35% |

之前两条路径共同承担的设备级同步掩盖了部分 attention 内核差异。现在每个 step 只保留 EOS 所需同步，FA2 按有效 KV 长度读取的优势会直接体现，尤其是 bucket 768。百分比仍与硬件和 P-state 有关；在改变默认值前，应在目标部署 GPU 上复现，并完成语义/音频回归。

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
| `BIGVGAN_CACHE_ROOT` | 包内缓存 | 可选的产品侧/可写 BigVGAN 编译扩展缓存根目录 |
| `AQUA_API_KEY` | *(未设置)* | 服务器所有端点的 Bearer 令牌；未设置则无鉴权 |
| `AQUA_VOICE_JSON` | `./voices.json` | 角色注册表 JSON 文件路径。**建议始终设置**——默认值相对于进程 CWD，目录切换后数据将丢失 |
| `AQUA_SESSION_CACHE_MAX` | `8` | 参考音频 session 最大缓存数量 |
| `ENABLE_CUDA_GRAPH` | `1` | 启用 CUDA Graph 回放 |
| `ENABLE_CUDA_GRAPH_PRECAPTURE` | `1` | 启动时预捕获所有桶图 |
| `AQUATTS_T2S_FLASH_ATTN` | `0` | 启用实验性 FlashAttention2 `flash_attn_with_kvcache` T2S 路径 |
| `AQUATTS_T2S_FLASH_ATTN_MODE` | `valid` | FlashAttention 模式：`valid` 使用真实 KV 长度；`bucket` 保留零填充 bucket 长度 |
| `TTS_OUTPUT_LANGUAGE` | `日文` | 默认输出语言。非日语用户请改为 `中文` 或 `英文` |
| `TTS_REF_TEXT_JA` | `こんにちは。今日はいい天気ですね。` | 默认日语参考文本 |
| `TTS_REF_TEXT_EN` | *(空)* | 默认英语参考文本 |
| `TTS_STREAM_SYNC_TIMING` | `0` | 启用逐步 CFM 计时（增加 GPU 同步开销） |

## 基准测试

```bash
# T2S 对比（官方 vs 官方+CUDA Graph vs Aqua-TTS）
python benchmarks/t2s_comparison_bench.py \
  --upstream-home /path/to/GPT-SoVITS \
  --gpt-model /path/to/xxx-e15.ckpt \
  --flash-ab

# TTFP 基准（端到端流式）
python benchmarks/aqua_ttfp.py \
    --gpt-model GPT_weights_v3/s1v3.ckpt \
    --sovits-model SoVITS_weights_v3/your_model.pth \
    --ref-audio "reference audio/ref_audio.wav" \
    --ref-text "参考音频的文本内容"

# BigVGAN 原始内核计时
python benchmarks/bigvgan_raw_bench.py
```

完整对比方法和结果见 [benchmarks/README.md](https://github.com/Lucas1479/Aqua-TTS/blob/main/benchmarks/README.md)。

## 声卡播放 Demo

```bash
pip install -e ".[playback]"
python examples/play_ete.py --gpt-sovits-home /path/to/GPT-SoVITS-v3lora
```

该 demo 会通过 PyAudio 直接播放三句 Kurisu 风格日语文本（短、中、长），并为 scripted 句子打印 TTFP 和实时句子 T2S 吞吐。传入 `--show-total` 可额外打印音频时长、完整耗时和 RTF。
播放 demo 默认使用低延迟 live profile（`--top-p 1 --speed 1.1 --sample-steps 4 --how-to-cut 按标点符号切`），并通过队列式播放、chunk 合并、短淡入淡出和末尾 padding 来减少块边界感。如果想看不按标点拆分的整句 T2S 吞吐，可传入 `--how-to-cut 不切`。

如果要录交互式 demo，可以只加载一次音色，然后输入任意日文文本播放：

```bash
python examples/live_talk.py --gpt-sovits-home /path/to/GPT-SoVITS-v3lora
```

## 致谢

Aqua-TTS 的灵感来源于 [GENIE-TTS](https://github.com/High-Logic/Genie-TTS)——它证明了一个专注、自包含的推理运行时能够切实缩短 GPT-SoVITS 的延迟。"优化运行时而非模型"这一思路，奠定了本项目的方向。

## 许可证

MIT — 详见 [LICENSE](https://github.com/Lucas1479/Aqua-TTS/blob/main/LICENSE)。

第三方代码：
- **GPT-SoVITS**：以单独的上游 checkout 提供并遵循其 MIT 许可证；Aqua 不再分发其 T2S 模型源码。
- **NVIDIA BigVGAN**：CUDA 内核源码基于 Apache 2.0 — 详见 [NOTICE](https://github.com/Lucas1479/Aqua-TTS/blob/main/NOTICE)。
- **alias-free-torch**：`aquatts/bigvgan/torch/` 基于 Apache 2.0 — 详见 [NOTICE](https://github.com/Lucas1479/Aqua-TTS/blob/main/NOTICE)。

## 开发

```bash
git clone https://github.com/Lucas1479/Aqua-TTS.git
cd Aqua-TTS
pip install -e ".[runtime]"
pip install -r requirements-dev.txt

export GPT_SOVITS_HOME=/path/to/GPT-SoVITS
python -m pytest tests/ -v
```

贡献指南见 [CONTRIBUTING.md](https://github.com/Lucas1479/Aqua-TTS/blob/main/CONTRIBUTING.md)，更新日志见 [CHANGELOG.md](https://github.com/Lucas1479/Aqua-TTS/blob/main/CHANGELOG.md)。
发布维护者请按照 [PUBLISHING.md](https://github.com/Lucas1479/Aqua-TTS/blob/main/PUBLISHING.md) 操作。
