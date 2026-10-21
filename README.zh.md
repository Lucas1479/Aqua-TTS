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

Aqua-TTS 是面向 GPT-SoVITS 角色实时语音对话的 GPU 推理运行时，结合静态 KV 缓存、分桶 CUDA Graph、可选 FlashAttention2 和缓存的 BigVGAN CUDA 内核。2026-10-21 的 Aqua 专项复测中，修复后 USB4 外接 RTX 4070 Ti SUPER 的 FA2 吞吐为 **619–739 同步 it/s**，SDPA 为 **461–504 it/s**。Laptop 的旧测量单列于下文，尚未复测本次修复。V3 在 FA2 路径下的短／中／长输入预热首个原始 PCM（裁切前）中位数为 **213 / 256 / 307 ms**。这些模型侧数字不包含声卡播放，完整协议与负载身份见下文。

**新增 v2 系列支持：** Aqua-TTS 现已支持 **v2、v2Pro、v2ProPlus** 权重，
包括 Pro／Plus 的说话人编码器。下方提供 [v2 系列性能基准](#新增v2-系列性能基准)
和[权重配置说明](#v2v2-pro-与-v2-pro-plus-权重)。当前 V3 与 RTX 4070 Laptop 测量数据见亮点部分。

## 亮点

**测量日期：2026-10-21，仅运行 Aqua 执行方式。** TTFP 延续历史定义：
**裁切／缓存生成静音之前，第一个原始 PCM 块可用的时间**。
不含模型加载、声卡缓冲和播放。仅基准进程绕过起音处理，正常产品推理继续启用静音裁切。

| 指标 | Aqua Graph + SDPA | Aqua Graph + FA2 `valid` |
|---|---|---|
| T2S 短 / bucket 448 | 504.3 it/s | 618.7 it/s |
| T2S 对话 / bucket 512 | 488.8 it/s | 687.5 it/s |
| T2S 长 / bucket 768 | 460.8 it/s | 738.5 it/s |
| V3 TTFP 短 (3) | 241.9 ms | 212.8 ms |
| V3 TTFP 中 (19) | 281.5 ms | 256.5 ms |
| V3 TTFP 长 (64) | 361.2 ms | 307.2 ms |

环境为 Windows、Python 3.12.10、Torch 2.6.0+cu124、fp16、上游 `08d627c`；
保持默认 `bucket` KV 布局，关闭 CFM unpadded。T2S 沿用 51／123／180 步固定形状，
先预热 15 次，再对每种形状同步测量 7 次。V3 与 v2 系列延迟复用十月实际使用的 `model_latency.py`：
每条文本预热 5 次、固定种子测量 20 次、单独诊断 1 次。仅基准进程绕过起音处理。
实际运行环境为 Torch 2.6.0+cu124、FlashAttention2 2.7.4.post1。
V3 确认使用 BigVGAN CUDA 内核；首页 FA2 数据均在捕获前启用禁止回退检查。

| 机制 | 上游结构 | Aqua Graph + SDPA | Aqua Graph + FA2 |
|---|---|---|---|
| 模型定义 | 直接使用上游模块 | **校验上游 + 内存覆盖层** | **校验上游 + 内存覆盖层** |
| Decode attention | 原生 PyTorch + 动态 KV | 静态 bucket 上的 SDPA | **按真实 KV 长度读取的 FA2** |
| KV 缓存写入 | 每 token `torch.cat` | **原位 `scatter_`** | **FA2 KV-cache update** |
| KV 分配 | 每 token 增长 | **按 bucket 预分配并限界** | **按 bucket 预分配并限界** |
| CUDA Graph | 标准入口无 | **每个准备桶一图，最多 6 图 + 惰性捕获** | **每个准备桶一图，最多 6 图 + 惰性捕获** |
| Replay 同步 | eager launch | **CUDA stream 保序；无逐 token 设备同步** | **CUDA stream 保序；无逐 token 设备同步** |
| EOS 主机读取 | 上游条件判断 | **前 11 步跳过；随后合并一次读取** | **前 11 步跳过；随后合并一次读取** |
| 语义历史 | 每 token 拼接 | **请求内复用缓冲，仅暴露有效前缀** | **请求内复用缓冲，仅暴露有效前缀** |
| Graph 并发 | 不适用 | **按桶或 nested 共享根持有完整 AR 租约** | **按桶或 nested 共享根持有完整 AR 租约** |
| 失败回退 | 动态 decoder | **Graph → 静态 KV → 动态** | **FA2 → SDPA；Graph → 静态 KV → 动态** |
| BigVGAN 激活 | 运行时扩展/JIT 路径 | **ABI 缓存 CUDA → 纯 PyTorch 回退** | **ABI 缓存 CUDA → 纯 PyTorch 回退** |
| 流式契约 | 上游原生 generator | **保留 generator；仅补丁直接 `infer_panel()`** | **保留 generator；仅补丁直接 `infer_panel()`** |

Ti SUPER 经 USB4 连接并承担 Windows 桌面显示，桌面活动和主机同步会影响结果。
请在同一张卡内比较执行方式，不用跨卡数字给显卡性能排名。旧 V3 报告缺少文件哈希，
同名权重不能证明历史负载逐字节相同。[本轮报告](benchmarks/results/latency-fix-20261021.md)与[全部样本](benchmarks/results/latency-fix-20261021.json)
记录当前哈希、范围、p95、总耗时／RTF 和 GPU 遥测。
[旧上游对比](benchmarks/results/4070ti-super-win-py312-torch251-cu124-upstream-overlay.md)保留归档；
本轮没有重跑官方列，也没有把旧官方数字混入新表。

### 8 GB 移动显卡验证

下表保留修复前 `ad91a3d` 的 Laptop 测量，本次未重跑：系统目前只枚举 Ti SUPER。
当时 Laptop 不承担显示输出，使用 Torch 2.6/cu124、同一固定负载、15 次预热和 7 次正式测量。“捕获后首次 AR”不包含加载模型和准备 Graph。

| GPU | 捕获后首次 AR | Short / 448 | Conversation / 512 | Long / 768 |
|---|---|---|---|---|
| RTX 4070 Laptop (8 GB) | 276.7 it/s | 435.4 it/s | 501.9 it/s | 501.6 it/s |

这些是独立 T2S 吞吐。完整管线延迟还取决于声学模型、参考输入和输出分块策略。

https://github.com/user-attachments/assets/581cef5f-f8ce-4570-81ae-a6c092698223

## 特性

- **静态 KV 缓存** — 预分配的 scatter 缓冲区，消除每步 `torch.cat` 开销
- **语义稳定性保护** — 在声码器前拒绝重复 token 塌缩，并进行一次有界重试
- **分桶 CUDA Graph** — 每个准备桶一图，支持六个配置容量、可选 nested KV 存储及惰性捕获
- **BigVGAN CUDA 扩展缓存** — 按 GPU/Python/Torch/CUDA ABI 缓存 NVIDIA 内核，支持 torch 回退
- **自适应 FlashAttention2 KV 缓存** — 自动优先 `flash_attn_with_kvcache` `valid` 模式，并可回退 SDPA
- **流式 API** — 基于生成器的 `infer_stream()`，首包尽早 yield
- **内置预设** — 快速 / 均衡 / 质量 三种生成预设；完整 / 最小 / 延迟 / 关闭 四种 CUDA Graph 预设
- **角色管理** — 将角色名映射到参考音频和文本提示，支持 JSON 持久化
- **HTTP 服务器** — 轻量 FastAPI 服务，支持流式 TTS 接口、角色管理和健康检查
- **PyPI 安装** — `pip install "aqua-tts[runtime]"` → `from aquatts import TTSInferencer`

> **定位说明** — Aqua-TTS 是上游 GPT-SoVITS **v2、v2Pro、v2ProPlus、v3** 的优化层（保留原有 v1 路径）。它加载所选上游的 `Text2SemanticDecoder`，验证兼容契约，并且只替换直接 `infer_panel()` 路径；上游 batching/streaming 入口保持原样。不兼容的上游变更会明确失败。当前不支持 GPT-SoVITS v4。

## 新增：v2 系列性能基准

**最新测量：2026-10-21。** 环境为 USB4 外接 RTX 4070 Ti SUPER、Windows、
Torch 2.6.0+cu124、fp16、上游 `08d627c`。下表为预热后、起音裁切之前的首个原始 PCM 块延迟，
沿用 3／19／64 字符的短、中、长日语输入。

每个模型／执行方式／文本组合先预热 5 次，固定种子 `20260926` 测量 20 次，
再单独诊断 1 次；三个 v2 系列模型合计 **360 个正式样本**。首次调用单独保存。
正式请求未捕获新图，FA2 测试禁止回退。本轮只运行 Aqua SDPA 和 FA2。

**首个 PCM 包延迟（p50）**

| 模型 / 输入 | Aqua Graph + SDPA | Aqua Graph + FA2 |
|---|---|---|
| v2 · 短句 | 96.9 ms | 86.1 ms |
| v2 · 中句 | 162.0 ms | 132.4 ms |
| v2 · 长句 | 284.1 ms | 211.0 ms |
| v2Pro · 短句 | 100.7 ms | 88.8 ms |
| v2Pro · 中句 | 160.6 ms | 128.1 ms |
| v2Pro · 长句 | 267.2 ms | 211.2 ms |
| v2ProPlus · 短句 | 97.3 ms | 85.6 ms |
| v2ProPlus · 中句 | 169.2 ms | 138.9 ms |
| v2ProPlus · 长句 | 256.9 ms | 203.3 ms |

**短句首个 PCM 包延迟（p95，20 样本最近秩）**

| 模型 / 输入 | Aqua Graph + SDPA | Aqua Graph + FA2 |
|---|---|---|
| v2 · 短句 | 101.3 ms | 87.7 ms |
| v2Pro · 短句 | 104.2 ms | 97.8 ms |
| v2ProPlus · 短句 | 104.1 ms | 94.2 ms |

V2 Pro／FA2 短句最低 **85.2 ms**，p50 **88.8 ms**，p95 **97.8 ms**。
v2 沿用已有底模，Pro／Plus 沿用 Kurisu 微调权重，请在同一组权重内比较执行方式。
不同后端的浮点差异可能改变采样 token，因此这是同输入下的实测延迟，不是固定 token
数量的纯计算加速。v2 系列输出 32 kHz，按文本段解码后分块传输。基准进程单独绕过起音处理，以延续历史统计口径。

[本轮报告](benchmarks/results/latency-fix-20261021.md)与[原始样本](benchmarks/results/latency-fix-20261021.json)包含各输入的 p95、模型加载、
首次调用、总耗时／RTF、token 哈希和负载身份。
[九月报告](benchmarks/results/v2-model-retest-20260927.md)及原有官方基线完整保留归档。

## V2、V2 Pro 与 V2 Pro Plus 权重

传入匹配的 GPT／SoVITS 权重组合即可。Aqua 根据 SoVITS 版本头或上游模型身份
识别架构，不根据文件名猜测。Pro 与 Plus 还需要同一份 ERes2Net 说话人编码器：

```python
from aquatts import TTSInferencer

tts = TTSInferencer(
    gpt_path="/models/voice.ckpt",
    sovits_path="/models/voice.pth",
    sv_model_path="/models/pretrained_eres2netv2w24s4ep4.ckpt",  # 仅 Pro／Plus
    language="zh_CN",
)
sr, audio = tts.infer(
    text="今日はいい天気ですね。",
    ref_audio_path="/voices/reference.wav",
    prompt_text="与参考音频匹配的转写文本。",
    text_language="日文", prompt_language="日文",
)
```

`GPT_SOVITS_HOME` 需要指向包含 Pro 模型定义和 `GPT_SoVITS/eres2net/` 的上游版本
（已验证 `08d627c`）。未传入 `sv_model_path` 时，从上游目录的
`GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt` 加载。
编码器权重由用户提供，仅 Pro／Plus 加载；普通 v2 无需此权重。v2 系列也不加载
V3 底模或 BigVGAN，仍需 CNHuBERT 和相应文本前端资源；中文特征请求才需要 BERT 资源。

`infer()` 和 `infer_stream()` 均支持这些模型。主参考音频的说话人特征按会话缓存，
额外参考音频的频谱和说话人特征逐一配对。**v2 系列先解码完整文本段，再分块输出音频**；
`chunk_size_seconds` 控制传输块大小，不代表句内增量生成。T2S 的 static KV、CUDA Graph
及可选 FlashAttention 仍可使用；V3 的 CFM／BigVGAN 参数（例如 `sample_steps`）
不会加速 v2 系列解码，首页的 V3 延迟数据也不代表 v2 系列性能。

服务器命令 `python -m aquatts.server --gpt-model ... --sovits-model ...` 可增加
`--sv-model /models/pretrained_eres2netv2w24s4ep4.ckpt`。`/tts` 返回单声道 float32 PCM，
`X-Sample-Rate` 使用实际采样率；`/tts/file` 的 WAV 头使用相同采样率
（v2 系列权重通常为 32 kHz，v3 为 24 kHz）。流式合成失败会向调用者报告，
不再用静音音频块伪装成功。

具体权重与验证范围见[模型验证报告](benchmarks/results/v2-model-support.md)。
本次未加入 MPS／ROCm 支持。

### 可选模型按需加载

BERT 默认延迟加载：首次中文目标或参考片段才加载 tokenizer 和模型。目标及参考均为
纯英文／日文时，BERT 始终不加载；混合语言按文本前端产生的中文片段触发。
首次中文请求需要承担加载耗时。可在构造时设置 `preload_bert=True`，或在应用准备阶段
调用 `tts.preload_bert_model()`。`tts.bert_loaded` 和 `tts.bert_load_seconds` 分别表示
当前就绪状态和首次成功加载的耗时。并发首次调用只加载一次，失败后可以重试。

AP-BWE 模块及模型仅在请求超分时加载。缺少可选依赖或权重时，保留原有音频回退和警告。

### 参考预热与就绪

公共预热与两个推理 API 使用相同的归一化和缓存入口：

```python
result = tts.warm_reference_cache("/voices/reference.wav", "こんにちは。", "日文")
assert result["ready"]  # status 为 built 或 hit
```

构造器可接收 `warmup_references=[voice, ...]`，条目为 `Voice` 或包含
`ref_audio_path`、`prompt_text`、`prompt_language` 的字典。仅预热调用者提供的引用，
结果保存在 `tts.reference_warmup_results`。默认严格模式在失败时抛错；构造器设置
`warmup_strict=False`，或公共方法设置 `strict=False`，可获得明确的 `failed` 结果。

缓存键包含规范路径、文件修改时间和大小、归一化文本及语言、模型代次和实际设备／精度。
替换文件会重新构建，也可调用 `tts.clear_reference_cache()` 显式失效。V3 空文本 fallback
与正常推理共用同一引用；非 V3 无参考文本模式仅缓存声学条件，不编码未使用的语义 prompt。
并发构建只发布完整条目，CUDA event 保证跨 stream 读取顺序；公共预热在声明就绪前同步
所选设备。

`AQUA_SESSION_CACHE_MAX` 为正整数容量，默认 8，采用 FIFO 淘汰。构造器预热列表的不同
归一化引用数超过容量时直接拒绝。成功结果还包含键、驻留数量和被淘汰的键。已捕获 CUDA
Graph 时，更改设备或精度应新建推理器。省略预热仍可按请求构建参考缓存。

### CUDA 设备与 vocoder 预热

`device="cuda"` 在构造时解析为当前 CUDA 设备；`cuda:N` 指定进程可见设备索引，并遵循
`CUDA_VISIBLE_DEVICES` 映射。CUDA 不可用或索引越界会在加载权重前报错。BigVGAN 扩展缓存
选择与模型调用使用所选 CUDA 上下文。

BigVGAN 默认仍预热 40/70/128 mel 帧，各五次。可设置
`bigvgan_warmup_shapes=[9, 23, 47]`、`bigvgan_warmup_repeats=2`，空列表可关闭预热。
省略参数时读取 `AQUATTS_BIGVGAN_WARMUP_SHAPES`（逗号分隔，空值关闭）和
`AQUATTS_BIGVGAN_WARMUP_REPEATS`，显式参数优先。长度去重且必须为正整数；最多 16 个
不同形状、每形状 4096 帧、20 次重复，预热总帧数不超过 65536。

预热使用模型实际精度、设备及 mel 通道，在 inference mode 下执行并保留 CPU 与所选 GPU
的 RNG 状态。`tts.vocoder_warmup_stats` 报告完成、关闭或失败，失败会记录日志；
`tts.vocoder_shape_counts` 统计真实请求长度。24 kHz V3 的 0.1/0.25/0.5 秒块可能对应
9/23/47 帧，另有短尾块，因此默认集合不覆盖所有负载。普通 vocoder 调用不增加全设备同步，
诊断计时仍需显式开启。辅助 `load_bigvgan(..., device="cuda:N")` 也支持指定设备。

## 语言支持

Aqua-TTS 使用所选 GPT-SoVITS 权重对应的文本前端。通过 `text_language` / `prompt_language` 参数传入语言代码，参考音频和目标文本可使用不同语言。

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
│   │   └── t2s_flash_attn.py      # 自适应 FlashAttention2 KV-cache 补丁
│   ├── bigvgan/
│   │   ├── cuda/                  # 独立 CUDA 内核加载器与源码
│   │   └── torch/                 # 纯 PyTorch 回退
│   ├── inference/
│   │   ├── streaming.py           # 音频后处理
│   │   ├── semantic_stability.py  # 声码器前的语义塌缩准入检查
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

Aqua-TTS 已在 RTX 4070 Ti SUPER 上使用 PyTorch 2.6.0+cu124 测试通过。

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

`infer()` 和 `infer_stream()` 省略 `enable_cuda_graph` 或传入 `None` 时继承实例配置。
显式 `False` 关闭本次 replay；`True` 在 NVIDIA CUDA 与静态 KV 可用时开启，包括
在 `off` 实例上按需捕获。`enable_static_kv=False` 始终关闭 Graph。
`ENABLE_CUDA_GRAPH=0/1` 在构造时覆盖 preset；`ENABLE_CUDA_GRAPH_PRECAPTURE`
和 `CUDA_GRAPH_PRECAPTURE_BUCKETS` 只覆盖启动捕获。构造后修改环境变量不改变该实例。

请在**开始接收请求前**完成服务所需桶的预捕获。捕获需要在进程内协作的 CUDA 工作之间
取得独占准入；存在其他活跃请求（包括暂停的流）时会推迟捕获，本次解码回退 eager。
持续重叠的请求可能让未准备的桶一直走 eager。`lazy`、`minimal` 未覆盖的桶、
启动捕获失败，以及开服后才构造的实例都会遇到这一情况。`full` 只有在启动捕获成功后，
才能避免其配置桶在请求中临时捕获。需要补捕获时，先停止接收新请求并结束现有流，
再调用 `tts.t2s_model.model.precapture_cuda_graph([128, 256, 448, 512, 768, 1024])`，
按显存预算选择桶，确认返回的各桶布尔值后恢复服务。当前没有后台捕获或保证捕获机会
的调度机制；同进程的其他 CUDA 使用方也需要接入共享工作准入。

请求完成后可查看 `tts.t2s_model.model.cuda_graph_stats["graph_replay_steps"]`
确认实际执行；捕获成功不等于已 replay。

同一 decoder 中共享同一 Graph 工作区的请求，会在完整 T2S AR 生成期间串行执行，
覆盖 prompt 安装与采样。跨 stream 交接使用 CUDA event，不增加逐 token 的设备同步。
不同存储 owner 和不同 decoder 实例保持独立所有权；关闭 Graph 的请求使用私有 KV。
计数器描述最后完成的 AR 请求，不是并发请求的累加总数。

每个计算桶只捕获一张 Graph，不同真实前缀长度复用该图。
`TTSInferencer(kv_cache_layout="bucket")` 按桶独立保存 KV，是当前默认值；
`kv_cache_layout="nested"` 使用同一个最大容量根的短桶视图，减少多桶 KV 驻留并保留
短桶 attention 形状。共享根的 Graph 请求共用一个 owner。省略参数时构造器读取
`AQUATTS_KV_CACHE_LAYOUT`（默认 `bucket`），显式参数优先。捕获后改变布局、设备、
dtype 或容量需要新建 decoder。只用少量短图时 bucket 可能更省显存，nested 并非始终更小。
本轮保留这些默认策略：`bucket` 维持各桶独立所有权，预设限制启动显存开销，
CFM unpadded 在进一步硬件与质量验证前仍需显式开启。需要共享 KV 驻留及已准备桶升桶时，
可显式选择 `nested`。full/minimal 只准备各自声明的桶；lazy/off 不在启动时分配 Graph 根。nested 桶满后，
切换到最小的、更大的**已捕获**桶，原地保留有效 KV，并清零新暴露的尾区。升桶不捕获
新图，也不扩大准备集合。没有已准备目标或使用 bucket 存储时，保留完整历史转动态解码。

`decoder.cuda_graph_stats` 记录最后完成请求的起止 KV、起止桶、升桶次数、Graph/静态/
动态步数、捕获尝试和工作区等待秒数。数值日志提供相同路径信息，不记录文本内容。

**自适应 FlashAttention2 T2S 路径**（兼容时自动优先）：

Aqua-TTS 可以用 `flash_attn_with_kvcache` 替换 q_len=1 的 T2S 静态 KV attention 步骤。`valid` 模式按真实 KV 长度参与注意力，而不是读取完整零填充 bucket；现在只要能导入兼容的 FlashAttention2 就默认使用。如果 FA2 不存在，或者内核拒绝当前 shape/device，Aqua 会回退到 SDPA。

默认不需要环境变量。需要时可以覆盖自动选择：

```bash
export AQUATTS_T2S_FLASH_ATTN=0  # 强制 SDPA
# export AQUATTS_T2S_FLASH_ATTN=1  # 显式请求 FA2
export AQUATTS_T2S_FLASH_ATTN_MODE=valid  # valid | bucket
```

也可以通过 Python 的 `use_flash_attn=False` / `True` 显式覆盖：

```python
tts = TTSInferencer(..., use_flash_attn=True, flash_attn_mode="valid")
```

2026 年 6 月“短句无收益、长句约提升 8%”的结论是在 CUDA Graph replay 逐 token 强制同步仍存在时测得，已不代表当前性能。

2026-10-21 的固定工作量复测使用 USB4 外接 RTX 4070 Ti SUPER、
Torch 2.6.0+cu124，预热 15 次，每个形状同步测量 7 次：

| 形状 | SDPA | FA2 | 本轮差值 |
|---|---|---|---|
| short / 448 | 504.3 it/s | 618.7 it/s | 23% |
| conversation / 512 | 488.8 it/s | 687.5 it/s | 41% |
| long / 768 | 460.8 it/s | 738.5 it/s | 60% |

这些是同卡固定 T2S 工作量的实测差值，仍受桌面调度和 GPU P-state 影响，不能用来预测完整管线首包延迟。原始 PCM 的 TTFP 与裁切后的 API 首包是两个指标。已将独立 #12 的历史缓冲复用与前 11 步 EOS 跳过整合到迁移栈。同场配对未复现此前的大幅首包差距；报告保留所有批次，并区分代码收益和运行波动。0.3.0 保留已标注的 Laptop 旧测量，新的 Laptop 数据留待后续补充。

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

`/tts` 在发送成功响应头前完成参考准备与校验。参考准备失败时返回 HTTP 500，
响应为带 `detail` 的 JSON，不发送 `X-Sample-Rate`。冷参考会让响应头等待准备完成；
缓存命中时复用已准备的特征。

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

所有版本的在线请求都会在语义生成前传播参考准备错误，只复用完整参考缓存。
显式可选预热仍可使用 `strict=False` 返回失败状态。V3 本地参考音频改用 SoundFile
解码；TorchAudio 仅用于重采样，该路径无需 TorchCodec。

### 并发 CUDA 工作与 Graph 捕获

Aqua 推理、流式生成器完整生命周期、构造器、参考/BERT 预加载和 vocoder 预热
共享进程级捕获准入。请求之间仍可并发；有其他工作租约时推迟捕获，本次请求使用
eager 路径，之后可重试。捕获已获准时，新工作等待捕获结束。流式租约可跨线程池
线程恢复，HTTP 取消时会关闭释放。

同进程其他 GPU 调用方应从 `aquatts.inference.cuda_execution` 导入
`cuda_execution`，用同步的 `with cuda_execution.work():` 包住 CUDA 工作。
绕过该协作准入的外部 CUDA 调用无法由 Aqua 保护。暂停的流会保留租约，直到正常
耗尽或显式调用 `close()`。

### V3 语义与声学参考

`infer()` 和 `infer_stream()` 均可传 `semantic_reference`，接受 `Voice` 或包含
`ref_audio_path`、`prompt_text`、`prompt_language`（默认日文）的字典。普通参考
提供声学条件，显式参考提供语义条件：

| 字段 | 来源 |
| --- | --- |
| `prompt`、`phones1`、`bert1` | 语义参考 |
| `refer_spec`、`prompt_fea_ref`、`prompt_ge`、`mel2_norm` | 普通声学参考 |

```python
from aquatts import Voice

semantic = Voice("semantic", "semantic.wav", "对应的真实转写文本。", "中文")
sr, audio = tts.infer(
    "需要合成的内容。", ref_audio_path="acoustic.wav",
    prompt_text="声学参考的真实转写文本。", prompt_language="中文",
    text_language="中文", semantic_reference=semantic,
)
```

混合时两份转写均须显式且非空，语言必须有效；`ref_free=True` 和非 V3 模型会被
拒绝。不完整参考明确报错，不会悄悄换用另一份条件。每个请求独立复制组合字段，
不改写共享缓存。可用 `warmup_references` 或 `warm_reference_cache` 预热两份参考；
要同时驻留两个不同的归一化条目，缓存容量至少为 2。省略可选参数保持普通参考行为。
该 API 明确条件来源，不承诺固定情绪强度或音色完全不变，音质和听检另行评估。

### V3 CFM mask 优化实验

`TTSInferencer(cfm_unpadded=True)` 显式启用实验路径，默认 `False`。
Aqua 构造每个 V3 CFM 块时已掌握完整长度，只有这个单样本、无 padding 的情况省略
attention mask。未知长度、batch 和有 padding 的输入仍使用 mask，并在扩散循环外
构建一次。不会把 GPU 长度标量读回 CPU。

实例适配器检查上游接口后，只在内存中调整 DiT 的 mask 赋值。CFM 扩散循环、
`infer=True` tuple 协议、条件/CFG 文本缓存、时间步缓存和训练分支均保留上游实现，
不覆盖外部 checkout 文件或修改类/全局函数。不兼容的上游会在启用时报错；关闭参数
即可恢复普通 mask 路径，非 V3 模型明确拒绝该参数。

`benchmarks/cfm_unpadded_bench.py --config ... --output ...` 提供同噪声的
4/16/32 步、CFG、FP16/FP32、短尾和长参考测试，以及普通/流式配对测量。
数值阈值在测量前固定，保留基线重复误差和原始耗时样本。更多硬件和人工听检完成前
保持默认关闭。

### 本地资源与离线模型加载

构造时传 `offline_only=True`，或在 `examples/basic_usage.py` 中使用
`--offline-only`，要求资源事先安装到本地；默认值为 `False`。
BERT 和两个 BigVGAN 加载器会传 `local_files_only=True`；CNHuBERT 上游已使用
本地加载。缺文件错误给出组件和预期目录。BERT 仍在首次中文使用或显式
`preload_bert_model()` 时才检查。语言识别使用各实例独立的 detector，以及
`fast_langdetect_path/lid.176.bin`，离线实例不会回退下载。在线实例保留原有
下载行为及全局 detector，不再要求私有分词适配接口；离线 detector 仍为实例独立。

前端资源需在首次导入前准备：

- 英文：NLTK 的 `cmudict`、`averaged_perceptron_tagger` zip 资源；NLTK 3.9+
  还需要 `averaged_perceptron_tagger_eng`。
- 日文：`OPEN_JTALK_DICT_DIR` 指向完整词典，须在导入 pyopenjtalk 前设置；
  也可使用包内已安装的默认词典。
- 中文：上游 G2PW 仍读取相对进程工作目录的 `GPT_SoVITS/text/G2PWModel`，
  其中需有 config、字符表、发音 JSON 和 `g2pW.onnx`/`g2pw.onnx`。
  它的 tokenizer 使用上游 `bert_path` 环境变量（须在导入 `text.chinese2`
  前设置），默认是 `GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large`。
  这与构造参数 `bert_path` 是两个入口，都必须指向本地资源。进程启动时配置好
  资源布局；Aqua 推理期间不会切换 cwd。
- 可选 AP-BWE：所选上游 `tools/audio_sr.py` 旁的
  `AP_BWE_main/24kto48k/{config.json,g_24kto48k.zip}`。检查和加载使用同一路径，
  不依赖工作目录。缺组件时保留既有不执行超分的回退行为。

本轮检查边界覆盖英、日、中及其混合，离线韩语/粤语请求会明确拒绝。这是模型资源
约束，不是网络沙箱；不会设置 Hub 环境开关或修改 Hub 私有状态，可在 Hub/Transformers
已导入后使用。上游 LangSegmenter 首次导入仍会设置其进程默认 detector；应先初始化
该上游模块，再初始化其他使用它全局默认值的调用方，或使用显式 detector 实例。
离线构造器绑定私有函数，不改写上游函数和全局默认值；在线构造器保留上游全局
detector 配置方式。已核对上游
`08d627c`、fast-langdetect 1.0.1、split-lang 2.1.1、g2p-en 2.1.0、pyopenjtalk
0.4.1；私有分词适配接口不兼容仅影响离线模式。

`benchmarks/optional_models_smoke.py --offline-only --config ... --output ...
--frontend-cwd ...` 会在独立进程拦截网络连接，使用本地真实权重测试英日中和混合推理。

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
| `AQUATTS_T2S_FLASH_ATTN` | `auto` / 未设置 | `0` 强制 SDPA；`1` 显式请求 FlashAttention2；`auto`、空值或未设置时优先尝试 FA2 |
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

每个合成单元现在会裁掉开头低能量静音，保留首个达到 -45 dBFS 的 10 ms
RMS 窗口之前的 50 ms。单元内部停顿、尾部音频和 `pause_second` 保留，全程
未达到阈值的音频原样返回。流式路径等完整检测帧到齐再判断，文本随首个输出
音频块发送。这是振幅规则，不是语义语音检测器，不保证所有权重都有同样收益。

首页 TTFP 延续裁切前首个原始 PCM 块的定义；裁切后的 API 首包另行统计，两者都不代表声卡出声时刻。
Demo 记录 API 首包可用时间，另加 `lead_ms`（首个有效窗口之前的 PCM 时长）和
`first_voiced_chunk_ms`（包含该窗口的音频块准备提交的时刻），均不是扬声器
声学回录时间。暂存生成的静音块可能使首包时间变长，却减少实际提交播放的静音；
应同时比较有声样本位置与听感。

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
