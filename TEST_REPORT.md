# Aqua-TTS 开源项目本地部署与测试报告

日期：2026-05-24  
测试目录：`F:\aquatts\Aqua-TTS`  
主参考 GPT-SoVITS 目录：`F:\Computer_Science\GPT-SoVITS-official`  
用户参考音频与 v3 LoRA 目录：`F:\BaiduNetdiskDownload\GPT-SoVITS\GPT-SoVITS-v3lora-20250401`

## 1. 测试目标

本轮测试目标是按照 README 思路完成本地部署，验证项目是否能在真实 GPT-SoVITS v3 LoRA 资产下运行，并检查测试、示例脚本、benchmark 口径是否可靠。

重点关注：

- 新 clone 项目能否安装。
- Python/PyTorch/CUDA 环境是否可用。
- vendored GPT-SoVITS 模块是否正确生效。
- 示例脚本和 benchmark 是否能跑。
- TTFP benchmark 是否真正测量“首声可播放音频”。
- README/benchmark 数据是否有可复现标准。

## 2. 本地环境

### 2.1 Python 与依赖

已创建并使用项目虚拟环境：

```powershell
py -3.12 -m venv --clear .venv
.venv\Scripts\python.exe -m pip install -e ".[runtime,server]" -r requirements-dev.txt
```

安装过程中额外安装了：

```powershell
torch==2.5.1+cu121
torchaudio==2.5.1+cu121
```

并安装 GPT-SoVITS requirements：

```powershell
.venv\Scripts\python.exe -m pip install -r F:\Computer_Science\GPT-SoVITS-official\requirements.txt
```

### 2.2 GPU 与 CUDA

PyTorch CUDA 可用，检测到 2 张 GPU：

- `NVIDIA GeForce RTX 4070 Ti SUPER`
- `NVIDIA GeForce RTX 4070 Laptop GPU`

benchmark 实际使用：

- `NVIDIA GeForce RTX 4070 Ti SUPER`

### 2.3 使用的 GPT-SoVITS 资产

主仓库：

```text
F:\Computer_Science\GPT-SoVITS-official
branch: main
HEAD: 08d627c...
commit date: 2026-04-30
```

用户提供的 v3 LoRA 资产：

```text
F:\BaiduNetdiskDownload\GPT-SoVITS\GPT-SoVITS-v3lora-20250401
HEAD: 12adc6c...
commit date: 2026-05-24
subject: perf: precapture short utterance graph keys
```

实际测试使用：

```text
GPT model:
F:\BaiduNetdiskDownload\GPT-SoVITS\GPT-SoVITS-v3lora-20250401\GPT_weights_v3\xxx-e15.ckpt

SoVITS model:
F:\BaiduNetdiskDownload\GPT-SoVITS\GPT-SoVITS-v3lora-20250401\SoVITS_weights_v3\xxx_e2_s174_l32.pth

Reference audio:
F:\BaiduNetdiskDownload\GPT-SoVITS\GPT-SoVITS-v3lora-20250401\reference audio\kurisu_reference.wav

Reference text:
そういえば,正式に自己紹介していませんでしたね……牧瀬紅莉栖です.改めてまして,よろしく
```

## 3. 初始发现的问题

### 3.1 `pyproject.toml` 文件开头存在 BOM，导致 editable install 失败

现象：

```text
tomllib.TOMLDecodeError: Invalid statement (at line 1, column 1)
```

原因：

- `pyproject.toml` 文件开头存在 UTF-8 BOM。
- Python 3.12 的 `tomllib` 对该文件解析失败。

处理：

- 移除 `pyproject.toml` 开头 BOM。

建议：

- 保持 TOML 文件为普通 UTF-8 无 BOM。
- 后续 CI 可增加最小安装测试，提前发现 packaging 问题。

### 3.2 package 名称与示例导入不一致

现象：

示例与 benchmark 原先使用：

```python
from aqua import TTSInferencer
```

但实际包名是：

```python
from aquatts import TTSInferencer
```

影响：

- 用户按 README 或 examples 跑时会直接导入失败。
- benchmark 脚本无法作为可信复现入口。

处理：

- 修正 `examples/basic_usage.py`
- 修正 `examples/streaming_inference.py`
- 修正 `benchmarks/aqua_ttfp.py`
- 修正 `benchmarks/t2s_speed_bench.py`

建议：

- README、examples、benchmark 中统一只使用 `aquatts`。
- 如果项目想保留 `aqua` 兼容导入，应显式增加 alias package 或兼容模块。

### 3.3 vendored GPT-SoVITS 路径优先级不足

现象：

Aqua-TTS 有 vendored GPT-SoVITS patch，但上游代码中存在类似导入：

```python
from AR.models.t2s_model import Text2SemanticLightningModule
```

该导入不是：

```python
from GPT_SoVITS.AR.models.t2s_model import ...
```

因此如果只把 `aquatts/_vendor` 加入 `sys.path`，仍可能导入外部 GPT-SoVITS 的顶层 `AR` 包，而不是 Aqua vendored 版本。

影响：

- 本地运行时实际加载的是外部 GPT-SoVITS 文件。
- Aqua 的 patch 可能没有生效。
- 曾触发外部文件中的 GBK 编码打印问题。
- 测试结果不稳定，依赖用户机器上的路径顺序。

处理：

- 在 `aquatts/__init__.py` 中优先插入：

```text
aquatts/_vendor/GPT_SoVITS
aquatts/_vendor
```

- 增加测试确保：

```python
import AR.models.t2s_model
```

来自 Aqua vendored 路径。

建议：

- 继续保持 vendored GPT-SoVITS 路径优先。
- CI 中加入路径断言，防止未来改动破坏 patch 生效顺序。

### 3.4 BigVGAN CUDA kernel 在 pytest collection 阶段被提前编译

现象：

导入 `aquatts.bigvgan.cuda` 时会触发 CUDA extension 相关逻辑，pytest collection 阶段可能开始编译或访问 CUDA build cache。

影响：

- 单元测试收集阶段变慢。
- 没有 CUDA 或没有 nvcc 的机器上更容易失败。
- 单元测试不应该因为 import 就触发重型 CUDA side effect。

处理：

- `aquatts/bigvgan/cuda/__init__.py` 改为 lazy load `Activation1d` 与 `FusedAntiAliasActivation`。

建议：

- 保持 CUDA extension 在真正推理或显式调用时才初始化。
- 单元测试默认不依赖 GPU 编译。

### 3.5 benchmark 脚本路径仍指向旧目录结构

现象：

部分 benchmark 仍引用旧路径：

```text
Aqua/_vendor
```

实际项目目录为：

```text
aquatts/_vendor
aquatts/_vendor/GPT_SoVITS
```

影响：

- `bigvgan_raw_bench.py` 与 `t2s_speed_bench.py` 不能在当前 repo 布局下可靠执行。
- 可能误用外部 GPT-SoVITS 代码。

处理：

- 修正 benchmark 中的 vendor path。
- 确保 `sys.path` 插入顺序优先使用 Aqua vendored 路径。

## 4. 已执行的功能测试

### 4.1 单元测试

命令：

```powershell
$env:GPT_SOVITS_HOME='F:\BaiduNetdiskDownload\GPT-SoVITS\GPT-SoVITS-v3lora-20250401'
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUTF8='1'
.venv\Scripts\python.exe -m pytest tests -vv --tb=short
```

结果：

```text
66 passed, 1 warning in 11.76s
```

说明：

- 修正后测试套件通过。
- pytest cache 目录在 sandbox 下有权限 warning，但不影响测试结果。

### 4.2 端到端 smoke test

测试目标：

- 使用用户提供的 v3 LoRA 权重。
- 使用 `kurisu_reference.wav` 作为参考音频。
- 生成一段实际 wav。

结果：

```text
output: F:\aquatts\Aqua-TTS\smoke_kurisu.wav
sample rate: 24000 Hz
channels: mono
duration: 4.27s
frames: 102464
```

性能观察：

- 首次 BigVGAN CUDA kernel 编译导致整体耗时约 108s。
- 编译缓存生成后，实际推理约 9.92s。

已生成 CUDA kernel cache：

```text
F:\aquatts\Aqua-TTS\aquatts\_vendor\GPT_SoVITS\BigVGAN\alias_free_activation\cuda\build_sm89_16gb_nvidia_geforce_rtx_4070_ti_super\anti_alias_activation_cuda.pyd
```

建议：

- README 应说明首次运行可能会编译 BigVGAN CUDA kernel。
- benchmark 应区分 cold start 与 warm cache。

## 5. Benchmark 测试结果

### 5.1 BigVGAN raw benchmark

命令核心：

```powershell
.venv\Scripts\python.exe benchmarks\bigvgan_raw_bench.py
```

结果：

```text
Loaded in 4.2s
use_cuda_kernel=True
FP16
num_mels=100

mel_T=70   median 27.1ms
mel_T=128  median 30.0ms
mel_T=298  median 50.6ms
mel_T=598  median 82.7ms

GPU: NVIDIA GeForce RTX 4070 Ti SUPER
```

说明：

- 已加载缓存的 CUDA kernel。
- 此项适合衡量 vocoder 单独性能。

### 5.2 T2S speed benchmark

命令核心：

```powershell
$env:GPT_SOVITS_HOME='F:\BaiduNetdiskDownload\GPT-SoVITS\GPT-SoVITS-v3lora-20250401'
.venv\Scripts\python.exe benchmarks\t2s_speed_bench.py --gpt-model "...\GPT_weights_v3\xxx-e15.ckpt" --repeats 5
```

结果：

```text
median T2S speed: 469 it/s
trials: 469, 473, 453, 465, 501 it/s
graph replay: ~98.7% - 99.3%
bucket: 256
```

说明：

- 此项主要衡量 GPT semantic token 推理速度。
- 需要与 TTFP、完整生成耗时分开报告。

### 5.3 原始 TTFP benchmark 口径问题

原始 `aqua_ttfp.py` 存在两个口径问题：

1. 默认使用 `how_to_cut="不切"`。
2. 未传入 `chunk_size_seconds`，因此 v3 没有走真正的分块播放路径。

代码现象：

- `infer_stream()` 的 v3 真流式分支依赖 `chunk_size_seconds`。
- 当 `chunk_size_seconds=None` 时，v3 会先累积完整 mel，再调用 BigVGAN。
- benchmark 记录的第一个非空 chunk 实际接近“完整句子生成后第一次 yield”，不是“第一声播放”。

原始结果：

```text
short   median ttfp 295.6ms
medium  median ttfp 416.0ms
long    median ttfp 901.6ms
```

问题判断：

- 这个结果偏保守。
- 对长文本尤其不准确，因为它把整句生成耗时混入了首声指标。

## 6. 修正后的 TTFP 标准

### 6.1 建议定义

TTFP 建议定义为：

```text
从调用 infer_stream 开始，到拿到第一个非空、可直接播放的 audio chunk 为止的时间。
```

该指标不应包括完整文本生成完成时间。

### 6.2 benchmark 默认口径

已将 `benchmarks/aqua_ttfp.py` 改为：

```text
Time-To-First-Playback
```

默认参数：

```text
--how-to-cut 按标点符号切
--chunk-size-seconds 0.25
```

默认行为：

- 拿到第一个非空 audio chunk 后立即停止计时。
- 输出列使用 `first_audio` 与 `elapsed`。
- 如需完整生成耗时，显式加：

```text
--measure-total
```

### 6.3 修正后 TTFP 结果

命令核心：

```powershell
$env:GPT_SOVITS_HOME='F:\BaiduNetdiskDownload\GPT-SoVITS\GPT-SoVITS-v3lora-20250401'
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUTF8='1'
.venv\Scripts\python.exe benchmarks\aqua_ttfp.py `
  --gpt-model "F:\BaiduNetdiskDownload\GPT-SoVITS\GPT-SoVITS-v3lora-20250401\GPT_weights_v3\xxx-e15.ckpt" `
  --sovits-model "F:\BaiduNetdiskDownload\GPT-SoVITS\GPT-SoVITS-v3lora-20250401\SoVITS_weights_v3\xxx_e2_s174_l32.pth" `
  --ref-audio "F:\BaiduNetdiskDownload\GPT-SoVITS\GPT-SoVITS-v3lora-20250401\reference audio\kurisu_reference.wav" `
  --ref-text "そういえば,正式に自己紹介していませんでしたね……牧瀬紅莉栖です.改めてまして,よろしく"
```

运行日志确认进入 v3 真流式路径：

```text
[v3-stream] enable true streaming: mel_chunk=23 (default=466, target_frames=23)
```

结果：

```text
short    chars=3   median ttfp=262.2ms
medium   chars=19  median ttfp=317.5ms
long     chars=64  median ttfp=408.9ms

GPU: NVIDIA GeForce RTX 4070 Ti SUPER
```

说明：

- 这组结果更符合“首声播放”定义。
- 长文本 TTFP 从约 901.6ms 降至约 408.9ms，主要原因是 benchmark 不再等待完整句子输出。

## 7. 推荐的 benchmark 报告标准

建议 README 或 `benchmarks/README.md` 中固定如下字段，避免不同机器结果不可比：

```text
OS:
Python:
torch:
torchaudio:
CUDA available:
GPU:
GPU driver:
Aqua-TTS commit:
GPT-SoVITS commit:
GPT model path/type:
SoVITS model path/type:
Reference audio:
Reference text language:
Target text language:
how_to_cut:
chunk_size_seconds:
sample_steps:
speed:
CUDA Graph:
Static KV:
BigVGAN CUDA kernel:
Cold start or warm cache:
```

建议性能表至少分四类：

```text
1. Smoke test: 是否能生成 wav
2. TTFP: first playable audio chunk
3. Full generation: 完整文本生成完成
4. Component benchmark: T2S speed / BigVGAN raw
```

不要把以下指标混在同一列：

- 首声播放时间
- 首个完整句子输出时间
- 完整文本生成时间
- vocoder 单独耗时
- T2S token speed

## 8. 当前代码改动摘要

本轮涉及的主要文件：

```text
pyproject.toml
aquatts/__init__.py
aquatts/bigvgan/cuda/__init__.py
examples/basic_usage.py
examples/streaming_inference.py
benchmarks/aqua_ttfp.py
benchmarks/bigvgan_raw_bench.py
benchmarks/t2s_speed_bench.py
tests/test_vendor_paths.py
```

改动方向：

- 修复安装失败。
- 修复包名导入错误。
- 修复 vendored GPT-SoVITS path 顺序。
- 避免 pytest collection 触发 BigVGAN CUDA 编译。
- 修复 benchmark 路径。
- 修正 TTFP benchmark 口径。
- 增加 vendored 顶层 `AR` 导入路径测试。

## 9. 未跟踪产物

本地测试过程中产生了以下未跟踪文件：

```text
.tmp/
aqua-ttfp.log
pytest-run.log
smoke_kurisu.wav
tts-smoke.log
```

建议：

- 不要直接提交这些临时产物。
- 如需保留性能数据，可整理为 `benchmarks/results/*.md`。
- 如需保留 smoke 音频，建议放到 release artifact 或外部数据集，不建议进入主仓库。

## 10. 建议后续提交拆分

建议拆成 3 个 commit：

### Commit 1: 修复安装与导入路径

包含：

- `pyproject.toml`
- `aquatts/__init__.py`
- `tests/test_vendor_paths.py`

建议 commit message：

```text
fix: ensure editable install and vendored GPT-SoVITS path precedence
```

### Commit 2: 修复 examples 与 benchmark 脚本入口

包含：

- `examples/basic_usage.py`
- `examples/streaming_inference.py`
- `benchmarks/bigvgan_raw_bench.py`
- `benchmarks/t2s_speed_bench.py`

建议 commit message：

```text
fix: update examples and benchmarks for aquatts package layout
```

### Commit 3: 修正 TTFP benchmark 口径

包含：

- `benchmarks/aqua_ttfp.py`
- 可选：`benchmarks/README.md`
- 可选：`benchmarks/results/4070ti-super-win11-torch251-cu121-v3lora.md`

建议 commit message：

```text
fix: measure TTFP as first playable audio chunk
```

## 11. 建议后续补充

建议继续补充：

- `benchmarks/README.md` 中写清 benchmark 标准。
- `benchmarks/results/` 保存可复现实验表。
- CI 增加：
  - package install test
  - import test
  - non-GPU pytest
  - benchmark script `--help` 或 dry-run test
- README 中说明：
  - 首次 BigVGAN CUDA kernel 编译会很慢。
  - warm cache 后性能才适合和 benchmark 表比较。
  - TTFP 指 first playable audio chunk。

## 12. 当前结论

项目在本地 Windows + Python 3.12 + PyTorch 2.5.1 cu121 + RTX 4070 Ti SUPER 环境下可以跑通。

当前主要问题不是核心推理完全不可用，而是：

- packaging 细节导致新用户安装失败。
- examples/benchmark 与真实 package 布局不一致。
- vendored GPT-SoVITS patch 路径优先级容易被外部仓库污染。
- TTFP benchmark 旧口径不等于首声播放。

修正后：

- 单元测试通过：`66 passed`
- v3 LoRA 端到端 smoke test 通过
- BigVGAN raw benchmark 可跑
- T2S speed benchmark 可跑
- TTFP benchmark 已按 first playable audio chunk 重新定义并跑出新数据

建议优先提交功能修复与 benchmark 口径修正，再补 README/benchmark result 文档。
