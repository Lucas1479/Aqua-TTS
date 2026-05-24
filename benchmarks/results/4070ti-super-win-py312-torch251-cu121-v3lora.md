# Aqua-TTS Benchmark Result

Date: 2026-05-24

## Environment

| Field | Value |
|-------|-------|
| OS | Windows |
| Python | 3.12 |
| PyTorch | 2.5.1+cu121 |
| CUDA | 12.1 |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER |
| Aqua-TTS path | `F:\aquatts\Aqua-TTS` |
| GPT-SoVITS asset path | `F:\BaiduNetdiskDownload\GPT-SoVITS\GPT-SoVITS-v3lora-20250401` |
| GPT-SoVITS asset commit | `12adc6c...` |
| BigVGAN CUDA kernel | enabled, warm cache |

## Model Assets

| Asset | Path |
|-------|------|
| GPT model | `GPT_weights_v3\xxx-e15.ckpt` |
| SoVITS model | `SoVITS_weights_v3\xxx_e2_s174_l32.pth` |
| Reference audio | `reference audio\kurisu_reference.wav` |
| Reference language | Japanese |
| Text language | Japanese |

Reference text:

```text
そういえば,正式に自己紹介していませんでしたね……牧瀬紅莉栖です.改めてまして,よろしく
```

## TTFP Method

TTFP means **Time-To-First-Playback**: the elapsed wall-clock time from entering `infer_stream()` to the first non-empty audio chunk that can be sent to playback.

Parameters:

```text
how_to_cut=按标点符号切
chunk_size_seconds=0.25
speed=1.1
sample_steps=4
top_k=5
top_p=1
temperature=0.6
CUDA Graph=enabled
Static KV=enabled
BigVGAN CUDA kernel=enabled
```

The benchmark stops at the first playable audio chunk by default. Use `--measure-total` to measure full generation latency separately.

Command:

```powershell
$env:GPT_SOVITS_HOME='F:\BaiduNetdiskDownload\GPT-SoVITS\GPT-SoVITS-v3lora-20250401'
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUTF8='1'
.venv\Scripts\python.exe benchmarks\aqua_ttfp.py `
  --gpt-model 'F:\BaiduNetdiskDownload\GPT-SoVITS\GPT-SoVITS-v3lora-20250401\GPT_weights_v3\xxx-e15.ckpt' `
  --sovits-model 'F:\BaiduNetdiskDownload\GPT-SoVITS\GPT-SoVITS-v3lora-20250401\SoVITS_weights_v3\xxx_e2_s174_l32.pth' `
  --ref-audio 'F:\BaiduNetdiskDownload\GPT-SoVITS\GPT-SoVITS-v3lora-20250401\reference audio\kurisu_reference.wav' `
  --ref-text 'そういえば,正式に自己紹介していませんでしたね……牧瀬紅莉栖です.改めてまして,よろしく'
```

Log confirmation:

```text
[v3-stream] enable true streaming: mel_chunk=23 (default=466, target_frames=23)
```

## TTFP Results

| Case | Chars | Median TTFP |
|------|------:|------------:|
| short | 3 | 262.2ms |
| medium | 19 | 317.5ms |
| long | 64 | 408.9ms |

Raw repeats:

| Case | Repeats |
|------|---------|
| short | 259.0ms, 262.2ms, 242.7ms, 266.9ms, 271.2ms |
| medium | 317.5ms, 322.4ms, 323.0ms, 314.2ms, 314.2ms |
| long | 435.2ms, 403.4ms, 402.8ms, 417.1ms, 408.9ms |

## Component Benchmarks

### T2S Speed

| Metric | Value |
|--------|------:|
| Median speed | 469 it/s |
| Repeats | 469, 473, 453, 465, 501 it/s |
| Graph replay | ~98.7%-99.3% |
| Bucket | 256 |

### BigVGAN Raw

| mel_T | Median |
|------:|-------:|
| 70 | 27.1ms |
| 128 | 30.0ms |
| 298 | 50.6ms |
| 598 | 82.7ms |

Settings:

```text
use_cuda_kernel=True
precision=FP16
num_mels=100
```

## Notes

- The first run may be much slower because BigVGAN CUDA kernels can compile on cold cache.
- TTFP, full generation latency, T2S throughput, and BigVGAN raw timing are separate metrics and should not be compared as the same latency.
- This result uses a v3 LoRA asset set and should be compared with runs using the same model family and reference audio.
