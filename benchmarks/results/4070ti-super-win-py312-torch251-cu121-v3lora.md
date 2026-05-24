# Aqua-TTS Benchmark Result

Date: 2026-05-25

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

TTFP means **Time-To-First-Playback**: the elapsed wall-clock time from entering TTS inference to the first non-empty audio buffer that can be sent to playback.

For non-streaming baselines, the first returned playable audio may coincide with full utterance completion. For Aqua-TTS, TTFP is the first streaming audio chunk returned by `infer_stream()`.

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

| Case | Chars | Median TTFP | Mean ± Std |
|------|------:|------------:|------------|
| short | 3 | 256.7ms | 260.4 ± 21.0ms |
| medium | 19 | 301.4ms | 304.2 ± 5.3ms |
| long | 64 | 404.4ms | 404.0 ± 7.6ms |

Raw repeats:

| Case | Repeats |
|------|---------|
| short | 293.6ms, 229.0ms, 254.4ms, 256.7ms, 268.1ms |
| medium | 301.4ms, 311.4ms, 297.9ms, 309.7ms, 300.6ms |
| long | 416.6ms, 400.2ms, 404.4ms, 393.3ms, 405.4ms |

## Component Benchmarks

### T2S Speed

| Metric | Value |
|--------|------:|
| Median speed | 449 it/s |
| Repeats | 449, 463, 494, 443, 311 it/s |
| Graph replay | ~98.6%-99.5% |
| Bucket | 256 (trials 1-4), 256→fallback (trial 5) |

> **Note:** Trial 5 hit a CUDA Graph bucket boundary mid-sequence and fell back to the static (non-graph) path
> (`Graph write position at bucket boundary, falling back to static path`), yielding 311 it/s. This is a real
> edge case when a sequence length reaches the end of a bucket window. Median excluding the fallback trial: **463 it/s**.
> The fallback is non-crashing and produces correct output; it only affects throughput for that sequence.

### BigVGAN Raw

| mel_T | Median | Min | Max |
|------:|-------:|----:|----:|
| 70 | 28.2ms | 27.7ms | 28.7ms |
| 128 | 31.0ms | 30.4ms | 32.4ms |
| 298 | 52.9ms | 52.1ms | 54.1ms |
| 598 | 86.3ms | 85.3ms | 87.5ms |

Settings:

```text
use_cuda_kernel=True
precision=FP16
num_mels=100
warmup=10 forward passes per size
repeats=20 (torch.cuda.synchronize() before + after each)
```

## Notes

- All numbers are **warm cache** (CUDA Graph pre-captured, BigVGAN kernel pre-compiled, GPU allocator primed by 2-text warmup). Cold start adds ~108s for BigVGAN kernel compilation on first run.
- TTFP, full generation latency, T2S throughput, and BigVGAN raw timing are separate metrics and should not be compared as the same latency.
- This result uses a v3 LoRA asset set and should be compared with runs using the same model family and reference audio.
- T2S trial 5 triggered a CUDA Graph bucket boundary fallback (311 it/s); documented above. Normal steady-state throughput is 443-494 it/s.

## Run-to-Run Stability (vs 2026-05-24)

| Metric | 2026-05-24 | 2026-05-25 | Delta |
|--------|------------|------------|-------|
| TTFP short | 262.2ms | 256.7ms | −5.5ms |
| TTFP medium | 317.5ms | 301.4ms | −16.1ms |
| TTFP long | 408.9ms | 404.4ms | −4.5ms |
| T2S median (all trials) | 469 it/s | 449 it/s | −20 it/s |
| T2S median (excl. fallback) | 469 it/s | 463 it/s | −6 it/s |
| BigVGAN mel_T=70 | 27.1ms | 28.2ms | +1.1ms |
| BigVGAN mel_T=598 | 82.7ms | 86.3ms | +3.6ms |
