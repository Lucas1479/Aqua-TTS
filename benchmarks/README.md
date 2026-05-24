# Aqua Benchmark Suite

Fair comparison of Aqua vs official GPT-SoVITS inference performance.

## Prerequisites

### 1. Clone official GPT-SoVITS

```bash
git clone https://github.com/RVC-Boss/GPT-SoVITS.git official-gpt-sovits
cd official-gpt-sovits
pip install -r requirements.txt
```

### 2. Prepare model weights

- GPT weights (T2S): `s1v3.ckpt` (official) or `xxx-e15.ckpt` (self-trained)
- SoVITS weights: `xxx_e2_s174_l32.pth` (with LoRA)
- Reference audio: 3-10 second .wav file

## Benchmarks

### T2S throughput comparison

```bash
# Run all three variants (official, official CUDA Graph, Aqua)
python benchmarks/t2s_comparison_bench.py --gpt-model GPT_weights_v3/xxx-e15.ckpt
```

### Aqua — All optimizations

```bash
python benchmarks/aqua_ttfp.py \
    --gpt-model GPT_weights_v3/xxx-e15.ckpt \
    --sovits-model SoVITS_weights_v3/xxx_e2_s174_l32.pth \
    --ref-audio "reference audio/ref_audio.wav" \
    --ref-text "reference transcript"
```

### Aqua — Ablation (disable optimizations)

```bash
# No CUDA Graph, no BigVGAN kernel
python benchmarks/aqua_ttfp.py ... --no-cuda-graph --no-bigvgan-kernel

# All off (equivalent to official path)
python benchmarks/aqua_ttfp.py ... --no-cuda-graph --no-static-kv --no-bigvgan-kernel
```

### T2S throughput (single variant)

```bash
python benchmarks/t2s_speed_bench.py --gpt-model GPT_weights_v3/xxx-e15.ckpt
```

### BigVGAN raw kernel timing

```bash
python benchmarks/bigvgan_raw_bench.py
```

## Methodology

### TTFP (Time-To-First-Playback)

TTFP measures **text-ready to first-playable-audio latency** — the wall-clock time from calling TTS inference to the first non-empty audio buffer that can be sent to playback. This is the latency a user experiences before hearing the first sound.

For non-streaming baselines, the first returned playable audio may coincide with full utterance completion. For Aqua-TTS, TTFP is the first streaming audio chunk returned by `infer_stream(text=...)`.

**Measurement protocol:**

1. **Warmup**: 2 texts (`"これはテストです。"`, `"何の実験だか気になるけど、まあいいわ。"`) run before any measurement. This primes CUDA Graph buckets, cuDNN autotuning, and GPU allocator state.
2. **Repeats**: 5 per benchmark text, **median** reported (not mean — resistant to outlier cold starts).
3. **Timing**: `time.perf_counter()` around `infer_stream()` loop. `torch.cuda.synchronize()` is NOT called inside the timing window — TTFP measures wall-clock latency as experienced by the user.
4. **No `empty_cache`**: `torch.cuda.empty_cache()` is never called in the hot path (was found to add ~500ms of GPU page-fault latency).
5. **Streaming mode**: `how_to_cut="按标点符号切"`, `chunk_size_seconds=0.25`, `speed=1.1`, `sample_steps=4`, `top_k=5`, `top_p=1`, `temperature=0.6`.
6. **First playable chunk only**: The default measurement stops at the first non-empty audio chunk (TTFP is not total duration). Use `--measure-total` when full generation latency is needed.

### T2S Throughput

T2S throughput measures the **raw AR decoder speed** in tokens per second. The benchmark calls `infer_panel_naive()` with random input tensors and measures wall-clock time per decode step.

**Measurement protocol:**

1. **Warmup**: 3 calls to `infer_panel_naive()` before timing.
2. **Repeats**: 5, **median** reported.
3. **Timing**: `torch.cuda.synchronize()` before AND after `infer_panel_naive()` — measures GPU execution only.
4. **Comparison**: All three variants (official `torch.cat` KV, official CUDA Graph, Aqua) tested in isolated subprocesses to avoid `sys.modules` contamination. Same GPT checkpoint, same input shapes.
5. **Official CUDA Graph note**: The official `t2s_model_cudagraph.py` calls `torch.cuda.empty_cache()` every 100 steps and at end-of-sequence, which impacts its measured throughput.

### BigVGAN Raw

Measures the **pure BigVGAN forward pass** time — no CFM, no streaming wrapper, no `empty_cache`.

**Measurement protocol:**

1. **Warmup**: 10 forward passes per mel-T size.
2. **Repeats**: 20, **median** reported.
3. **Timing**: `torch.cuda.synchronize()` before AND after each forward pass.
4. **Sizes**: mel_T ∈ {70, 128, 298, 598} — covers typical first-chunk to full-utterance mel sizes.
5. **Precision**: FP16 (`model.half().to("cuda")`).

### Hardware

| Component | Detail |
|-----------|--------|
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER |
| VRAM | 16 GB GDDR6X |
| Driver | 555.99 |
| OS | Windows 11 Pro (build 26200) |
| Python | 3.12 |
| PyTorch | 2.5.1+cu121 |
| CUDA | 12.1 |

### Model Weights

| File | Description |
|------|-------------|
| `xxx-e15.ckpt` | GPT T2S model (24 layers, 512 hidden dim, 16 heads) |
| `xxx_e2_s174_l32.pth` | SoVITS vocoder with LoRA adapter (32-rank) |
| `ref_audio.wav` | 3-10 second 24 kHz WAV reference audio with known transcript |

## Test Texts

| Case | Chars | Text |
|------|-------|------|
| short | 3 | 実験？ |
| medium | 19 | 何の実験だか気になるけど、まあいいわ。 |
| long | 64 | Paxosが提案ベースでメッセージフローが複雑なのに対して、Raftは明確なリーダー選出とログ複製のフェーズに分けられているわ。 |

## Results (RTX 4070 Ti SUPER, float16)

| Metric | Official (no Graph) | Official (CUDA Graph) | Aqua |
|--------|---------------------|-----------------------|------------|
| T2S throughput | ~80-90 it/s | ~230 it/s | **440-470 it/s** |
| TTFP Short | 1061ms | 1016ms | **~257ms** |
| TTFP Medium | 1599ms | 1476ms | **~301ms** |
| TTFP Long | 3598ms | 2852ms | **~404ms** |
| KV cache | Dynamic `torch.cat` | Static `scatter_` | **Static `scatter_`** |
| CUDA Graph | None | Single graph, lazy capture | **17 graphs, pre-captured** |
| BigVGAN | PyTorch JIT | PyTorch JIT | **Pre-compiled CUDA kernel** |
| GPU memory safety | OOM (long texts) | OOM (long texts) | **Bounded static buffers** |
| `empty_cache` | No | Every 100 steps | **Never in hot path** |
| V3 streaming | Batch only | Batch only | **True streaming** |

*T2S measured with `benchmarks/t2s_comparison_bench.py`. TTFP measured with `benchmarks/aqua_ttfp.py` on an NVIDIA GeForce RTX 4070 Ti SUPER (16 GB VRAM), Windows 11, PyTorch 2.5.1+cu121. All tests use the same model weights (xxx-e15.ckpt + xxx_e2_s174_l32.pth). T2S measured at 500-token target; TTFP is time to the first playable audio returned to the caller. For non-streaming baselines this may coincide with full utterance completion; for Aqua-TTS it is the first streaming chunk with `chunk_size_seconds=0.25`.*
