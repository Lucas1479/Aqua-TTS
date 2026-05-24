# Aqua-TTS — Technical Deep Dive

## Architecture

Aqua-TTS is an optimization layer between GPT-SoVITS's AR Text-to-Semantic decoder and its BigVGAN vocoder. It does not replace any model weights — it replaces the *execution strategy*.

### Pipeline

```
Text → BERT → T2S AR Decoder → Speech Tokens → SoVITS CFM → Mel Spec → BigVGAN → Audio
                    ↑                                                           ↑
            [Aqua patch]                                  [Aqua CUDA kernel]
         Static KV + CUDA Graph                               Pre-compiled .pyd cache
```

## Static KV Cache

### Problem

The original GPT-SoVITS T2S decoder uses `torch.cat` to append each new token's KV to the cache:

```python
k_cache = torch.cat([k_cache, k], dim=1)  # shape changes every step
```

This has two problems:
1. **Dynamic shapes** — tensor dimensions change each step, making CUDA Graph capture impossible.
2. **Unbounded memory** — cache grows without limit, causing OOM on long sequences.

### Solution

`T2SBlockWithStaticCache` uses `scatter_` to write into a **fixed-size pre-allocated buffer**:

```python
k_cache.scatter_(1, pos_idx, k)  # shape stays [B, bucket_size, hidden]
```

- Buffer size is fixed at capture time (one of 6 bucket sizes).
- `pos_idx` is a persistent GPU tensor updated via `fill_()` outside the graph.
- Attention sees the full bucket — unwritten positions are 0, yielding near-zero softmax weight.
- If generation hits the bucket boundary, a sliding window keeps the most recent tokens.

## Bucketed CUDA Graph

### Why Buckets?

A CUDA Graph captures a specific tensor shape. Different prompt lengths produce different initial KV cache lengths, which would require a separate graph for each length. Instead, we:

1. Round up the prompt KV length to the nearest stride (32) boundary.
2. Select the smallest bucket that fits `aligned_kv + 96 generation slots`.
3. Capture one graph per `(bucket_size, initial_len)` pair.

### Bucket Design

| Bucket | Use case |
|--------|----------|
| 128 | Very short prompts (< 32 tokens) |
| 256 | Short turns (~64-128 tokens) |
| 448 | Single sentence (~256-352 tokens) |
| 512 | Typical single sentence (~330-416 tokens) |
| 768 | Multi-sentence merge (~420-672 tokens) |
| 1024 | Long compound turns |

### Pre-capture Strategy

At model load time, all viable `(bucket, initial_len)` pairs are enumerated and captured eagerly. The initial_len range per bucket is computed as:

```python
lo = max(stride * 6, prev_bucket - gap)   # stride*6 = 192, covers initial_len=224
hi = bucket - gap
```

This produces **13 graphs** across the 6 buckets, covering all initial_len values reachable from typical prompt lengths. The lower bound of `stride*6` (192) was chosen to include `initial_len=224`, which arises from short single-sentence prompts — avoiding expensive on-the-fly capture during the first inference.

### Thread Safety

Each `(bucket, initial_len)` pair gets its own `threading.Lock`. Multi-threaded servers (e.g., FastAPI with multiple workers) can safely replay different graphs concurrently. Only threads hitting the same bucket+initial_len key serialize.

### Graceful Degradation

```
CUDA Graph replay → (if fails) → static KV path → (if fails) → dynamic torch.cat
```

## Structured Logging

All performance-critical decisions are logged with structured prefixes for grep-friendly debugging:

| Prefix | What it logs |
|--------|-------------|
| `[precapture]` | Target buckets, per-bucket results, total graph count, graph keys |
| `[graph]` | Per-step bucket selection, graph key, replay/fallback decisions |
| `[graph] EOS stats` | End-of-sequence summary: bucket, graph_key, total_steps, replay_pct, bucket_misses, fallback reason |
| `[sovits-timing]` | CFM duration, BigVGAN duration (CFM timing gated by `TTS_STREAM_SYNC_TIMING`) |
| `[tts-stream]` | Per-chunk streaming info: text, token count, elapsed, speed factor, audio len |

The `[graph]` log at each decode step follows this format:

```
[graph] bucket=X key=Y decision=Z reason=W initial_len=N aligned_len=M total_len=O
```

At end-of-sequence, a single summary line aggregates all stats:

```
[graph] EOS stats: bucket=512 graph_key=(512,288) total_steps=187 replay_steps=186 replay_pct=99.5% bucket_misses=0 fallback=none
```

## BigVGAN Pre-compiled CUDA Kernel

### Problem

NVIDIA's official BigVGAN uses `torch.utils.cpp_extension.load()` which compiles the fused anti-alias activation CUDA kernel on first use. This requires Ninja and a C++ compiler at inference time, adding ~2-3s of startup latency.

### Solution

`aqua.bigvgan.cuda.load` provides:

1. **MSVC auto-discovery** — scans for Visual Studio via `vswhere.exe` on Windows.
2. **Per-GPU cache** — compiled `.pyd` files are cached under `build_smXX_cudaXX/` directories, keyed by compute capability and CUDA version.
3. **Lazy imports** — `torch` and `cpp_extension` are imported inside function bodies, so the package can be structurally inspected without a GPU.

### Cache Structure

```
bigvgan/cuda/
├── build_sm89_16gb_nvidia_geforce_rtx_4070_ti_super/   # Per-GPU cache
│   └── anti_alias_activation_cuda.pyd
├── build_sm86_cuda118/                                   # RTX 30-series
│   └── anti_alias_activation_cuda.pyd
├── anti_alias_activation.cpp, .cu, .h  # Source files
└── load.py                            # MSVC auto-discovery + loader

bigvgan/torch/
├── resample.py    # UpSample1d / DownSample1d (Kaiser-windowed)
├── filter.py      # LowPassFilter1d (sinc + Kaiser window)
└── act.py         # Activation1d (non-fused PyTorch fallback)
```

### Third-party components

- **NVIDIA BigVGAN**: CUDA kernel sources (`*.cpp, *.cu, *.h`) under Apache 2.0 — see [NOTICE](NOTICE).
- **alias-free-torch**: `torch/resample.py`, `torch/filter.py`, `torch/act.py` adapted from [alias-free-torch](https://github.com/junjun3518/alias-free-torch) under Apache 2.0 — see [NOTICE](NOTICE).

## Performance Analysis

Measured on an NVIDIA GeForce RTX 4070 Ti SUPER (16 GB VRAM), Windows 11, PyTorch 2.1.2+cu121.

### T2S Decoding Speed

The AR decoder is memory-bound on attention — each step reads the full KV cache. The static cache with CUDA Graph eliminates:

- `torch.cat` memory allocation overhead
- Python interpreter dispatch overhead
- CUDA kernel launch overhead
- Shape-inference overhead

Measured throughput: **440-470 iterations/second** (median) — a **5.5x speedup** over the official `torch.cat` KV cache baseline (~80-90 it/s) and **2x faster** than the official CUDA Graph implementation (~230 it/s). CUDA Graph replay hit rate is typically **98-99.5%** across bucket sizes.

| Variant | T2S Speed | KV Cache | CUDA Graph |
|---------|-----------|----------|------------|
| Official (no Graph) | ~80-90 it/s | `torch.cat` | None |
| Official (CUDA Graph) | ~230 it/s | Static `scatter_` | Single graph, lazy |
| Aqua | **~440-470 it/s** | Static `scatter_` | 13 graphs, pre-captured |

*All measured with the same 24-layer GPT checkpoint on RTX 4070 Ti SUPER via `benchmarks/t2s_comparison_bench.py`.*

Key reasons Aqua outperforms the official CUDA Graph implementation:
- **No `empty_cache` in hot path** — official calls `torch.cuda.empty_cache()` every 100 decode steps and at end-of-sequence, flushing the CUDA allocator cache
- **Pre-captured graphs** — 13 graphs captured at load time vs official's single lazy-captured graph
- **Bucketed sizing** — bucket selection based on initial_len provides near-optimal graph reuse
- **No NestedTensor overhead** — Aqua uses regular tensors throughout, avoiding the prototype API overhead of `torch.nested`

### BigVGAN Kernel (Raw)

Standalone forward-pass timing with `torch.cuda.synchronize()` before and after each measurement, 10 warmup + 20 measured per mel_T size (FP16):

| mel_T | median | min | max |
|-------|--------|-----|-----|
| 70 | 27ms | 26ms | 30ms |
| 128 | 31ms | 30ms | 34ms |
| 298 | 51ms | 50ms | 55ms |
| 598 | 82ms | 80ms | 87ms |

These are the pure BigVGAN kernel costs after the CFM generates the mel spectrogram — they do not include CFM time or any auxiliary work.

### TTFP (Time-To-First-Packet)

Measured with streaming audio output (2 chunks per utterance). All timings exclude `torch.cuda.empty_cache()` which was found to add ~500ms of GPU cache rebuild latency:

| text length | chars | TTFP (median) | total (median) |
|------------|-------|--------------|---------------|
| short      | 3     | 456 ms       | 456 ms        |
| medium     | 19    | 484 ms       | 484 ms        |
| long       | 64    | 499 ms       | 499 ms        |

Model load time: ~10 s (includes BigVGAN CUDA kernel load from pre-compiled cache, plus CUDA Graph pre-capture of 13 bucket/initial_len pairs at ~0.25 s each).

The BigVGAN CUDA pre-compiled kernel eliminates:

- PyTorch JIT compilation of the upsampling + activation + downsampling chain (~800ms cold)
- Python-side filter kernel launches (~200ms)
- CPU-GPU synchronization points in the alias-free activation path (~300ms)

### Keep-Warm

After model load, a lightweight warmup pass primes all GPU execution paths before the first user request:

- **BigVGAN shape warmup**: 3 mel-T sizes (40, 70, 128) × 5 iterations each with `torch.cuda.synchronize()` — covers the range of first-chunk mel sizes.
- **T2S graph pre-capture**: all 13 bucket/initial_len pairs are captured eagerly, ensuring the first `infer_stream()` call hits a pre-warmed CUDA Graph with zero cold-start overhead.

This prevents the common "first request penalty" where CUDA lazy initialization, cuDNN autotuning, and kernel compilation would otherwise add 200-500ms to the first inference.

## Environment Variables

| Variable | Default | Effect |
|----------|---------|--------|
| `ENABLE_CUDA_GRAPH` | `1` | Enable CUDA Graph for T2S decode steps |
| `ENABLE_CUDA_GRAPH_PRECAPTURE` | `1` | Pre-capture all bucket graphs at model load |
| `CUDA_GRAPH_PRECAPTURE_BUCKETS` | (all) | Comma-separated bucket sizes to pre-capture |
| `TTS_STREAM_SYNC_TIMING` | `0` | Enable per-step CFM timing (adds GPU sync overhead) |
| `TORCH_CUDA_ARCH_LIST` | `""` | CUDA arch list (set by loader, not user) |
