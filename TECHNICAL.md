# Spectralis-TTS — Technical Deep Dive

## Architecture

Spectralis-TTS is an optimization layer between GPT-SoVITS's AR Text-to-Semantic decoder and its BigVGAN vocoder. It does not replace any model weights — it replaces the *execution strategy*.

### Pipeline

```
Text → BERT → T2S AR Decoder → Speech Tokens → SoVITS CFM → Mel Spec → BigVGAN → Audio
                    ↑                                                           ↑
            [Spectralis patch]                                  [Spectralis CUDA kernel]
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

### Thread Safety

Each `(bucket, initial_len)` pair gets its own `threading.Lock`. Multi-threaded servers (e.g., FastAPI with multiple workers) can safely replay different graphs concurrently. Only threads hitting the same bucket+initial_len key serialize.

### Graceful Degradation

```
CUDA Graph replay → (if fails) → static KV path → (if fails) → dynamic torch.cat
```

## BigVGAN Pre-compiled CUDA Kernel

### Problem

NVIDIA's official BigVGAN uses `torch.utils.cpp_extension.load()` which compiles the fused anti-alias activation CUDA kernel on first use. This requires Ninja and a C++ compiler at inference time, adding ~2-3s of startup latency.

### Solution

`spectralis.bigvgan.cuda.load` provides:

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

Measured throughput: **462 iterations/second** (median), with CUDA Graph replay at **99.6%** hit rate on bucket 768.

### TTFP (Time-To-First-Packet)

Measured with streaming audio output (2 chunks per utterance):

| text length | chars | TTFP (median) | total (median) |
|------------|-------|--------------|---------------|
| short      | 3     | ~520 ms      | ~520 ms       |
| medium     | 19    | ~670 ms      | ~670 ms       |
| long       | 64    | ~1,300 ms    | ~1,300 ms     |

Model load time: ~10 s (includes BigVGAN CUDA kernel load from pre-compiled cache, plus CUDA Graph pre-capture of 11 bucket/initial_len pairs at ~0.25 s each).

The BigVGAN CUDA pre-compiled kernel eliminates:

- PyTorch JIT compilation of the upsampling + activation + downsampling chain (~800ms cold)
- Python-side filter kernel launches (~200ms)
- CPU-GPU synchronization points in the alias-free activation path (~300ms)

## Environment Variables

| Variable | Default | Effect |
|----------|---------|--------|
| `ENABLE_CUDA_GRAPH` | `1` | Enable CUDA Graph for T2S decode steps |
| `ENABLE_CUDA_GRAPH_PRECAPTURE` | `1` | Pre-capture all bucket graphs at model load |
| `CUDA_GRAPH_PRECAPTURE_BUCKETS` | (all) | Comma-separated bucket sizes to pre-capture |
| `TORCH_CUDA_ARCH_LIST` | `""` | CUDA arch list (set by loader, not user) |
