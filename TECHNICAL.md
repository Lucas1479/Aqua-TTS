# Aqua-TTS — Technical Deep Dive

## Architecture

Aqua-TTS is an optimization layer between GPT-SoVITS's AR Text-to-Semantic decoder and its BigVGAN vocoder. It does not replace any model weights — it replaces the *execution strategy*.

Since 0.2.1, the model definition is loaded from the checkout selected by
`GPT_SOVITS_HOME`; Aqua no longer distributes a forked `t2s_model.py`. The
runtime validates the upstream decoder shape, builds the static-cache blocks
from its loaded weights, and replaces only `infer_panel()`. Upstream
`infer_panel_naive()` remains a generator for native batching and streaming.

### Pipeline

```
Text → BERT → T2S AR Decoder → Speech Tokens → SoVITS CFM → Mel Spec → BigVGAN → Audio
                    ↑                                                           ↑
            [Aqua patch]                                  [Aqua CUDA kernel]
         Static KV + CUDA Graph                               Pre-compiled .pyd cache
```

## Optional model lifecycle

The BERT tokenizer/model begin unset. Chinese feature extraction enters a
double-checked per-instance lock and builds both components locally, including
dtype/device conversion and eval mode, before publishing them together. Failed
loads leave the state unset and can retry. Explicit preload uses the same entry.
The language frontend selects Chinese segments for both target and reference text;
English/Japanese zero BERT features do not invoke the loader. AP-BWE import and
construction similarly happen only at the requested super-resolution boundary.

## Reference cache lifecycle

Warmup and both inference APIs normalize reference paths, language labels and
terminal punctuation before lookup. Keys include file stat identity, normalized
conditioning, model generation and actual acoustic-model dtype/device. Successful
GPT/SoVITS replacement and explicit clearing invalidate the cache. Construction
is serialized and coordinated with acoustic decoding; incomplete or failed builds
are not published. FIFO eviction bounds residency, and CUDA completion events order
cross-stream hits without a normal-hit device synchronization. Explicit warmup
synchronizes the selected device and checks residency before reporting ready.

Version-specific validation requires semantic fields only when used, Pro speaker
features for Pro models, and V3 mel/reference-encoder features for V3. Existing
mutable reference-feature cloning remains in both V3 paths. Ordinary V3 cache hits
also skip redundant reference audio loading, matching the streaming path.

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
- A persistent validity mask excludes every unwritten slot and Graph-alignment
  gap. Zero KV is not padding: its zero logit would still receive softmax
  weight.
- Without Graph, SDPA slices directly to the known contiguous valid prefix.
- FlashAttention2 `valid` and `bucket` update modes both expose only the real
  contiguous prefix; the latter keeps its explicit scatter mechanism but no
  longer attends the padded allocation.
- If generation fills a bucket, Aqua preserves the complete prompt/history and
  continues with dynamically allocated KV instead of dropping conditioning.

## Bucketed CUDA Graph

### Why Buckets?

A CUDA Graph captures a specific tensor shape. The bucket fixes the computation
shape; actual prefix length remains mutable GPU position/mask data. We:

1. Round up the prompt KV length to the nearest stride (32) boundary.
2. Select the smallest bucket that fits `aligned_kv + 96 generation slots`.
3. Capture one graph per `bucket_size`, independent of actual initial length.

Alignment is used for routing only. Persistent `pos_idx` and
the validity mask are reset from the real prompt length for each sentence.
Warmup/capture tensors use fixed non-zero constants so graph creation does not
advance the caller's sampling RNG.

### Bucket Design

| Bucket | Use case |
|--------|----------|
| 128 | Configured capacity; not selected for a positive prefix under the current reserve rule |
| 256 | Initial KV 1–128 |
| 448 | Initial KV 129–320 |
| 512 | Initial KV 321–384 |
| 768 | Initial KV 385–640 |
| 1024 | Initial KV 641–896 |

Initial KV includes text and reference semantic tokens, not character count.
The strict routing inequality rejects initial KV >=897 with these settings.

### Pre-capture Strategy

The resolved full/minimal preset selects preparation targets; each target gets
at most one Graph, using synthetic inputs. Full prepares six graphs and minimal
prepares 256/512/1024. Lazy skips startup capture and off skips Graph entirely.
Explicit target lists are not expanded automatically. The direct preparation
API treats `None` as all available buckets and an empty list as no targets.

`bucket` storage gives each prepared bucket independent K/V. Optional `nested`
stores K/V in roots shaped `[layers, 1, max_capacity, hidden]`; per-layer prefix
views keep the short bucket computation shape and stable captured addresses.
The default remains bucket. Layout, device, dtype or capacity changes require a
new decoder. At 24 layers/hidden512/FP16, full bucket uses 147 MiB unique KV and
nested uses 48 MiB; a single 448 bucket instead uses 21 MiB versus 48 MiB.

### Prepared-only Promotion

When a nested view fills, select the smallest larger captured bucket. Its KV views
alias the same roots, so valid history is neither copied nor reallocated. Clear
only `[old_capacity:new_capacity)` (including nonfinite stale data), reset the
target validity mask and position, then replay its existing graph. Uncaptured
targets are skipped; sparse sets may jump directly from 256 to 1024. Initial lazy
capture is permitted before prompt installation, but promotion never captures on
live KV. No target, maximum overflow, or bucket storage uses full-prefix private
dynamic KV. Replay failure uses eager static decode before eventual fallback.

Request-local counters include owner wait, initial/final KV, start/final bucket,
promotions, capture attempts and per-path steps; published stats are the last
completed request. `total_steps` includes prefill and `generated_tokens` preserves
the existing return convention. Promotion does not change token/EOS policy.

### Thread Safety

Each bucket storage owner holds a reentrant lease for the complete AR request,
including prompt installation and sampling. Nested layouts share one root
owner across all buckets. Completion events order ownership transfer between
CUDA streams. Different independent owners may progress concurrently, while
capture itself is serialized process-wide. Graph-disabled or batch>1 requests
do not enter the shared batch-one workspace.

CUDA stream dependencies order graph replay before sampling, so Aqua does not
perform a device-wide synchronization after every token. Set
`CUDA_GRAPH_REPLAY_SYNC=1` only when diagnosing graph failures. Greedy and
sampled EOS conditions are combined before the single device-to-host check.

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
[graph] EOS stats: bucket=512 graph_key=512 total_steps=187 replay_steps=186 replay_pct=99.5% bucket_misses=0 fallback=none
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

Measured on an NVIDIA GeForce RTX 4070 Ti SUPER (16 GB VRAM), Windows 11,
Python 3.12, PyTorch 2.5.1+cu124, against upstream GPT-SoVITS `08d627c`.

### T2S Decoding Speed

The AR decoder is memory-bound on attention — each step reads the full KV cache. The static cache with CUDA Graph eliminates:

- `torch.cat` memory allocation overhead
- Python interpreter dispatch overhead
- CUDA kernel launch overhead
- Shape-inference overhead

The current benchmark fixes the input length, bucket and reported AR steps,
synchronizes CUDA around every measured call, performs 15 P-state warmups and
reports the median of seven repeats:

| Variant | Short / 448 | Conversation / 512 | Long / 768 | KV Cache |
|---------|------------:|-------------------:|-----------:|----------|
| Current upstream | 145.5 it/s | 153.5 it/s | 156.0 it/s | Dynamic `torch.cat` |
| Aqua Graph + SDPA | **490.0 it/s** | **519.3 it/s** | **476.9 it/s** | Static `scatter_` |
| Aqua Graph + FA2 `valid` | **568.5 it/s** | **627.4 it/s** | **644.9 it/s** | Valid-length FlashAttention2 |

CUDA Graph replay coverage was 98.1–99.4% across these shapes. Absolute rates
remain sensitive to Windows GPU P-state; raw ranges and cold measurements are
recorded in `benchmarks/results/4070ti-super-win-py312-torch251-cu124-upstream-overlay.md`.

Key reasons Aqua outperforms the current upstream path:

- **Static KV buffers** — no per-step `torch.cat` growth or allocation.
- **Pre-captured graphs** — each prepared computation bucket is reusable across initial lengths.
- **No device-wide replay sync in production** — stream ordering handles graph replay; `CUDA_GRAPH_REPLAY_SYNC=1` is diagnostic-only.
- **One EOS synchronization** — greedy and sampled EOS conditions are combined into one device-to-host read.
- **Bucketed sizing** — selection based on aligned initial length reserves generation space without routing ordinary conversation shapes to an unnecessarily large bucket.

The no-replay-sync change also alters the FlashAttention2 trade-off. The older
June 2026 result (no short benefit, about 8% on the long text) included a large
shared synchronization cost. Once that cost is removed, valid-length FA2 reads
showed approximately 16%, 21% and 35% directional gains at buckets 448, 512 and
768 in this run. Aqua now prefers FA2 `valid` automatically when the package is
importable and falls back to SDPA when it is absent or rejects a runtime shape.
Set `AQUATTS_T2S_FLASH_ATTN=0` for explicit SDPA continuity.

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

These historical tables measure the first nonempty PCM chunk and predate
generated-lead trimming. They are not first-voiced-sample measurements. Onset
processing retains a 50 ms preroll before a -45 dBFS / 10 ms RMS window, defers
partial streaming frames until more samples or EOF, and leaves internal and
caller-added pauses unchanged. Playback demo onset fields describe PCM position
and chunk submission readiness, not acoustic speaker timing.

Measured with the automatic FA2 `valid` path, a matching cached BigVGAN CUDA
extension and 0.25-second streaming chunks. Two utterances warm the pipeline;
the table reports the median of five first-playable-chunk measurements:

| T2S execution | Short / 3 chars | Medium / 19 chars | Long / 64 chars |
|---|---:|---:|---:|
| Current upstream dynamic path | 416.8 ms | 692.7 ms | 1135.2 ms |
| Aqua Graph + SDPA | 250.5 ms | 305.0 ms | 394.2 ms |
| Aqua Graph + FA2 `valid` | **233.1 ms** | **287.7 ms** | **348.3 ms** |

All three TTFP rows use the same Aqua text, SoVITS and BigVGAN pipeline; only
the T2S execution mode changes.

The first long-text repeat took about 3 seconds because that text shape triggered
one-time frontend initialization; the following four repeats were 342–369 ms.
It is retained as cold-shape evidence and does not change the five-repeat median.

Model load time was 10.21 s, including BigVGAN CUDA cache loading and CUDA Graph
pre-capture of 15 common bucket/initial-length pairs at roughly 0.25 s each.

The BigVGAN CUDA pre-compiled kernel eliminates:

- PyTorch JIT compilation of the upsampling + activation + downsampling chain (~800ms cold)
- Python-side filter kernel launches (~200ms)
- CPU-GPU synchronization points in the alias-free activation path (~300ms)

### Keep-Warm

After model load, a lightweight warmup pass primes all GPU execution paths before the first user request:

- **BigVGAN shape warmup**: 3 mel-T sizes (40, 70, 128) × 5 iterations each with `torch.cuda.synchronize()` — covers the range of first-chunk mel sizes.
- **T2S graph pre-capture**: 15 common bucket/initial-length pairs are captured eagerly; uncommon shapes retain lazy capture.

This prevents the common "first request penalty" where CUDA lazy initialization, cuDNN autotuning, and kernel compilation would otherwise add 200-500ms to the first inference.

## Environment Variables

| Variable | Default | Effect |
|----------|---------|--------|
| `ENABLE_CUDA_GRAPH` | `1` | Enable CUDA Graph for T2S decode steps |
| `ENABLE_CUDA_GRAPH_PRECAPTURE` | `1` | Pre-capture all bucket graphs at model load |
| `CUDA_GRAPH_PRECAPTURE_BUCKETS` | (all) | Comma-separated bucket sizes to pre-capture |
| `AQUATTS_SEMANTIC_GUARD` | `1` | Reject collapsed semantic candidates and retry once before vocoder work |
| `TTS_STREAM_SYNC_TIMING` | `0` | Enable per-step CFM timing (adds GPU sync overhead) |
| `CUDA_GRAPH_REPLAY_SYNC` | `0` | Force a device sync after each graph replay for diagnostics |
| `AQUATTS_T2S_FLASH_ATTN` | `auto` / unset | `0` forces SDPA; `1` explicitly requests FA2; `auto`, empty, or unset prefers FA2 when importable |
| `AQUATTS_T2S_FLASH_ATTN_MODE` | `valid` | FA2 reads the true KV length; `bucket` uses explicit scatter with the same safe valid prefix |
| `BIGVGAN_CACHE_ROOT` | package CUDA directory | Product-owned root for ABI-keyed compiled BigVGAN extensions |
| `TORCH_CUDA_ARCH_LIST` | `""` | CUDA arch list (set by loader, not user) |

## V2-family model integration

SoVITS architecture selection delegates to upstream checkpoint metadata. Text
symbols (`v1`/`v2`) stay separate from the `v2Pro`/`v2ProPlus` architecture tag;
unsupported architectures fail before loading models. Pro checkpoints must load
all inference parameters, including speaker projections; only training-only
`enc_q` discrepancies are ignored. Existing v3 LoRA key detection is retained.

`aquatts.inference.speaker.SpeakerEncoder` imports ERes2NetV2 and Kaldi features
from the configured upstream checkout and accepts an explicit weight path.
References are converted to 16 kHz mono and encoded using upstream's 80-bin
filterbank / `forward3` contract. The primary embedding shares the reference
session cache. Both inference APIs use `_decode_v2` to preserve spectrum/speaker
pairing, including failed extra references. Plain v1/v2 never receives `sv_emb`.

V2-family `infer_stream` decodes a complete text segment before chunking its PCM.
It does not use v3 CFM or BigVGAN and does not yet call upstream `decode_streaming`.
The server reads the initial sample-rate event in a worker thread before sending
headers, retaining first-event audio and closing the generator after streaming.
Model exceptions propagate to callers instead of becoming silent audio.

The change is limited to model support; it adds no MPS/ROCm device paths.
