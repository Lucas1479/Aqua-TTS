# Deterministic T2S AR benchmark — upstream overlay

Measured 2026-08-30 after Aqua moved from a vendored T2S model fork to an
in-memory optimization overlay on upstream GPT-SoVITS.

## Environment

| Component | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER (16 GB) |
| OS | Windows 11 |
| Python | 3.12 |
| PyTorch | 2.5.1+cu124 |
| CUDA runtime | 12.4 |
| FlashAttention2 | 2.7.0.post2 |
| Upstream GPT-SoVITS | `08d627c3338173c3229286d8787060d6559fe0f8` |
| GPT checkpoint | v3, 24 layers, 512 hidden, 16 heads, fp16 |

## Protocol

- `torch.cuda.synchronize()` immediately before and after every measured call.
- Fixed input shapes and deterministic early-stop boundaries; sampled EOS does
  not decide run length.
- One cold conversation-shape call, then 15 warmup calls to stabilize allocator,
  kernels and GPU P-state.
- Seven measured repeats per shape; median synchronized throughput reported.
- CUDA Graph replay synchronization disabled (`CUDA_GRAPH_REPLAY_SYNC` unset).
- Each engine/FlashAttention configuration ran in an isolated process.

The shapes intentionally route across the three conversational buckets:

| Case | Initial KV length | Bucket | Reported AR steps |
|---|---:|---:|---:|
| short | 308 | 448 | 51 |
| conversation | 377 | 512 | 123 |
| long | 448 | 768 | 180 |

## Results

Synchronized throughput, median it/s:

| Engine | Cold conversation | Short / 448 | Conversation / 512 | Long / 768 |
|---|---:|---:|---:|---:|
| Current upstream, unpatched | 106.8 | 145.5 | 153.5 | 156.0 |
| Aqua CUDA Graph, SDPA fallback | 346.7 | 490.0 | 519.3 | 476.9 |
| Aqua CUDA Graph, FA2 `valid` (runtime default when available) | 398.6 | 568.5 | 627.4 | 644.9 |

Aqua's SDPA fallback was 3.1–3.4x faster than the current upstream decoder.
FlashAttention2 added approximately 16% at bucket 448, 21% at bucket 512 and
35% at bucket 768 relative to the SDPA run.

## Warm TTFP result

The full caller-visible benchmark used the same Aqua text, SoVITS and BigVGAN
pipeline for every column so only T2S execution changes. It used a cached
ABI-matching BigVGAN CUDA extension, 0.25-second streaming chunks, two warmup
utterances, and five repeats per case:

| T2S execution | Short / 3 chars | Medium / 19 chars | Long / 64 chars |
|---|---:|---:|---:|
| Current upstream dynamic path | 416.8 ms | 692.7 ms | 1135.2 ms |
| Aqua Graph + SDPA | 250.5 ms | 305.0 ms | 394.2 ms |
| Aqua Graph + FA2 `valid` | **233.1 ms** | **287.7 ms** | **348.3 ms** |

FA2 raw repeats (ms):

- Short: 259.8, 223.4, 242.0, 213.4, 233.1.
- Medium: 333.7, 287.7, 280.7, 295.1, 276.2.
- Long: 3059.3, 342.1, 348.3, 343.9, 369.1.

The 3059.3 ms first long repeat includes a one-time text-frontend
initialization for that shape. It is retained as cold-shape evidence; the next
four long repeats were 342.1–369.1 ms.

## Interpretation

The earlier June 2026 FlashAttention A/B was measured before CUDA Graph replay
synchronization became opt-in. The device-wide synchronization was a large
per-token cost shared by both SDPA and FlashAttention2, so it hid part of the
attention-kernel difference. Once removed, FA2's valid-length KV reads are much
more visible, especially at bucket 768.

The runtime now prefers FA2 `valid` when the optional package is importable.
Missing or rejected FA2 kernels fall back to SDPA, and
`AQUATTS_T2S_FLASH_ATTN=0` forces the fallback for continuity checks.

Windows GPU P-state and desktop scheduling can move absolute SDPA throughput
between processes. That is why the benchmark records cold and steady-state
numbers separately, uses 15 warmups and reports per-run ranges in its JSON
output. Relative conclusions should be reproduced on the target deployment GPU.

## Commands

```powershell
python benchmarks/t2s_speed_bench.py `
  --upstream-home F:/path/to/GPT-SoVITS `
  --gpt-model F:/path/to/xxx-e15.ckpt `
  --engine aqua --flash-attn off

python benchmarks/t2s_speed_bench.py `
  --upstream-home F:/path/to/GPT-SoVITS `
  --gpt-model F:/path/to/xxx-e15.ckpt `
  --engine aqua --flash-attn valid

python benchmarks/t2s_comparison_bench.py `
  --upstream-home F:/path/to/GPT-SoVITS `
  --gpt-model F:/path/to/xxx-e15.ckpt `
  --flash-ab
```
