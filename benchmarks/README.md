# Aqua benchmark suite

Reproducible comparisons of Aqua and the selected upstream GPT-SoVITS
checkout. Benchmark commands require `GPT_SOVITS_HOME` or an explicit
`--upstream-home`; Aqua no longer benchmarks a vendored T2S model fork.

## T2S AR throughput

Run the default Aqua CUDA Graph + SDPA path:

```bash
python benchmarks/t2s_speed_bench.py \
  --upstream-home /path/to/GPT-SoVITS \
  --gpt-model /path/to/xxx-e15.ckpt
```

Run the optional FlashAttention2 path:

```bash
python benchmarks/t2s_speed_bench.py \
  --upstream-home /path/to/GPT-SoVITS \
  --gpt-model /path/to/xxx-e15.ckpt \
  --flash-attn valid
```

Compare current upstream, Aqua static KV, Aqua CUDA Graph, and optionally FA2
in isolated subprocesses:

```bash
python benchmarks/t2s_comparison_bench.py \
  --upstream-home /path/to/GPT-SoVITS \
  --gpt-model /path/to/xxx-e15.ckpt \
  --flash-ab
```

The authoritative number is `sync`, not tqdm's smoothed display rate. The
benchmark disables progress rendering, synchronizes CUDA before and after each
call, and also reports caller-visible `wall` throughput as a diagnostic.

### Fixed cases

Run length never depends on a sampled EOS token. Each shape has a deterministic
early-stop boundary and maps to a known static-KV bucket.

| Case | Initial KV length | Bucket | Reported AR steps | Purpose |
|---|---:|---:|---:|---|
| cold/conversation | 377 | 512 | 123 | First host-loop use after graph capture |
| short | 308 | 448 | 51 | Short conversational response |
| conversation | 377 | 512 | 123 | Reproduces the Amadeus 500+ it/s trace shape |
| long | 448 | 768 | 180 | Exposes long-bucket attention cost |

### Measurement protocol

1. Load the same fp16 GPT checkpoint for every engine.
2. Pre-capture only buckets 448, 512 and 768 for the Aqua Graph engines.
3. Measure one cold conversation-shape call.
4. Run 15 conversation-shape warmups. Five warmups were not sufficient to
   stabilize the RTX 4070 Ti SUPER's Windows P-state in repeated testing.
5. Measure seven repeats per steady-state shape and report the median.
6. Keep `CUDA_GRAPH_REPLAY_SYNC` unset. A device-wide replay sync is available
   only for graph diagnostics and is not part of production behavior.
7. Run comparison engines in isolated subprocesses to prevent `sys.modules`
   contamination between upstream and Aqua overlays.

JSON evidence can be saved with `--json-output result.json`.

## Current T2S result

RTX 4070 Ti SUPER, Windows 11, Python 3.12, PyTorch 2.5.1+cu124,
upstream GPT-SoVITS `08d627c`, seven-repeat median:

| Engine | Cold | Short / 448 | Conversation / 512 | Long / 768 |
|---|---:|---:|---:|---:|
| Current upstream | 106.8 | 145.5 | 153.5 | 156.0 |
| Aqua Graph + SDPA (default) | 346.7 | 490.0 | 519.3 | 476.9 |
| Aqua Graph + FA2 `valid` | 398.6 | 568.5 | 627.4 | 644.9 |

See the [raw methodology and interpretation](results/4070ti-super-win-py312-torch251-cu124-upstream-overlay.md).

The old June 2026 claim that FlashAttention2 had no short-case benefit and only
about 8% long-case benefit predates the removal of the per-token CUDA Graph
replay synchronization. It is historical, not the current recommendation.
FlashAttention2 remains opt-in because dependency and semantic/audio regression
coverage are separate from throughput.

Absolute rates are sensitive to Windows GPU P-state and desktop scheduling.
Use the JSON trial range and reproduce on the deployment GPU before treating a
percentage as portable.

## TTFP (time to first playable audio)

```bash
python benchmarks/aqua_ttfp.py \
  --gpt-model /path/to/xxx-e15.ckpt \
  --sovits-model /path/to/xxx_e2_s174_l32.pth \
  --ref-audio /path/to/ref_audio.wav \
  --ref-text "reference transcript"
```

TTFP measures text-ready to first non-empty audio returned by `infer_stream()`.
Use two or more warmup texts, five repeats per text, and report the median. Do
not place `torch.cuda.empty_cache()` in the hot path. TTFP intentionally does
not add CUDA synchronization inside its timing window because it measures the
latency visible to the caller.

## BigVGAN raw kernel timing

```bash
python benchmarks/bigvgan_raw_bench.py
```

The raw benchmark measures only the fp16 BigVGAN forward pass. Warm each mel
shape first, synchronize immediately around measured calls, and report medians
for mel lengths representative of first chunks and full utterances.
