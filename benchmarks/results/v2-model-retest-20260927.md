# Fixed-seed steady-state retest — 2026-09-27

No inference implementation changes were made for this retest. Runtime `666cdce`,
upstream `08d627c`, Windows, Python 3.12.10, Torch 2.5.1+cu121, fp16, and the same
local v2/Pro/Plus checkpoint pairs and reference inputs as the previous report.
The selected physical GPU is the RTX 4070 Ti SUPER (16 GB), locked by UUID rather
than relying on NVIDIA/CUDA index order. Torch uses 8 intra-op and 8 inter-op threads.
No model downloads, application shutdowns, or GPU power-setting changes were needed.

## What changed in the measurement

- Each text now has one separately recorded first use, five warmups of that exact
  text/seed, twenty measured repetitions, and three separate diagnostic repetitions.
  Seed 20260926 is reset for every request. All 540 measured trials used stable
  per-engine semantic tokens; none captured another CUDA Graph. FA2 fallback was forbidden.
- Main measurements use the normal unsynchronized inference API: synchronized T2S
  statistics are off, and PCM validation/hashing occurs after the timer. Diagnostic
  runs separately synchronize and measure frontend, T2S and acoustic decoding.
- Same text segmentation, top-k/top-p/temperature, speed=1.1, 0.25 s chunks and
  0.3 s pauses as the previous comparison. RTF includes the generated pause audio.
- First use includes reference/frontend/shape initialization after model loading.
  Model-load time is recorded separately. First-use latency is not pure graph-capture
  cost and is not included in steady percentiles. P95 is nearest-rank over 20 samples.
- Total workload: 540 steady trials, 135 warmups, 27 first-use trials and 81 diagnostics.
  Jobs ran serially. These are model-side first-PCM timings, not speaker playback latency.

## Main result

V2 Pro + Graph/FA2 short-text latency is **96.4 ms median**, **93.1 ms minimum**,
and **110.3 ms p95**. This is close to the recalled 89 ms, but this run did not
reproduce 89 ms and no matching historical raw record was found in the bounded
local evidence search. The former 198.3 ms median and this 96.4 ms median use
different measurement protocols; the difference is not an implementation speedup.

| Model | T2S engine | Short p50 (ms) | Short p95 (ms) | Medium p50 (ms) | Long p50 (ms) |
|---|---|---:|---:|---:|---:|
| v2 | Upstream T2S | 255.7 | 293.6 | 519.5 | 1005.1 |
| v2 | Graph + SDPA | 107.5 | 116.4 | 177.1 | 306.0 |
| v2 | Graph + FA2 | 93.9 | 103.3 | 142.6 | 231.0 |
| v2Pro | Upstream T2S | 272.7 | 303.7 | 547.4 | 965.4 |
| v2Pro | Graph + SDPA | 110.9 | 122.5 | 183.4 | 293.0 |
| v2Pro | Graph + FA2 | 96.4 | 110.3 | 143.1 | 221.4 |
| v2ProPlus | Upstream T2S | 252.9 | 283.3 | 537.0 | 927.2 |
| v2ProPlus | Graph + SDPA | 107.0 | 119.8 | 180.2 | 291.9 |
| v2ProPlus | Graph + FA2 | 100.7 | 107.4 | 144.7 | 219.6 |

## Complete response statistics

| Model | Engine | Text | Min (ms) | p50 (ms) | p95 (ms) | Total p50 (ms) | Audio (s) | RTF p50 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| v2 | Upstream T2S | short | 241.4 | 255.7 | 293.6 | 255.9 | 1.26 | 0.203 |
| v2 | Upstream T2S | medium | 477.9 | 519.5 | 556.2 | 814.4 | 3.96 | 0.206 |
| v2 | Upstream T2S | long | 873.6 | 1005.1 | 1090.9 | 2068.5 | 9.86 | 0.210 |
| v2 | Graph + SDPA | short | 105.3 | 107.5 | 116.4 | 107.7 | 1.26 | 0.085 |
| v2 | Graph + SDPA | medium | 172.0 | 177.1 | 185.2 | 296.8 | 3.96 | 0.075 |
| v2 | Graph + SDPA | long | 290.5 | 306.0 | 314.3 | 653.5 | 9.86 | 0.066 |
| v2 | Graph + FA2 | short | 91.4 | 93.9 | 103.3 | 94.0 | 1.26 | 0.075 |
| v2 | Graph + FA2 | medium | 140.1 | 142.6 | 150.1 | 245.4 | 3.96 | 0.062 |
| v2 | Graph + FA2 | long | 223.5 | 231.0 | 241.6 | 474.9 | 9.86 | 0.048 |
| v2Pro | Upstream T2S | short | 250.8 | 272.7 | 303.7 | 272.8 | 1.32 | 0.207 |
| v2Pro | Upstream T2S | medium | 499.1 | 547.4 | 618.2 | 869.9 | 4.14 | 0.210 |
| v2Pro | Upstream T2S | long | 726.7 | 965.4 | 1072.3 | 1979.9 | 9.46 | 0.209 |
| v2Pro | Graph + SDPA | short | 109.0 | 110.9 | 122.5 | 111.0 | 1.32 | 0.084 |
| v2Pro | Graph + SDPA | medium | 176.6 | 183.4 | 192.6 | 308.0 | 4.14 | 0.074 |
| v2Pro | Graph + SDPA | long | 283.6 | 293.0 | 302.3 | 629.9 | 9.46 | 0.067 |
| v2Pro | Graph + FA2 | short | 93.1 | 96.4 | 110.3 | 96.5 | 1.32 | 0.073 |
| v2Pro | Graph + FA2 | medium | 140.1 | 143.1 | 154.4 | 248.6 | 4.14 | 0.060 |
| v2Pro | Graph + FA2 | long | 213.4 | 221.4 | 234.1 | 459.6 | 9.76 | 0.047 |
| v2ProPlus | Upstream T2S | short | 230.2 | 252.9 | 283.3 | 253.1 | 1.22 | 0.207 |
| v2ProPlus | Upstream T2S | medium | 488.4 | 537.0 | 587.2 | 848.7 | 4.04 | 0.210 |
| v2ProPlus | Upstream T2S | long | 872.3 | 927.2 | 1013.5 | 1966.7 | 9.56 | 0.206 |
| v2ProPlus | Graph + SDPA | short | 104.8 | 107.0 | 119.8 | 107.1 | 1.22 | 0.088 |
| v2ProPlus | Graph + SDPA | medium | 176.9 | 180.2 | 188.6 | 301.8 | 4.04 | 0.075 |
| v2ProPlus | Graph + SDPA | long | 278.3 | 291.9 | 297.3 | 640.6 | 9.56 | 0.067 |
| v2ProPlus | Graph + FA2 | short | 91.9 | 100.7 | 107.4 | 100.8 | 1.22 | 0.083 |
| v2ProPlus | Graph + FA2 | medium | 142.6 | 144.7 | 155.6 | 245.2 | 3.96 | 0.062 |
| v2ProPlus | Graph + FA2 | long | 213.3 | 219.6 | 225.9 | 462.6 | 9.70 | 0.048 |

## Environment and interpretation

Pre-load telemetry still showed desktop GPU activity at low P8 clocks (typically
210 MHz and about 2,800 MiB allocated). During optimized steady measurements the GPU
ran at compute clocks, as summarized below. These utilization values include both
the benchmark and any desktop activity, so they cannot attribute contention to a
particular application. The previous run had no comparable telemetry. Consequently,
the improvement cannot be assigned entirely to other programs, or to warmup alone.
Fixed workload, per-case warmup, separated instrumentation, and environment all
differ from the earlier measurement. Existing published metrics remain unchanged.

| Model | Engine | Steady sensor samples | SM clock min/median/max (MHz) | GPU utilization median | Temperature max |
|---|---|---:|---|---:|---:|
| v2 | Upstream T2S | 124 | 1800/2295/2760 | 35% | 55 C |
| v2 | Graph + SDPA | 42 | 2745/2760/2760 | 74% | 62 C |
| v2 | Graph + FA2 | 32 | 2745/2745/2760 | 54% | 64 C |
| v2Pro | Upstream T2S | 123 | 1785/2340/2760 | 34% | 55 C |
| v2Pro | Graph + SDPA | 42 | 2760/2775/2775 | 69% | 53 C |
| v2Pro | Graph + FA2 | 33 | 2760/2760/2775 | 52% | 56 C |
| v2ProPlus | Upstream T2S | 120 | 1845/2460/2775 | 34% | 48 C |
| v2ProPlus | Graph + SDPA | 42 | 2760/2760/2775 | 70% | 54 C |
| v2ProPlus | Graph + FA2 | 32 | 2760/2760/2775 | 62% | 59 C |

## Diagnostic timing

The following are medians of three **separate instrumented short-text runs**.
For medium/long texts, raw stage totals span the complete multi-segment response;
they should not be added to explain only its first audio packet.

| Model | Engine | First use short (ms) | Text frontend (ms) | T2S (ms) | Acoustic decode (ms) | Instrumented first audio (ms) |
|---|---|---:|---:|---:|---:|---:|
| v2 | Upstream T2S | 3078.3 | 0.3 | 213.7 | 55.5 | 272.7 |
| v2 | Graph + SDPA | 3153.9 | 0.3 | 61.0 | 45.1 | 107.6 |
| v2 | Graph + FA2 | 2996.4 | 0.3 | 49.9 | 45.5 | 99.8 |
| v2Pro | Upstream T2S | 3293.9 | 0.3 | 267.8 | 54.3 | 323.3 |
| v2Pro | Graph + SDPA | 3140.9 | 0.3 | 65.8 | 50.7 | 118.8 |
| v2Pro | Graph + FA2 | 3016.3 | 0.3 | 50.8 | 46.3 | 98.3 |
| v2ProPlus | Upstream T2S | 3127.0 | 0.2 | 192.4 | 42.2 | 238.8 |
| v2ProPlus | Graph + SDPA | 3082.5 | 0.3 | 60.9 | 46.3 | 111.1 |
| v2ProPlus | Graph + FA2 | 3106.3 | 0.3 | 48.2 | 46.4 | 95.9 |

## Reproduction and raw evidence

Validation after adding the steady protocol: full local suite **148 passed,
1 skipped**; model-free environment **94 passed, 8 skipped**. Ruff, diff checks,
wheel/sdist builds and `twine check` passed. The runtime implementation is unchanged.

Use the existing private model/reference JSON config and local weights:

```powershell
python benchmarks/model_latency.py --config F:/local/v2Pro.json --engine flash `
  --steady --seed 20260926 --warmup-per-case 5 --repeats 20 --diagnostic-repeats 3 `
  --gpu-uuid GPU-YOUR-DEVICE-UUID --telemetry-output F:/local/gpu.csv `
  --output F:/local/steady.json
```

Repeat for `upstream`, `sdpa`, and `flash`, one process at a time. `nvidia-smi -L`
lists physical GPU UUIDs. Telemetry is sampled every 500 ms and includes three
seconds before imports/model loading. The raw report stores only a UUID hash.

[Raw trials and telemetry summaries](v2-model-retest-20260927.json) include every
sample, token hashes, first-use latency, diagnostics and graph-capture counts.
[GPU telemetry CSV](v2-model-retest-20260927-gpu.csv) combines the original captures
with a variant label. No absolute local paths, reference transcript or audio are published.
The [earlier report](v2-model-latency.md) is retained in full. FA2 may sample
different tokens from SDPA for the same seed; cross-engine hash matches are in
the raw report. This is not a fixed-token kernel benchmark or a voice-quality comparison.
