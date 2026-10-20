# Aqua README performance retest — 2026-10-21

Measured source: `74dfa923c9d0032faeff7547e1add028fa112a96` for T2S and steady latency,
`77e5006f6ffce950d0f869d1ae3a3b840d88bc83` for the original V3 protocol, and
`a1c9054d5a872530704b5b37763faa016ba9a42c` for the final T2S FA2 measurements.
These commits only change benchmark evidence recording and fallback checks; inference code is unchanged from
the dual-GPU-validated stack `4206bb1996ca1ca50ba992ea1b180adb8fe61d2c`.
Upstream: `08d627c3338173c3229286d8787060d6559fe0f8`. Only Aqua engines were run.

## Environment and scope

- Windows; Python 3.12.10; Torch 2.5.1+cu121; FlashAttention 2.5.9.post1;
  NVIDIA driver 581.57; fp16.
- RTX 4070 Ti SUPER 16 GB via USB4 with desktop graphics still active.
- RTX 4070 Laptop 8 GB, not driving a display, for the published mobile T2S row.
- Default `bucket` KV; CFM unpadded off. V3 BigVGAN CUDA kernel was verified.
- Headline FA2 jobs used `valid` mode and rejected SDPA fallback before capture.
- **63 fixed-work measured T2S trials**, **30 original-protocol V3 requests** and
  **360 v2-family steady requests** supply the README tables.
  Another **120 V3 steady requests** are supplementary, for **510 total measured
  latency requests**. First use, warmup and diagnostics are additional records.
- The first two T2S FA2 jobs (42 trials) installed the fallback guard after
  pre-capture. Both were rerun with the guard moved before capture; the final
  guarded jobs supply the headline. The initial jobs remain in the JSON,
  giving **105 total measured T2S trials**. No successful timing code changed.
- No audio playback, official baseline, optional nested/CFM speedup matrix or
  new perceptual-quality assessment was performed.

## T2S throughput

The historical 51/123/180-step shapes, 15 conversation warmups and seven repeats
are unchanged. EOS is forced to -1 for fixed work, as in the old benchmark.
Rates are synchronized before/after the request; no per-token diagnostic sync is
enabled. First AR excludes loading and graph preparation.

| GPU / engine | First AR after capture | Short / 448 | Conversation / 512 | Long / 768 |
|---|---|---|---|---|
| Ti SUPER · SDPA | 362.2 | 454.0 | 487.6 | 456.1 |
| Ti SUPER · FA2 | 437.6 | 669.6 | 715.1 | 688.1 |
| Laptop · FA2 | 331.7 | 542.1 | 572.4 | 550.0 |

## V3 headline: original first-PCM protocol

The original `aqua_ttfp.py` measurement function is AST-identical to `4206bb1`.
Only out-of-window evidence saving was added. Each fresh process runs the same
two warmup texts, then five measurements per short/medium/long text, retains
`collect_t2s_stats=True`, does not reset the benchmark seed, and stops at first
nonempty PCM. Chunk size is 0.25 seconds, speed 1.1, CFM steps four.
The current API's generated-lead trimming remains enabled.

| Engine | Input | Median ms | All five trials, ms (execution order) |
|---|---|---|---|
| sdpa | short | 491.4 | 468.1 / 491.4 / 491.0 / 617.1 / 612.5 |
| sdpa | medium | 522.2 | 491.0 / 522.2 / 515.5 / 529.8 / 539.0 |
| sdpa | long | 630.0 | 619.9 / 778.3 / 625.6 / 630.0 / 742.4 |
| flash | short | 685.7 | 666.8 / 757.9 / 591.7 / 815.6 / 685.7 |
| flash | medium | 751.3 | 884.0 / 672.8 / 651.9 / 751.3 / 755.9 |
| flash | long | 1018.6 | 1037.6 / 1018.6 / 1090.2 / 779.9 / 892.1 |

These are the V3 values used by both READMEs. The initial 20-repeat V3 run used
the v2 steady protocol; it is retained below as supplementary evidence, rather
than silently substituting a different measurement method for the old V3 script.

## V2-family current API latency

Five same-text warmups, twenty fixed-seed (`20260926`) measurements and three
separate diagnostics follow each text's first use, reusing `model_latency.py`.
Current product onset behavior is retained.
No formal request captured a new graph. P95 is nearest rank over twenty samples.
RTF includes the configured 0.3-second pauses. V3 outputs 24 kHz; v2-family 32 kHz.

| Model | Engine | Input | Min ms | p50 ms | p95 ms | Total p50 ms | Audio p50 s | RTF p50 |
|---|---|---|---|---|---|---|---|---|
| v2 | sdpa | short | 152.3 | 185.9 | 218.8 | 186.1 | 0.8 | 0.248 |
| v2 | sdpa | medium | 235.9 | 270.5 | 288.5 | 448.9 | 2.8 | 0.160 |
| v2 | sdpa | long | 371.4 | 403.9 | 484.6 | 908.4 | 8.5 | 0.107 |
| v2 | flash | short | 131.6 | 144.6 | 172.2 | 144.9 | 0.8 | 0.193 |
| v2 | flash | medium | 135.6 | 146.3 | 159.1 | 246.3 | 2.8 | 0.088 |
| v2 | flash | long | 215.3 | 221.6 | 230.8 | 458.4 | 8.5 | 0.054 |
| v2Pro | sdpa | short | 106.3 | 112.7 | 120.5 | 112.9 | 0.9 | 0.131 |
| v2Pro | sdpa | medium | 165.5 | 172.7 | 177.7 | 288.1 | 2.8 | 0.104 |
| v2Pro | sdpa | long | 281.6 | 288.0 | 294.1 | 595.3 | 8.1 | 0.073 |
| v2Pro | flash | short | 86.7 | 96.1 | 102.0 | 96.3 | 0.9 | 0.112 |
| v2Pro | flash | medium | 131.9 | 137.0 | 145.2 | 234.6 | 2.8 | 0.085 |
| v2Pro | flash | long | 215.2 | 222.1 | 226.4 | 443.3 | 8.2 | 0.054 |
| v2ProPlus | sdpa | short | 94.5 | 105.9 | 112.5 | 106.1 | 0.7 | 0.149 |
| v2ProPlus | sdpa | medium | 179.5 | 192.5 | 211.9 | 312.8 | 2.8 | 0.113 |
| v2ProPlus | sdpa | long | 270.0 | 281.9 | 300.2 | 621.0 | 8.0 | 0.078 |
| v2ProPlus | flash | short | 89.2 | 99.8 | 113.2 | 100.0 | 0.7 | 0.141 |
| v2ProPlus | flash | medium | 147.1 | 155.6 | 165.2 | 261.8 | 2.8 | 0.094 |
| v2ProPlus | flash | long | 207.1 | 217.2 | 230.1 | 454.7 | 8.2 | 0.056 |

## Supplementary V3 steady measurements

Same 5/20/3 fixed-seed steady protocol as the preceding v2 table. These are
full-generation requests with instrumentation disabled for formal timing,
unlike the original V3 first-chunk protocol above; do not mix their medians.
No formal request captured a new graph. This section supplies supplementary
tail latency, total time and RTF, not V3 README headlines.

| Model | Engine | Input | Min ms | p50 ms | p95 ms | Total p50 ms | Audio p50 s | RTF p50 |
|---|---|---|---|---|---|---|---|---|
| v3 | sdpa | short | 466.9 | 693.3 | 865.1 | 1033.2 | 0.8 | 1.240 |
| v3 | sdpa | medium | 345.1 | 373.4 | 435.3 | 3047.5 | 3.5 | 0.880 |
| v3 | sdpa | long | 677.4 | 748.4 | 990.3 | 5257.0 | 8.1 | 0.645 |
| v3 | flash | short | 409.7 | 437.6 | 466.2 | 665.3 | 0.8 | 0.798 |
| v3 | flash | medium | 225.7 | 232.2 | 245.0 | 2050.9 | 3.5 | 0.592 |
| v3 | flash | long | 617.9 | 692.7 | 937.0 | 5301.9 | 8.1 | 0.651 |

## Interpretation

These are new measurements of the current API, not a matched migration-before/
after experiment. The old V3 report does not contain file hashes, so its
same-named assets cannot be proven byte-identical; its Torch CUDA build and
FlashAttention version and API onset behavior also differ. Do not attribute historical differences solely
to code changes. The current JSON records checkpoint/reference hashes and every
sample. Higher latencies are retained rather than replaced with historical values.

The original V3 first-PCM run was slower with FA2 than SDPA, despite the
fixed-work T2S advantage. The supplementary steady run gives a different
ordering. These observations do not identify a cause or establish a migration
regression; the protocols, graph preparation and sampled work differ. Both
sets are preserved, and no latency acceptance threshold was declared passed.

The eGPU also drives the desktop. Pre-load graphics utilization was nonzero
(including a burst at the beginning of the first T2S run). Telemetry includes
desktop and benchmark activity together; it cannot assign utilization to an
application. Reported ranges/p95 retain this variation. USB4 host synchronization
affects T2S, so compare variants on one card and avoid cross-card hardware rankings.

## Reproduction and evidence

Use `t2s_speed_bench.py --engine aqua --flash-attn off` and `--flash-attn valid`,
with `--warmup 15 --repeats 7 --seed 20260830` and the v3 GPT checkpoint.
Use `aqua_ttfp.py --json-output ...` for the V3 headline, with
`AQUATTS_T2S_FLASH_ATTN=0` or `1`; leave `--measure-total` off.
Use `model_latency.py --engine sdpa` or `flash` for v2-family and supplementary
V3 with `--steady --seed 20260926 --warmup-per-case 5 --repeats 20
--diagnostic-repeats 3`; supplementary V3 also requires `--require-bigvgan-kernel`.
Run each configuration in a fresh process with one
physical GPU selected by UUID, serially. Private paths/transcripts are not published.

All final samples and workload hashes: [readme-performance-20261021.json](readme-performance-20261021.json).
Per-trial journals are also retained in the local audit directory.

GPU telemetry:

- [t2s_ti_super_sdpa](readme-performance-20261021-t2s_ti_super_sdpa-gpu.csv)
- [t2s_ti_super_flash](readme-performance-20261021-t2s_ti_super_flash-gpu.csv)
- [t2s_laptop_flash](readme-performance-20261021-t2s_laptop_flash-gpu.csv)
- [latency_v3_sdpa](readme-performance-20261021-latency_v3_sdpa-gpu.csv)
- [latency_v3_flash](readme-performance-20261021-latency_v3_flash-gpu.csv)
- [latency_v2_sdpa](readme-performance-20261021-latency_v2_sdpa-gpu.csv)
- [latency_v2_flash](readme-performance-20261021-latency_v2_flash-gpu.csv)
- [latency_v2Pro_sdpa](readme-performance-20261021-latency_v2Pro_sdpa-gpu.csv)
- [latency_v2Pro_flash](readme-performance-20261021-latency_v2Pro_flash-gpu.csv)
- [latency_v2ProPlus_sdpa](readme-performance-20261021-latency_v2ProPlus_sdpa-gpu.csv)
- [latency_v2ProPlus_flash](readme-performance-20261021-latency_v2ProPlus_flash-gpu.csv)
- [original_v3_sdpa](readme-performance-20261021-original_v3_sdpa-gpu.csv)
- [original_v3_flash](readme-performance-20261021-original_v3_flash-gpu.csv)
- [verified_t2s_ti_super_flash](readme-performance-20261021-verified_t2s_ti_super_flash-gpu.csv)
- [verified_t2s_laptop_flash](readme-performance-20261021-verified_t2s_laptop_flash-gpu.csv)
