# README performance — verified Torch 2.6/cu124 environment

Measured 2026-10-21. Source `ad91a3d048868ea89ae07a3f57abcc2d25460b9c`; production code remains
the reviewed migration stack. This run uses the actual 2026-10-14 benchmark
directory's scripts/protocol and asset configurations. Earlier cu121 attempts
are [nonbaseline observations](readme-performance-20261021.md), not acceptance data.

## Environment and workload

- Windows 11, Python 3.12.10, **Torch 2.6.0+cu124**, CUDA runtime **12.4**,
  **FlashAttention2 2.7.4.post1**, NVIDIA driver 581.57, fp16.
- RTX 4070 Ti SUPER over USB4 for T2S/latency; RTX 4070 Laptop for the mobile T2S row.
  Desktop graphics remain active on the Ti SUPER; no other model compute job was found.
- Upstream GPT-SoVITS `08d627c3338173c3229286d8787060d6559fe0f8`. Default bucket KV; CFM
  unpadded off. Every subprocess asserts its interpreter, package versions and Aqua source path.
- Reused the existing local BigVGAN CUDA extension for Torch 2.6/cu124. Its four
  CUDA/C++ source/header files match the current sources byte-for-byte. The original
  build used the installed CUDA 12.6 compiler with Torch's 12.4 runtime; this is
  the previously built cache, not a new compilation. Binary SHA-256:
  `d28f21a5a04c5adf07560d05975282223186cf442c0b1f2c130f7067d6dd9876`. Both V3 jobs confirm actual CUDA loading.
- T2S uses the original `kurisu_v3hc_ja-e13.ckpt` workload; V3 latency uses the
  separate README `xxx-e15.ckpt` / `xxx_e2_s174_l32.pth` pair and Kurisu reference.
  These configurations have different purposes and must not be interchanged.
  Current checkpoint/reference hashes are in the JSON.

## Protocol

**TTFP ends at the first raw PCM block before silence buffering/cropping.**
Only the isolated benchmark process bypasses onset processing; production
behavior is unchanged. Quiet PCM counts. Post-trim API availability is a separate metric.

All four latency models reuse the actual October steady protocol: each text's
first use, **5 warmups, 20 fixed-seed measurements, 1 separate diagnostic**;
seed `20260926`, speed 1.1, four CFM steps, 0.25-second chunks, 0.3-second pauses.
Formal timing disables stage synchronization, checks PCM after timing, and
captures no new graphs. All **480 formal requests** are journaled and retained.

T2S reuses the original fixed 51/123/180-step shapes, 15 warmups, seven repeats
per shape, synchronization on the selected device, and the native EOS predicate
with its stop result ignored. There are **63 formal T2S trials**. FA2 fallback
checks are installed before capture. Runs are serial and use fresh processes.

## T2S throughput (median synchronized it/s)

| GPU / engine | First AR after capture | Short / 448 | Conversation / 512 | Long / 768 |
|---|---|---|---|---|
| Ti SUPER · SDPA | 326.1 | 449.3 | 525.3 | 466.0 |
| Ti SUPER · FA2 | 404.8 | 580.9 | 699.8 | 713.7 |
| Laptop · FA2 | 276.7 | 435.4 | 501.9 | 501.6 |

## Raw-PCM latency

| Model | Engine | Input | Min ms | p50 ms | p95 ms | Total p50 ms | RTF p50 |
|---|---|---|---|---|---|---|---|
| v3 | sdpa | short | 241.1 | 296.4 | 324.1 | 960.1 | 0.671 |
| v3 | sdpa | medium | 281.5 | 307.2 | 411.2 | 2776.9 | 0.672 |
| v3 | sdpa | long | 355.9 | 400.2 | 470.5 | 5877.3 | 0.629 |
| v3 | flash | short | 208.8 | 224.5 | 361.0 | 770.2 | 0.538 |
| v3 | flash | medium | 238.4 | 254.6 | 274.5 | 2288.1 | 0.554 |
| v3 | flash | long | 313.7 | 410.2 | 508.6 | 6374.3 | 0.682 |
| v2 | sdpa | short | 100.4 | 113.9 | 122.8 | 114.0 | 0.091 |
| v2 | sdpa | medium | 160.7 | 188.3 | 268.7 | 332.0 | 0.079 |
| v2 | sdpa | long | 282.9 | 346.8 | 541.8 | 755.7 | 0.078 |
| v2 | flash | short | 90.2 | 104.0 | 121.9 | 104.2 | 0.083 |
| v2 | flash | medium | 140.5 | 151.7 | 166.7 | 272.2 | 0.065 |
| v2 | flash | long | 217.2 | 230.8 | 256.6 | 459.1 | 0.048 |
| v2Pro | sdpa | short | 103.5 | 113.0 | 128.0 | 113.1 | 0.086 |
| v2Pro | sdpa | medium | 159.7 | 169.3 | 189.3 | 286.7 | 0.071 |
| v2Pro | sdpa | long | 268.8 | 281.1 | 302.6 | 597.3 | 0.062 |
| v2Pro | flash | short | 130.2 | 145.9 | 176.7 | 146.1 | 0.111 |
| v2Pro | flash | medium | 188.8 | 206.3 | 243.1 | 348.9 | 0.093 |
| v2Pro | flash | long | 307.7 | 320.6 | 395.1 | 646.5 | 0.067 |
| v2ProPlus | sdpa | short | 137.4 | 151.7 | 189.5 | 152.0 | 0.125 |
| v2ProPlus | sdpa | medium | 236.1 | 259.2 | 301.6 | 424.1 | 0.104 |
| v2ProPlus | sdpa | long | 351.4 | 365.7 | 434.2 | 780.2 | 0.083 |
| v2ProPlus | flash | short | 124.5 | 141.7 | 166.1 | 141.9 | 0.116 |
| v2ProPlus | flash | medium | 200.2 | 230.7 | 264.3 | 375.7 | 0.092 |
| v2ProPlus | flash | long | 285.8 | 322.9 | 403.7 | 672.3 | 0.072 |

## Comparison with the actual October baseline

The earlier local baseline is `6d39b7d78115f32a21229bb6bd862961c56887cb` on the
semantic-history branch (#12), whose
[published evidence](https://github.com/Lucas1479/Aqua-TTS/blob/6d39b7d78115f32a21229bb6bd862961c56887cb/benchmarks/results/semantic-history-20261014.json)
records the October figures; the measured source is the migration stack.
The environment, workload paths and protocol now match that evidence. These
serial runs were not an alternating same-session A/B. Desktop scheduling,
sampling differences and the source differences remain visible in the data.
Token-hash equality is reported per case instead of assuming identical generated work.

| Model | Engine | Input | Oct 14 p50 | Current p50 | Change | Semantic hash sets equal |
|---|---|---|---|---|---|---|
| v3 | sdpa | short | 215.2 | 296.4 | +37.7% | True |
| v3 | sdpa | medium | 270.0 | 307.2 | +13.8% | True |
| v3 | sdpa | long | 368.7 | 400.2 | +8.5% | True |
| v3 | flash | short | 205.6 | 224.5 | +9.2% | True |
| v3 | flash | medium | 244.3 | 254.6 | +4.2% | True |
| v3 | flash | long | 312.5 | 410.2 | +31.3% | True |
| v2 | sdpa | short | 103.2 | 113.9 | +10.3% | True |
| v2 | sdpa | medium | 172.4 | 188.3 | +9.2% | True |
| v2 | sdpa | long | 281.8 | 346.8 | +23.1% | True |
| v2 | flash | short | 91.5 | 104.0 | +13.8% | True |
| v2 | flash | medium | 141.6 | 151.7 | +7.1% | True |
| v2 | flash | long | 225.6 | 230.8 | +2.3% | True |
| v2Pro | sdpa | short | 109.0 | 113.0 | +3.6% | True |
| v2Pro | sdpa | medium | 171.2 | 169.3 | -1.1% | True |
| v2Pro | sdpa | long | 280.2 | 281.1 | +0.3% | True |
| v2Pro | flash | short | 92.6 | 145.9 | +57.5% | True |
| v2Pro | flash | medium | 133.6 | 206.3 | +54.4% | True |
| v2Pro | flash | long | 219.1 | 320.6 | +46.3% | True |
| v2ProPlus | sdpa | short | 98.4 | 151.7 | +54.1% | True |
| v2ProPlus | sdpa | medium | 174.4 | 259.2 | +48.6% | True |
| v2ProPlus | sdpa | long | 265.5 | 365.7 | +37.7% | True |
| v2ProPlus | flash | short | 89.5 | 141.7 | +58.3% | True |
| v2ProPlus | flash | medium | 145.0 | 230.7 | +59.1% | True |
| v2ProPlus | flash | long | 216.5 | 322.9 | +49.2% | True |

Performance acceptance has not passed. The user requires restored performance;
unresolved differences need a stage-level explanation. Merging remains unauthorized.

## Evidence

All trials: [readme-performance-cu124-20261021.json](readme-performance-cu124-20261021.json). Local JSONL journals retain every completed
trial even on interruption. Loading/first use and diagnostic records are separate.

- [t2s_ti_super_sdpa GPU telemetry](readme-performance-cu124-20261021-t2s_ti_super_sdpa-gpu.csv)
- [t2s_ti_super_flash GPU telemetry](readme-performance-cu124-20261021-t2s_ti_super_flash-gpu.csv)
- [t2s_laptop_flash GPU telemetry](readme-performance-cu124-20261021-t2s_laptop_flash-gpu.csv)
- [latency_v3_sdpa GPU telemetry](readme-performance-cu124-20261021-latency_v3_sdpa-gpu.csv)
- [latency_v3_flash GPU telemetry](readme-performance-cu124-20261021-latency_v3_flash-gpu.csv)
- [latency_v2_sdpa GPU telemetry](readme-performance-cu124-20261021-latency_v2_sdpa-gpu.csv)
- [latency_v2_flash GPU telemetry](readme-performance-cu124-20261021-latency_v2_flash-gpu.csv)
- [latency_v2Pro_sdpa GPU telemetry](readme-performance-cu124-20261021-latency_v2Pro_sdpa-gpu.csv)
- [latency_v2Pro_flash GPU telemetry](readme-performance-cu124-20261021-latency_v2Pro_flash-gpu.csv)
- [latency_v2ProPlus_sdpa GPU telemetry](readme-performance-cu124-20261021-latency_v2ProPlus_sdpa-gpu.csv)
- [latency_v2ProPlus_flash GPU telemetry](readme-performance-cu124-20261021-latency_v2ProPlus_flash-gpu.csv)
