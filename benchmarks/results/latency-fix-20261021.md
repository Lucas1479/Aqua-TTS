# Migration latency diagnosis and semantic-history integration

Measured 2026-10-21 on the USB4 RTX 4070 Ti SUPER. Final measured source:
`82b7ad5737a6b94c2424d022c3ea1300be5c7d38`. Production fix: `8cef3a9`; the later commit only
adapts the offline dictionary test to pyopenjtalk-plus.

## Findings and fix

The October reference was #12 (`6d39b7d`), not migration main (`591fbb5`).
#12 was on a separate branch and had not been included in the migration stack.
Its request-local semantic-history buffer and masked-EOS host-read skip are now
integrated into the current decoder. This preserves the workspace leases,
capture admission, nested promotion, streaming locks and reference error fixes.
The sampler sees exactly the written prefix; the first 11 steps cannot emit EOS.
Product silence trimming remains unchanged.

This omission explains a small, measured T2S cost. It does **not** explain the
earlier 50–100 ms first-PCM differences. In the controlled same-session comparison,
the old implementation and the current implementation have similar latency.
The previous V3 long / FA2 410 ms result does not reproduce there; both revisions
reach roughly 310 ms. V2Pro long / FA2 likewise returns to roughly 220 ms.
The historical cu121 measurements used the wrong runtime and are not evidence
of a source regression; post-trim availability is also not historical TTFP.

## Controlled comparison

FA2, original model/reference files, seed 20260926, full generation and first raw
PCM. Two resident processes, serialized requests, six permutations of arm order.
The before/candidate arms share one model, graphs, allocator and process; only
their T2S function changes. Each text has first use, 5 warmups, 12 formal trials
per arm and 3 separate diagnostics. Logging is suppressed equally for this
diagnosis, so these results do not replace the original-script README measurements.

| Model | Input | October source now (ms) | Migration before (ms) | Candidate (ms) |
|---|---|---|---|---|
| v3 | short | 254.7 | 242.5 | 242.8 |
| v3 | long | 311.0 | 313.0 | 308.2 |
| v2Pro | short | 102.3 | 94.9 | 98.9 |
| v2Pro | long | 223.2 | 219.8 | 219.4 |

All three arms match token hashes, **PCM bytes**, and CUDA RNG state in all
60 matched formal/diagnostic groups. No graph was captured in those requests.
Formal timers have no stage synchronization. In separate event diagnostics,
cached references and frontend each take about 1 ms or less. V3 short's largest
observed stage variation is CFM (about 111–165 ms median across the three arms),
including the old source. T2S is about 53–55 ms and BigVGAN about 14–17 ms.
Only three diagnostic trials per arm were taken: this localizes observed
variation but does not identify its OS/driver cause. CUDA event intervals include
dispatch gaps; nested timings must not be summed blindly.

## Follow-up on the remaining V3 SDPA short-input gap

The original-script candidate batch still measured 241.9 ms against October's
215.2 ms. A targeted same-session SDPA follow-up used the same controlled
protocol above (12 formal trials per arm, plus separate warmups/diagnostics).
It measured **219.0 / 221.0 / 220.9 ms** for October source / migration before /
candidate. The 12.4% historical-batch difference did not track source revision
in this check. All 15 matched formal/diagnostic groups also have identical
semantic bytes, PCM bytes and CUDA RNG state, with no new captures.

This supports a small code effect plus temporal variation; it does not establish
which OS/driver mechanism caused that variation. The 241.9 ms original-script
measurement remains in README and all raw samples remain in evidence. The
controlled 220.9 ms value is not substituted into that table. Telemetry:
[SDPA follow-up](latency-fix-20261021-sdpa-short-followup-gpu.csv).

## Isolated T2S effect

Adapted the original October `compare_aqua_history.py` method using the existing
`t2s_speed_bench.py` loader/inputs: same process/model/graphs, 15 paired warmups,
20 alternating pairs per 51/123/180-step shape, native EOS computation retained
and only its stop result ignored. Seeds start at 20260830. The table reports
median paired throughput gain with 5,000-resample bootstrap intervals.

| Engine | Input | Paired gain | 95% bootstrap interval |
|---|---|---|---|
| sdpa | short | 1.98% | 0.31% to 4.67% |
| sdpa | conversation | 1.37% | 0.34% to 2.64% |
| sdpa | long | 0.95% | -0.95% to 1.87% |
| flash | short | 3.68% | 2.43% to 6.53% |
| flash | conversation | 4.65% | 2.24% to 6.64% |
| flash | long | 0.58% | -1.16% to 2.30% |

All 120 pairs match semantic bytes and final CUDA RNG state. Long-case intervals
include zero, so a reliable long-case speedup is not claimed. This optimization
cannot account for the earlier large pipeline gaps.

## Original README protocol after the fix

Reused `model_latency.py` for all four models: first use, 5 warmups, 20 fixed-seed
formal requests and 1 separate diagnostic per text; first **raw PCM**, full
generation, seed 20260926, speed 1.1, 4 CFM steps, chunks 0.25 s, pauses 0.3 s.
All 480 formal requests are retained and capture no new graphs. T2S reuses
15 warmups and 7 repeats per shape (42 formal trials). No official matrix ran.

Torch 2.6.0+cu124, FA2 2.7.4.post1, Python 3.12.10, driver 581.57; upstream
`08d627c3338173c3229286d8787060d6559fe0f8`; default bucket KV, CFM unpadded off.
FA2 fallback is forbidden before capture. V3 actually loads the original local
BigVGAN CUDA binary, SHA-256 `d28f21a5a04c5adf07560d05975282223186cf442c0b1f2c130f7067d6dd9876`; the CUDA sources
match. The original compiler was CUDA 12.6, linked against Torch's 12.4 runtime.
Checkpoint/reference identities and every trial are in the JSON.

| Ti SUPER engine | Input | Oct 14 it/s | Candidate it/s | Change |
|---|---|---|---|---|
| sdpa | short | 481.2 | 504.3 | +4.8% |
| sdpa | conversation | 498.3 | 488.8 | -1.9% |
| sdpa | long | 427.1 | 460.8 | +7.9% |
| flash | short | 628.3 | 618.7 | -1.5% |
| flash | conversation | 692.3 | 687.5 | -0.7% |
| flash | long | 725.1 | 738.5 | +1.8% |

The following batches are observations at different times, **not paired causal
estimates**. The controlled results above are the evidence for code impact.

| Model | Engine | Input | Oct 14 p50 ms | Pre-fix batch ms | Candidate batch ms | Candidate vs Oct |
|---|---|---|---|---|---|---|
| v3 | sdpa | short | 215.2 | 296.4 | 241.9 | +12.4% |
| v3 | sdpa | medium | 270.0 | 307.2 | 281.5 | +4.3% |
| v3 | sdpa | long | 368.7 | 400.2 | 361.2 | -2.0% |
| v3 | flash | short | 205.6 | 224.5 | 212.8 | +3.5% |
| v3 | flash | medium | 244.3 | 254.6 | 256.5 | +5.0% |
| v3 | flash | long | 312.5 | 410.2 | 307.2 | -1.7% |
| v2 | sdpa | short | 103.2 | 113.9 | 96.9 | -6.1% |
| v2 | sdpa | medium | 172.4 | 188.3 | 162.0 | -6.1% |
| v2 | sdpa | long | 281.8 | 346.8 | 284.1 | +0.8% |
| v2 | flash | short | 91.5 | 104.0 | 86.1 | -5.9% |
| v2 | flash | medium | 141.6 | 151.7 | 132.4 | -6.5% |
| v2 | flash | long | 225.6 | 230.8 | 211.0 | -6.4% |
| v2Pro | sdpa | short | 109.0 | 113.0 | 100.7 | -7.6% |
| v2Pro | sdpa | medium | 171.2 | 169.3 | 160.6 | -6.2% |
| v2Pro | sdpa | long | 280.2 | 281.1 | 267.2 | -4.6% |
| v2Pro | flash | short | 92.6 | 145.9 | 88.8 | -4.1% |
| v2Pro | flash | medium | 133.6 | 206.3 | 128.1 | -4.1% |
| v2Pro | flash | long | 219.1 | 320.6 | 211.2 | -3.6% |
| v2ProPlus | sdpa | short | 98.4 | 151.7 | 97.3 | -1.1% |
| v2ProPlus | sdpa | medium | 174.4 | 259.2 | 169.2 | -3.0% |
| v2ProPlus | sdpa | long | 265.5 | 365.7 | 256.9 | -3.3% |
| v2ProPlus | flash | short | 89.5 | 141.7 | 85.6 | -4.4% |
| v2ProPlus | flash | medium | 145.0 | 230.7 | 138.9 | -4.2% |
| v2ProPlus | flash | long | 216.5 | 322.9 | 203.3 | -6.1% |

## Validation and remaining limits

- Correct-runtime suite: **530 passed / 2 skipped**, zero failures.
  The initial run found one test assuming pyopenjtalk's private `_extract_dic`;
  pyopenjtalk-plus 0.4.1-post9 has no such symbol. The test now guards network
  access in both distributions and the legacy downloader when present. The
  production offline check did not need modification. Initial failure evidence
  remains archived locally alongside the successful full rerun.
- The 12 semantic-history tests cover prompt dtypes, storage reuse, sampler
  history, EOS, length stop, limit stop, and terminal-token trimming.
- Real-weight controlled validation covers V3 and v2Pro, short and long inputs.
- Only the Ti SUPER is currently enumerated. The user reports eGPU hotplug can
  affect dGPU loading; this is not treated as a latency bug. Laptop and two-card
  validation of this final candidate remain pending. Earlier two-card results
  used a different source/runtime and are not relabeled as this run.
- Desktop graphics remained active; no other model-compute process was found.
  No power settings were changed. No PR was merged. Performance acceptance is
  not inferred solely from completing the run or from a fast individual trial.

All data: [latency-fix-20261021.json](latency-fix-20261021.json). Paired GPU telemetry:
[latency-fix-20261021-paired-profile-gpu.csv](latency-fix-20261021-paired-profile-gpu.csv).
The original-script runs each include their own linked telemetry file in JSON.
The previous [cu124 report](readme-performance-cu124-20261021.md) is retained.
