# Additional v2-family latency benchmarks — 2026-09-26

These measurements supplement the existing v3 and RTX 4070 Laptop results;
those figures are unchanged. Runtime: Aqua `25c5b52`, upstream `08d627c`, Windows,
Python 3.12.10, PyTorch 2.5.1+cu121, RTX 4070 Ti SUPER 16 GB, fp16.

All v2-family weights came from the supplied local GPT-SoVITS v2pro package.
v2 uses official v2 GPT + s2G2333k; Pro uses kurisu_v2pro-e15 + kurisu_v2pro;
Plus uses kurisu_crs_gpt-e100 + kurisu_crs_s2-e12_v2ProPlus. The matching SHA-256
identities are in [the compatibility evidence](v2-model-support.json).
Weights and voice materials are external and are not redistributed.

## Protocol

- One fresh process per checkpoint pair and T2S engine; GPU jobs run serially.
- Upstream T2S remains unpatched. All engines share Aqua's reference cache,
  frontend, semantic guard, SoVITS decoder and output handling. This is not an
  official GPT-SoVITS web UI versus Aqua application comparison.
- Two warmup utterances, then five repeats for each existing short/medium/long
  TTFP text. Paired seeds are 20260926 through 20260930. All repeats are retained;
  lazy graph capture and cold text-shape outliers are not silently removed.
- top_k=5, top_p=1, temperature=0.6, speed=1.1; punctuation segmentation;
  0.25-second output chunks and 0.3-second inter-segment silence.
- First audio is measured from iterator entry to first nonempty PCM. Total time
  drains the complete response. Audio duration includes the configured silence;
  RTF is total generation seconds / generated audio seconds, computed per trial.
- Existing T2S timing instrumentation is enabled equally for all engines. Device
  synchronization before the caller-visible window clears earlier work; internal
  semantic timing also synchronizes. Sound-device startup/playback is excluded.
- Graph variants must actually capture and replay. FA2 measurements forbid the
  runtime's SDPA fallback, so a missing/unsupported FA2 kernel fails the run.
- 135 measured trials and 18 warmup utterances completed successfully. Medians
  below are five-repeat summaries, not minimum/best-case claims.

## First playable audio

| Model | T2S execution | Short / 3 chars (ms) | Medium / 19 chars (ms) | Long / 64 chars (ms) |
|---|---|---:|---:|---:|
| v2 | Upstream T2S | 281.4 | 703.2 | 1172.5 |
| v2 | Aqua Graph + SDPA | 194.1 | 280.3 | 406.4 |
| v2 | Aqua Graph + FA2 | 171.9 | 223.1 | 283.1 |
| v2Pro | Upstream T2S | 349.1 | 496.8 | 911.7 |
| v2Pro | Aqua Graph + SDPA | 180.7 | 186.3 | 338.6 |
| v2Pro | Aqua Graph + FA2 | 198.3 | 156.0 | 304.2 |
| v2ProPlus | Upstream T2S | 290.5 | 546.0 | 1224.3 |
| v2ProPlus | Aqua Graph + SDPA | 194.2 | 337.9 | 364.5 |
| v2ProPlus | Aqua Graph + FA2 | 265.8 | 260.5 | 274.3 |

## Complete response and RTF

| Model | T2S execution | Text | First audio (ms) | Total (ms) | Audio (s) | RTF |
|---|---|---|---:|---:|---:|---:|
| v2 | Upstream T2S | short | 281.4 | 281.7 | 1.04 | 0.281 |
| v2 | Upstream T2S | medium | 703.2 | 1085.3 | 4.22 | 0.261 |
| v2 | Upstream T2S | long | 1172.5 | 2485.0 | 9.82 | 0.258 |
| v2 | Aqua Graph + SDPA | short | 194.1 | 194.3 | 1.04 | 0.194 |
| v2 | Aqua Graph + SDPA | medium | 280.3 | 472.5 | 4.22 | 0.112 |
| v2 | Aqua Graph + SDPA | long | 406.4 | 899.1 | 9.82 | 0.094 |
| v2 | Aqua Graph + FA2 | short | 171.9 | 172.1 | 1.04 | 0.167 |
| v2 | Aqua Graph + FA2 | medium | 223.1 | 384.6 | 4.22 | 0.092 |
| v2 | Aqua Graph + FA2 | long | 283.1 | 582.3 | 9.82 | 0.059 |
| v2Pro | Upstream T2S | short | 349.1 | 349.3 | 1.32 | 0.246 |
| v2Pro | Upstream T2S | medium | 496.8 | 856.7 | 4.00 | 0.207 |
| v2Pro | Upstream T2S | long | 911.7 | 1888.2 | 9.54 | 0.194 |
| v2Pro | Aqua Graph + SDPA | short | 180.7 | 180.9 | 1.32 | 0.123 |
| v2Pro | Aqua Graph + SDPA | medium | 186.3 | 360.6 | 4.00 | 0.087 |
| v2Pro | Aqua Graph + SDPA | long | 338.6 | 680.9 | 9.54 | 0.071 |
| v2Pro | Aqua Graph + FA2 | short | 198.3 | 198.5 | 1.32 | 0.131 |
| v2Pro | Aqua Graph + FA2 | medium | 156.0 | 343.7 | 4.00 | 0.083 |
| v2Pro | Aqua Graph + FA2 | long | 304.2 | 631.9 | 9.76 | 0.064 |
| v2ProPlus | Upstream T2S | short | 290.5 | 290.7 | 1.26 | 0.225 |
| v2ProPlus | Upstream T2S | medium | 546.0 | 860.7 | 4.04 | 0.213 |
| v2ProPlus | Upstream T2S | long | 1224.3 | 2526.1 | 9.56 | 0.255 |
| v2ProPlus | Aqua Graph + SDPA | short | 194.2 | 194.5 | 1.26 | 0.144 |
| v2ProPlus | Aqua Graph + SDPA | medium | 337.9 | 569.9 | 4.04 | 0.144 |
| v2ProPlus | Aqua Graph + SDPA | long | 364.5 | 754.2 | 9.56 | 0.077 |
| v2ProPlus | Aqua Graph + FA2 | short | 265.8 | 266.0 | 1.26 | 0.182 |
| v2ProPlus | Aqua Graph + FA2 | medium | 260.5 | 463.4 | 4.00 | 0.114 |
| v2ProPlus | Aqua Graph + FA2 | long | 274.3 | 556.3 | 9.70 | 0.058 |

## Output equivalence and limits

FA2 is not uniformly lowest-latency: short Pro/Plus medians were higher than
SDPA in this run. The table preserves those observations without selecting a
preferred result for each model.

- v2 upstream vs sdpa: 15/15 paired trials have identical semantic token hashes.
- v2 upstream vs flash: 10/15 paired trials have identical semantic token hashes.
- v2Pro upstream vs sdpa: 15/15 paired trials have identical semantic token hashes.
- v2Pro upstream vs flash: 13/15 paired trials have identical semantic token hashes.
- v2ProPlus upstream vs sdpa: 15/15 paired trials have identical semantic token hashes.
- v2ProPlus upstream vs flash: 12/15 paired trials have identical semantic token hashes.

FA2 can change sampled tokens through floating-point backend differences, even
with identical seeds. Treat these as observed same-input generation latencies,
not a fixed-token microbenchmark or a guarantee of bit-identical output. Raw
trials include token hashes, token counts, retry counts and audio durations so
this distinction remains inspectable. Different checkpoint families also use
different voice weights; these results do not isolate architecture or voice quality.

V2-family synthesis decodes a full text segment before yielding its PCM chunks.
The model-side first audio here is not incremental within-segment decoding, and
must not be interpreted as physical speaker latency. Windows desktop scheduling
and GPU state can affect the small differences between engines. The old v3 table
used another Torch/CUDA build and remains a separate historical measurement.

## Reproduce

Use the local config format in [the benchmark guide](../README.md#real-weight-model-compatibility):

```powershell
python benchmarks/model_latency.py --config F:/local/v2ProPlus.json --engine upstream --output F:/local/upstream.json
python benchmarks/model_latency.py --config F:/local/v2ProPlus.json --engine sdpa --output F:/local/sdpa.json
python benchmarks/model_latency.py --config F:/local/v2ProPlus.json --engine flash --output F:/local/flash.json
```

Set `GPT_SOVITS_HOME` to the selected upstream checkout. Supply your existing
weights, BERT/CNHuBERT/langdetect assets and matching reference audio/transcript.
[Raw results](v2-model-latency.json) retain every repeat and warmup; no local paths
or reference transcripts are included.
