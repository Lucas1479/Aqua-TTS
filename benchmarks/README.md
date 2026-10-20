> Performance refresh pending: cu121 draft figures are nonbaseline observations, not acceptance evidence. Required environment: Torch 2.6/cu124.

# Aqua benchmark suite

Reproducible comparisons of Aqua and the selected upstream GPT-SoVITS
checkout. Benchmark commands require `GPT_SOVITS_HOME` or an explicit
`--upstream-home`; Aqua no longer benchmarks a vendored T2S model fork.

## T2S AR throughput

Run the Aqua CUDA Graph + SDPA fallback baseline:

```bash
python benchmarks/t2s_speed_bench.py \
  --upstream-home /path/to/GPT-SoVITS \
  --gpt-model /path/to/xxx-e15.ckpt
```

Run the preferred FlashAttention2 path explicitly (the product runtime selects
this automatically when FA2 is importable; the benchmark keeps variants explicit):

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

**2026-10-21, Aqua-only retest.** Windows, Python 3.12.10, Torch 2.5.1+cu121,
upstream `08d627c`, default bucket storage, 15 warmups and seven synchronized
measurements per fixed shape. Ti SUPER is USB4-attached and drives a desktop;
Laptop is not a display GPU. Values are median it/s.

| GPU / engine | First AR after capture | Short / 448 | Conversation / 512 | Long / 768 |
|---|---|---|---|---|
| Ti SUPER · SDPA | 362.2 | 454.0 | 487.6 | 456.1 |
| Ti SUPER · FA2 | 437.6 | 669.6 | 715.1 | 688.1 |
| Laptop · FA2 | 331.7 | 542.1 | 572.4 | 550.0 |

Each configuration ran in a fresh process. FA2 jobs reject SDPA fallback.
The [report](results/readme-performance-20261021.md), [all samples](results/readme-performance-20261021.json)
and linked telemetry preserve ranges and context. No official engine was rerun.
The [August upstream comparison](results/4070ti-super-win-py312-torch251-cu124-upstream-overlay.md)
remains historical; it used a different Torch CUDA build and lacks weight hashes.
Use same-card comparisons, not cross-GPU rates, for this USB4 host.

## TTFP (time to first playable audio)

Headline TTFP keeps the historical boundary: text-ready to the first raw PCM
block, before generated-lead buffering/trimming. It is not sound-device latency.
The 2026-10-21 V3 headline reuses `aqua_ttfp.py` with its original measurement
function (AST checked against `4206bb1`): two warmup texts, five repeats per test
text, T2S statistics enabled, no benchmark seed reset, and stop at first PCM.
Chunk size is 0.25 seconds, speed 1.1 and CFM steps four. Both latency scripts
default to `--audio-boundary raw-pcm`, which bypasses onset buffering/cropping
only in their isolated benchmark process. This restores both the original
endpoint and stop-after-first-chunk RNG consumption. Product inference is unchanged.

`--audio-boundary api-pcm` measures current post-trim API availability instead.
Keep that metric separate: silent generated chunks may delay its first output.

```powershell
$env:CUDA_VISIBLE_DEVICES = "GPU-YOUR-DEVICE-UUID"
$env:AQUATTS_T2S_FLASH_ATTN = "1"
python benchmarks/aqua_ttfp.py --gpt-model F:/local/gpt.ckpt `
  --sovits-model F:/local/sovits.pth --sovits-pretrain F:/local/s2Gv3.pth `
  --bert-model F:/local/bert --cnhubert-model F:/local/cnhubert `
  --bigvgan-model F:/local/bigvgan --ref-audio F:/local/reference.wav `
  --ref-text "the reference transcript" --json-output F:/local/v3-flash.json
```

Set `AQUATTS_T2S_FLASH_ATTN=0` in a separate process for SDPA. Set
`GPT_SOVITS_HOME` as described above. The local evidence runner additionally
asserted BigVGAN's actual CUDA kernel, checked the measurement AST, and rejected
FA2 fallback before capture; it did not change successful timing or sampling.

The v2-family tables reuse `model_latency.py --steady --seed 20260926
--warmup-per-case 5 --repeats 20 --diagnostic-repeats 3`, with `--engine sdpa`
or `flash`. The config format is documented below. We also ran that protocol on
V3 with `--require-bigvgan-kernel` before this correction; those 120 post-trim requests are supplementary and do
not supply the V3 headline. Steady-protocol formal requests disable diagnostic
timing and capture no new graphs.

| Metric | Aqua Graph + SDPA | Aqua Graph + FA2 `valid` |
|---|---|---|
| T2S short / bucket 448 | 454.0 it/s | 669.6 it/s |
| T2S conversation / bucket 512 | 487.6 it/s | 715.1 it/s |
| T2S long / bucket 768 | 456.1 it/s | 688.1 it/s |
| V3 TTFP short (3) | 322.5 ms | 325.6 ms |
| V3 TTFP medium (19) | 448.9 ms | 378.8 ms |
| V3 TTFP long (64) | 582.9 ms | 442.1 ms |

The [complete report](results/readme-performance-20261021.md) keeps all five original-protocol
V3 samples per case plus the separate steady V3 p95, total time/RTF and first use.
Historical tables remain archived. Their raw-PCM endpoint is now preserved,
but software versions and unverified historical asset hashes still prevent a
controlled implementation speedup claim.
All three benchmark scripts now fsync every completed trial to
an adjacent `.trials.jsonl` file, outside the measured window, and record workload
hashes/source identity. Keep those journals and the GPU CSV if a run is interrupted.

## BigVGAN raw kernel timing

```bash
python benchmarks/bigvgan_raw_bench.py
```

The raw benchmark measures only the fp16 BigVGAN forward pass. Warm each mel
shape first, synchronize immediately around measured calls, and report medians
for mel lengths representative of first chunks and full utterances.

## Real-weight model compatibility

`model_smoke.py` checks actual checkpoint loading, non-streaming output, and
18 streaming trials (three Japanese/English texts × two seeds × dynamic/static/
CUDA Graph). It rejects silent/error fallback, checks output rate and finite
PCM, verifies actual graph replay, and verifies that Pro's main speaker reference
is encoded only once. `--flash` additionally requires active FlashAttention.
Use one process per model; weights and references are not included.

Create a local JSON config (absolute paths recommended):

```json
{
  "expected_version": "v2ProPlus",
  "inferencer": {
    "device": "cuda",
    "gpt_path": "/models/voice.ckpt",
    "sovits_path": "/models/voice.pth",
    "sv_model_path": "/models/pretrained_eres2netv2w24s4ep4.ckpt",
    "bert_path": "/models/chinese-roberta-wwm-ext-large",
    "cnhubert_path": "/models/chinese-hubert-base",
    "fast_langdetect_path": "/models/fast_langdetect"
  },
  "reference": {
    "ref_audio_path": "/voices/reference.wav",
    "prompt_text": "Matching reference transcript.",
    "prompt_language": "日文"
  }
}
```

```bash
GPT_SOVITS_HOME=/path/to/GPT-SoVITS python benchmarks/model_smoke.py \
  --config /path/to/local-config.json --output /path/to/local-results
```

The optional `texts` config field accepts a list of `text`/`text_language`
objects, for example to test Chinese using already-installed upstream G2PW
assets. Documented language labels are accepted independently of the UI locale.
V3 needs its BigVGAN directory and, for LoRA weights, its base SoVITS checkpoint.
Reports include checkpoint SHA-256 identities and timing observations, not a
controlled performance or perceptual-quality comparison. Keep audio and configs
local. See [the recorded validation](results/v2-model-support.md).

## Additional v2-family latency benchmarks

`model_latency.py` uses the same local config as `model_smoke.py` and the existing
short/medium/long TTFP texts. Run each engine in a fresh process:

```powershell
python benchmarks/model_latency.py --config F:\local\v2ProPlus.json --engine upstream --output F:\local\upstream.json
python benchmarks/model_latency.py --config F:\local\v2ProPlus.json --engine sdpa --output F:\local\sdpa.json
python benchmarks/model_latency.py --config F:\local\v2ProPlus.json --engine flash --output F:\local\flash.json
```

The upstream variant leaves native T2S unpatched but shares Aqua's frontend,
reference cache, semantic guard and SoVITS decoder. The other variants use CUDA
Graph with SDPA or FA2 `valid`; the FA2 benchmark fails on kernel fallback.
Two warmup utterances precede five seeded repeats per text. Results retain all
repeats, including first-shape initialization, and report median first audio,
total generation time, audio duration and RTF. The configured segmentation is
`按标点符号切`, output chunks are 0.25 s, and inter-segment pauses are 0.3 s.
This is model-side latency, excluding sound-device playback.

The [additional v2-family results](results/v2-model-latency.md) supplement the
existing v3 and mobile-GPU measurements above. Model/voice weights differ across
families; comparisons should be made between engines within one checkpoint pair.
Floating-point backend differences can change sampled tokens even with paired
seeds, so the report also includes semantic hashes and per-trial audio durations.

### Fixed-seed steady-state retest

Use `--steady` to separate each text's first use, five fixed-seed warmups, twenty
fixed-seed measurements, and three diagnostic runs. The measured steady trials
disable synchronized T2S timing and validate PCM after the timer stops; the
diagnostic runs separately time text processing, T2S and v2 acoustic decoding.
No measured steady trial may create another CUDA Graph. P95 uses the nearest
rank of the recorded samples, and token hashes expose workload variation.

```powershell
python benchmarks/model_latency.py --config F:/local/v2Pro.json --engine flash `
  --steady --seed 20260926 --warmup-per-case 5 --repeats 20 --diagnostic-repeats 3 `
  --gpu-uuid GPU-YOUR-DEVICE-UUID --telemetry-output F:/local/gpu.csv `
  --output F:/local/steady.json
```

`nvidia-smi -L` lists physical GPU UUIDs. The UUID selects CUDA device 0 inside
this benchmark process and records that same GPU's utilization, clocks, memory,
power and temperature every 500 ms. Three seconds of pre-load activity are
recorded first. The tool does not close other applications or change GPU power
settings; telemetry therefore describes the actual desktop environment, not an
isolated GPU. The [2026-10-21 Aqua-only retest](results/readme-performance-20261021.md)
uses this protocol for the current v2-family README tables. The [2026-09-27 retest](results/v2-model-retest-20260927.md) preserves
the previous measurements and documents the changed measurement protocol.

### Graph KV storage layouts

Use a fresh process for each layout. The probe loads an existing GPT checkpoint,
captures synthetic SDPA decode steps, and reports Graph count, KV storage deduplicated
by backing storage, allocator totals and one preparation duration. It requires
`GPT_SOVITS_HOME`; it does not download weights or need reference voice material.

```powershell
python benchmarks/kv_layout_bench.py --gpt-model F:/local/gpt.ckpt --layout bucket --output F:/local/bucket.json
python benchmarks/kv_layout_bench.py --gpt-model F:/local/gpt.ckpt --layout nested --output F:/local/nested.json
python benchmarks/kv_layout_bench.py --gpt-model F:/local/gpt.ckpt --layout nested --precapture-buckets 448 --output F:/local/sparse.json
```

`--preset full|minimal|lazy|off` uses that preset's startup policy. An explicit
`--precapture-buckets` changes preparation targets, while the declared capacity
buckets remain unchanged. To compare the pre-change implementation, export
`37825477a8adb480d6ef02b37fc77bbad6fa7444:aquatts/modeling/t2s_streaming.py`
as UTF-8 and pass `--layout legacy --baseline-source path/to/snapshot.py`.

With the validated 24-layer, hidden-512 FP16 checkpoint and full preparation,
legacy has 15 graphs and 519 MiB unique KV, bucket has six graphs and 147 MiB,
and nested has six graphs and 48 MiB. Preparing only a 448 bucket uses 21 MiB
with bucket and 48 MiB with nested. These are KV figures; they exclude model
weights, Graph intermediates and other allocations. A single preparation sample
does not establish a startup latency distribution or an inference speedup.

### Prepared nested promotion

Run `kv_promotion_bench.py` in separate processes against current source and an
exported N2 source (`58cbec2347e15357ad50c9e0ab30152476366d70` path
`aquatts/modeling/t2s_streaming.py`). It forces fixed 51/224/600 decode steps with
synthetic conditioning, disables EOS, prepares 448/512/768/1024, and records every
trial plus path counters. FA2 rejects silent SDPA decode fallback during capture
and runtime; `--flash-attn off` selects SDPA. Use reversed process order to check
drift. These measurements describe T2S work and exclude acoustic synthesis.

```powershell
python benchmarks/kv_promotion_bench.py --gpt-model F:/local/gpt.ckpt --output F:/local/current.json
python benchmarks/kv_promotion_bench.py --gpt-model F:/local/gpt.ckpt --baseline-source F:/local/n2.py --output F:/local/baseline.json
```

### Optional BERT and AP-BWE loading

`optional_models_smoke.py` runs English, Japanese and Chinese streaming requests
with local V3 weights while rejecting every AP-BWE import. Run fresh processes
with and without `--preload-bert` to record startup, allocated/reserved CUDA memory,
RSS, first successful BERT load time and per-language PCM hashes. One run per mode
is a functional/resource check, not a latency distribution.

```powershell
python benchmarks/optional_models_smoke.py --config F:/local/v3.json --output F:/local/lazy
python benchmarks/optional_models_smoke.py --config F:/local/v3.json --preload-bert --output F:/local/preload
```

The selected upstream Chinese frontend also needs its G2PW resources. If it uses
cwd-relative paths, `--frontend-cwd F:/local/existing-upstream-assets` selects an
existing resource root for this standalone benchmark process. The result records
that condition; it does not establish arbitrary-cwd or fully offline frontend support.

### Reference warmup

`reference_warmup_smoke.py` supplies a `Voice` at construction, then checks the
public warmup hit and both inference APIs. It counts SSL, reference spectrum,
reference `decode_encp` and Pro speaker encoding, and verifies that explicit clear
causes exactly one rebuild. Timings are single-run observations.

```powershell
python benchmarks/reference_warmup_smoke.py --config F:/local/v3.json --output F:/local/reference-v3
python benchmarks/reference_warmup_smoke.py --config F:/local/v2Pro.json --output F:/local/reference-pro
```

### CUDA vocoder preparation

Run `vocoder_warmup_smoke.py` in fresh processes for `--warmup default`, `custom`
or `off`; add `--kernel pytorch --graph-preset off` for the explicit fallback
configuration. It checks real-model warmup RNG preservation, records the first
and two subsequent calls for six mel lengths, and synthesizes 0.1/0.25/0.5-second
and complete chunks. Request checks reject explicit device synchronization when
diagnostic timing is off. Request timings follow synthetic shape probes and
reference warmup, so they are not cold end-to-end measurements.

```powershell
python benchmarks/vocoder_warmup_smoke.py --config F:/local/v3.json --warmup default --output F:/local/default.json
python benchmarks/vocoder_warmup_smoke.py --config F:/local/v3.json --warmup custom --output F:/local/custom.json
python benchmarks/vocoder_warmup_smoke.py --config F:/local/v3.json --warmup off --kernel pytorch --graph-preset off --output F:/local/off.json
```

### CFM unpadded experiment

`cfm_unpadded_bench.py` compares the original masked estimator with the instance
adapter using identical weights, inputs and seeded noise. It records baseline
repeat error before candidate measurement, mel/PCM errors against fixed bounds,
and three alternating timing samples after warmup. The matrix covers FP16/FP32,
4/16/32 steps, CFG off/on, short tails and long references. FP32 computation uses
the same loaded FP16 weights converted to FP32. Long-reference feature repetition
is a numerical stress case, not an intelligible-speech fixture.

It also runs three paired warmed requests for ordinary synthesis and
0.1/0.25/0.5-second streaming, retaining samples, first nonempty/audible PCM,
total duration and RTF. These small samples do not establish P95 or perceptual
equivalence. Listen to the locally saved pairs separately; no voice assets are
committed. Run GPU measurements without concurrent GPU tests.

```powershell
python benchmarks/cfm_unpadded_bench.py --config F:/local/v3.json --output F:/local/cfm
```

### V3 mixed references

`mixed_reference_smoke.py` prewarms two supplied references and checks ordinary
and streaming synthesis across omitted, identical, mixed, reversed and restored
references. It verifies the source of all seven composed fields, unchanged shared
cache values, exactly two reference builds, and identical baseline/same/restored
PCM. The second audio may be an explicitly identified synthetic fixture with a
known transcript; such a test establishes routing, not emotion or speaker quality.

```powershell
python benchmarks/mixed_reference_smoke.py --config F:/local/v3.json --semantic-audio F:/local/semantic.wav --semantic-text "Its transcript." --semantic-language "英文" --output F:/local/mixed
```

### External audit concurrency regressions

`audit_concurrency_smoke.py` loads two real V3 instances. A suspended stream
keeps work admission while a second instance constructs and defers pre-capture.
It then exercises two concurrent `/tts` requests on one instance, concurrent
requests across both instances, and a successful deferred-capture retry. The
real sampler and CFM RNG remain enabled, and `torchaudio.load` is blocked to
verify SoundFile reference decoding. These are functional checks, not latency
comparisons. Run the GPU unit regressions separately, without a concurrent
benchmark process.

```powershell
python benchmarks/audit_concurrency_smoke.py --config F:/local/v3.json --output F:/local/concurrency.json
```

### Real multi-GPU correctness (explicit opt-in)

`tests/test_multi_gpu_runtime.py` is skipped **before importing Torch** unless
`AQUATTS_RUN_MULTI_GPU=1`. An explicitly requested run fails if fewer than two
CUDA devices are visible. With two visible GPUs it covers 24 cases, including
both current/target directions:

- Actual device resolution and allocation with current GPU A and target GPU B.
- Vocoder warmup success/failure, ordinary forward calls, loader selection and
  kernel-load fallback; restoration of the caller's current device and warmup's
  CPU/all-visible-GPU RNG state.
- T2S capture on A then B and B then A, alternating replay, and Graph/eager
  greedy-token equivalence for both `bucket` and `nested` layouts.
- Concurrent prepared replay on separate GPUs using the real sampler, plus
  capture deferral while another GPU has active work and successful idle retry.

All CUDA device APIs, allocations, synchronization, capture and replay are real.
The T2S decoder is a small upstream model with random weights. The vocoder is a
small Torch module; its loader factory replaces checkpoint IO and injects a
kernel-load failure. These tests do not validate a pretrained BigVGAN checkpoint,
extension compilation, speech quality, or performance.

Wait until **both selected GPUs are idle**. Do not run this suite alongside a
benchmark, training job, or inference server. `multi_gpu_validation.py` does not
decide whether hardware is idle; check utilization and process ownership before
passing `--execute`. In particular, do not infer availability from one low
utilization sample while another process still owns the eGPU.

The runner's default invocation only writes a plan and does not import Torch or
probe CUDA. Use a new output directory for each invocation:

```powershell
# Safe preparation: no GPU access.
python benchmarks/multi_gpu_validation.py --output F:/local/multi-gpu-plan

# Only after the selected GPUs are confirmed idle. UUIDs are local physical IDs.
$env:CUDA_VISIBLE_DEVICES = "<laptop-uuid>,<ti-super-uuid>"
$env:GPT_SOVITS_HOME = "F:/local/GPT-SoVITS"
python benchmarks/multi_gpu_validation.py --execute --output F:/local/multi-gpu-run
```

Execution requires a clean committed checkout and exactly two distinct physical
GPU UUIDs in `CUDA_VISIBLE_DEVICES`. It runs the dedicated multi-GPU suite first,
then the remaining full runtime suite twice, reversing the UUID order for the
second run. Both GPUs remain visible; each physical GPU gets a turn as `cuda:0`
so existing tests fixed to that index exercise both cards.
Each phase has a fresh process and a 900-second timeout. A workspace lock prevents
overlapping invocations of this runner. Do not remove an existing lock until its
recorded process has been confirmed to have exited.

Every test start and setup/call/teardown result is appended and fsynced to
`<phase>/events.jsonl`; stdout/stderr stream to `<phase>/output.log`.
`manifest.json` records the source commit, selected UUIDs, phase status and
outcomes. A timeout, driver loss, failure or missing result stops the remaining
phases and cannot produce a passing manifest. Skips are rejected in the dedicated
multi-GPU phase; conditional skips in the general runtime phase remain visible.
Preserve incomplete runs for diagnosis. This suite's hardware result stays
**pending** until an executed run's manifest reports `passed`.
