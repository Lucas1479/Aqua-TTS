# Changelog

## Unreleased

## [0.3.0] — 2026-10-21

### Added

- Optional nested KV storage shares a maximum-size root across bucket views.
  Requests can promote to larger already captured buckets without copying live
  KV, while newly exposed tails are cleared and fallback preserves full history.
  The default remains independent `bucket` storage.
- Public `warm_reference_cache()` and `clear_reference_cache()` APIs, optional
  constructor reference warmup, bounded cache residency, file/model/device/dtype
  invalidation, atomic builds and CUDA-event readiness across streams.
- V3 `semantic_reference` support lets semantic prompts and acoustic conditioning
  use independently selected references in both ordinary and streaming inference.
- Per-instance `offline_only` model/resource checks for supported local-loading
  paths, including actual AP-BWE resource locations. Online segmentation retains
  its upstream behavior; offline configuration does not change global Hub policy.
- Explicit BERT preloading through `preload_bert=True` / `preload_bert_model()`,
  with observable readiness, concurrent first-use protection and retry.
- Bounded, configurable BigVGAN warmup shapes/repeats with RNG preservation,
  preparation status and actual request shape counts.
- Experimental V3 single-item `cfm_unpadded=True` inference, preserving conditioner
  caches. It remains disabled by default.

### Changed

- Reuse one CUDA Graph per computation bucket across real prefix lengths.
- Coordinate GPU work and capture admission across requests and instances.
  Capture is deferred while other work is active; affected requests can use
  eager inference. Complete pre-capture before serving sustained concurrent work.
- Load Chinese BERT only when needed, and defer AP-BWE imports/model construction
  until super-resolution is requested.
- Reuse one request-local semantic-history buffer and skip EOS host reads during
  the first 11 steps where EOS is masked out. Sampling and RNG behavior are preserved.
- Retain historical first-raw-PCM timing in README benchmarks. Current Ti SUPER
  measurements use Torch 2.6.0+cu124 and the existing compiled BigVGAN kernel;
  the labeled pre-fix RTX 4070 Laptop measurements are retained pending a later retest.

### Fixed

- Inference requests inherit the instance CUDA Graph policy unless explicitly
  overridden; disabling Graphs also disables startup pre-capture.
- Resolve unindexed CUDA to the actual current device, reject unavailable indexes
  before loading models, and keep capture streams, extension caches and vocoder
  calls on the selected device.
- Own shared Graph KV and logits for the complete AR request, with CUDA-event
  stream handoff, exception-safe release and request-local counters.
- Release V3 decode ownership before yielding audio, and serialize HTTP generator
  advancement/closure so worker-thread changes, disconnects and cancellation clean up correctly.
- Validate both newly built and cached references; propagate reference failures
  before streaming headers rather than continuing with incomplete conditioning.
- Read V3 reference audio through soundfile instead of torchaudio.load, avoiding
  a TorchCodec requirement on newer torchaudio versions.
- Slice semantic output using the actual prompt length, including early-stop cases.
- Synchronize the selected device only for diagnostic timing; normal complete V3
  stream chunks avoid device synchronization for unused timing output.

## [0.2.1] — 2026-10-06

### Added

- v2/v2Pro/v2ProPlus checkpoint selection through upstream metadata, including
  renamed weights; Pro/Plus reuse upstream ERes2Net with an explicit `sv_model_path`.
- Speaker reference caching and paired multi-reference conditioning for both
  inference APIs, without loading v3-only assets for v2-family models.
- A real-weight smoke matrix for ordinary and streaming inference across dynamic,
  static-KV, and CUDA Graph modes.

### Changed

- Replaced the vendored GPT-SoVITS `t2s_model.py` fork with an in-memory patch
  over a validated upstream v3 checkout.
- Preserved upstream `infer_panel_naive()` batching/streaming generator
  semantics; Aqua now replaces only the direct `infer_panel()` path.
- Added the public `configure_gpt_sovits()` upstream-routing API for embedding
  Aqua in products such as Amadeus.
- Added explicit SoVITS base, BigVGAN, and fast-langdetect asset paths plus a
  product-owned `BIGVGAN_CACHE_ROOT` for embedded runtimes.
- Consolidated BigVGAN CUDA extension loading into one canonical Aqua module.
- Replaced EOS-dependent legacy T2S benchmarks with deterministic cold,
  bucket-448, bucket-512, and bucket-768 synchronized measurements, including
  isolated current-upstream and optional FlashAttention2 comparisons.
- Prefer FlashAttention2 `valid` automatically when it is importable, with
  transparent SDPA fallback and an explicit `AQUATTS_T2S_FLASH_ATTN=0` opt-out.
- Restored the public single-table highlight format and expanded it with
  current throughput, warm TTFP, and execution-mechanism comparisons.

### Fixed

- Remove low-energy generated lead before each synthesized item while retaining
  50 ms of preroll and the caller's `pause_second`. Complete and chunked output
  share the same 10 ms RMS boundary; quiet items remain intact.
- Keep playback demo TTFP as first PCM availability and additionally report
  `lead_ms` and `first_voiced_chunk_ms`; neither is an acoustic loopback clock.
- Return the actual sample rate in HTTP PCM streaming headers and propagate
  synthesis errors instead of emitting a silent fallback block.
- Include HTTP endpoint tests in model-free CI.
- Keep documented inference language labels independent of the host UI locale,
  and use the model's sample rate in playback examples.
- Mask unused static/CUDA-Graph KV slots and keep graph-key alignment gaps out
  of SDPA and FlashAttention2 attention.
- Preserve the complete prompt and generated context when a fixed KV bucket
  fills, continuing with dynamic KV instead of evicting the oldest entries.
- Make CUDA Graph capture sampling-RNG neutral.
- Reject and retry pathological semantic-token loops before SoVITS/BigVGAN.
- Preserve the v3 SoVITS acoustic architecture independently of its v2 text symbols.
- Preserve existing Japanese comma endings in prompt and generated text segments.
- Removed the default device-wide synchronize after every CUDA Graph replay;
  `CUDA_GRAPH_REPLAY_SYNC=1` retains it as a diagnostic.
- Folded greedy and sampled EOS checks into one device-to-host sync.
- Cached BigVGAN extensions no longer require a locally installed CUDA Toolkit.
- BigVGAN cache identities now include Python, Torch, and CUDA versions to
  avoid loading ABI-incompatible binaries after runtime upgrades.

## [0.2.0] — 2026-08-02

### Added

- PyPI-ready wheel and source distribution metadata
- Trusted Publishing workflows for TestPyPI and PyPI
- Package-build verification in continuous integration

### Changed

- PyPI installation is now the primary documented installation path
- Version metadata now has a single source in `aquatts.__version__`
- Importing Aqua-TTS no longer permanently changes the caller's working directory
- Packaging metadata and manifest paths were modernized

### Fixed

- Corrected the GENIE-TTS repository link
- Corrected source-distribution manifest paths and encoding
- Resolved the existing Ruff CI failures

## [0.1.0] — 2026-05-24

### Added

- Static KV cache with 6 bucketed sizes (128–1024) for OOM-free long-text generation
- CUDA Graph capture and replay for T2S AR decoder (~4x throughput improvement)
- Pre-compiled BigVGAN CUDA kernel loader with MSVC auto-detection and per-GPU caching
- Streaming inference with true chunk-by-chunk audio output
- `apply_fade_in` / `apply_fade_out` / `finalize_stream_chunk` utilities for click-free playback
- Inference parameter presets tuned by text length and sentence position
- Graceful degradation chain: CUDA Graph → static KV cache → dynamic `torch.cat` fallback
- TTFP and T2S throughput benchmarks with ablation flags
- Example scripts for basic and streaming inference

[0.3.0]: https://github.com/Lucas1479/Aqua-TTS/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/Lucas1479/Aqua-TTS/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Lucas1479/Aqua-TTS/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Lucas1479/Aqua-TTS/releases/tag/v0.1.0
