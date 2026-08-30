# Changelog

## [0.2.1] — 2026-08-30

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
- Expanded the public highlights into separate throughput, warm TTFP, and
  execution-mechanism matrices.

### Fixed

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

[0.2.1]: https://github.com/Lucas1479/Aqua-TTS/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Lucas1479/Aqua-TTS/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Lucas1479/Aqua-TTS/releases/tag/v0.1.0
