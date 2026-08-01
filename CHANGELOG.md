# Changelog

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

[0.2.0]: https://github.com/Lucas1479/Aqua-TTS/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Lucas1479/Aqua-TTS/releases/tag/v0.1.0
