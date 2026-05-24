# Changelog

## [1.0.0] — 2025-05-24

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

[1.0.0]: https://github.com/SiqiLiOcean/Aqua-TTS/releases/tag/v1.0.0
