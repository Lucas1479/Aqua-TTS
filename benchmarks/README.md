# Spectralis Benchmark Suite

Fair comparison of Spectralis vs official GPT-SoVITS inference performance.

## Prerequisites

### 1. Clone official GPT-SoVITS

```bash
git clone https://github.com/RVC-Boss/GPT-SoVITS.git official-gpt-sovits
cd official-gpt-sovits
pip install -r requirements.txt
```

### 2. Prepare model weights

- GPT weights (T2S): `s1v3.ckpt` (official) or `xxx-e15.ckpt` (self-trained)
- SoVITS weights: `xxx_e2_s174_l32.pth` (with LoRA)
- Reference audio: 3-10 second .wav file

## Benchmarks

### Spectralis — All optimizations

```bash
python benchmarks/spectralis_ttfp.py \
    --gpt-model GPT_weights_v3/xxx-e15.ckpt \
    --sovits-model SoVITS_weights_v3/xxx_e2_s174_l32.pth \
    --ref-audio "reference audio/kurisu_reference2.wav" \
    --ref-text "reference transcript"
```

### Spectralis — Ablation (disable optimizations)

```bash
# No CUDA Graph, no BigVGAN kernel
python benchmarks/spectralis_ttfp.py ... --no-cuda-graph --no-bigvgan-kernel

# All off (equivalent to official path)
python benchmarks/spectralis_ttfp.py ... --no-cuda-graph --no-static-kv --no-bigvgan-kernel
```

### T2S throughput

```bash
python benchmarks/t2s_speed_bench.py --gpt-model GPT_weights_v3/xxx-e15.ckpt
```

## Fair Comparison Principles

1. **Same weights**: Use identical GPT + SoVITS weights for official and Spectralis
2. **Same GPU**: All tests on same device
3. **Same texts**: Three benchmark texts (short/medium/long)
4. **Multiple runs**: 5 repeats, median reported
5. **Warmup**: 2 warmup runs before timing

## Test Texts

| Case | Length | Text |
|------|--------|------|
| short | 3 chars | 実験？ |
| medium | 19 chars | 何の実験だか気になるけど、まあいいわ。 |
| long | 64 chars | Paxosが提案ベースでメッセージフローが複雑なのに対して、Raftは明確なリーダー選出とログ複製のフェーズに分けられているわ。 |

## Metrics

- **TTFP (Time-To-First-Packet)**: Latency from text input to first audio chunk (ms)
- **Total**: Full inference wall time (ms)
- **T2S Speed**: Text-to-Semantic decoding throughput (it/s)

## Results (RTX 4070 Ti SUPER, float16)

| Metric | Official (no Graph) | Official (CUDA Graph) | Spectralis |
|--------|---------------------|-----------------------|------------|
| T2S throughput | ~90-120 it/s | ~244 it/s | **370-440 it/s** |
| TTFP Short | 1061ms | 1016ms | **424ms** |
| TTFP Medium | 1599ms | 1476ms | **501ms** |
| TTFP Long | 3598ms | 2852ms | **648ms** |
| Memory safety | OOM (long texts) | OOM (long texts) | Static KV |
| BigVGAN | PyTorch JIT | PyTorch JIT | CUDA Kernel |
| V3 streaming | return_fragment | return_fragment | True streaming |

*Measured with same weights (xxx-e15.ckpt + xxx_e2_s174_l32.pth).*
