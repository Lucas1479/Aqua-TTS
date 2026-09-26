# V2-family compatibility validation — 2026-09-26

Validated on Windows, Python 3.12.10, PyTorch 2.5.1+cu121, RTX 4070 Ti SUPER
(16 GB), and upstream GPT-SoVITS `08d627c3338173c3229286d8787060d6559fe0f8`.
The Aqua branch is based on `ebf45da`, which includes semantic-stability PR #4.
These are compatibility checks, not a controlled speed or perceptual-quality
comparison. No MPS or ROCm validation or implementation is included.

## Results

- Full local suite with the configured upstream/runtime: **147 passed, 1 skipped**.
  The skip is the inverse, unconfigured-upstream import test.
- Separate environment without Torch/upstream: **93 passed, 8 skipped** using the
  model-free CI command (including HTTP and playback sample-rate tests).
- Ruff, `git diff --check`, wheel/sdist build, and `twine check`: passed.
- Installed wheel: imports and adapter contents checked; no checkpoints, audio,
  compiled binaries, vendored ERes2Net, or vendored T2S model are included.
- Installed wheel + real Kurisu Plus weights + FastAPI TestClient: both `/tts`
  PCM and `/tts/file` WAV passed at **32,000 Hz**, with 78,720 finite non-silent
  samples/frames per response in this run. No physical playback is asserted.
- GPU matrix: **12 non-streaming runs + 180 streaming trials passed**. Repeated
  exploratory runs are excluded from this count.

| Checkpoint pair | Japanese/English SDPA trials | Chinese SDPA trials | Japanese/English FA2 trials | Output |
|---|---:|---:|---:|---:|
| Official v2 GPT + `s2G2333k.pth` | 18 | 6 | 18 | 32 kHz |
| `kurisu_v2pro-e15.ckpt` + `kurisu_v2pro.pth` | 18 | 6 | 18 | 32 kHz |
| `kurisu_crs_gpt-e100.ckpt` + `kurisu_crs_s2-e12_v2ProPlus.pth` | 18 | 6 | 18 | 32 kHz |
| Official v2 GPT + `s2Gv2Pro.pth` | 18 | — | — | 32 kHz |
| Official v2 GPT + `s2Gv2ProPlus.pth` | 18 | — | — | 32 kHz |
| Existing v3 GPT + `xxx_e2_s174_l32.pth` LoRA | 18 | — | — | 24 kHz |

Official v2 GPT is `s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt`.
Pro/Plus use `pretrained_eres2netv2w24s4ep4.ckpt`. V2-family assets are supplied
from the maintainer's existing local GPT-SoVITS v2pro package; earlier duplicated
local v2/Pro weights were confirmed to have matching SHA-256 identities. The
voice material and model weights are not part of this change.

Each 18-trial row runs three texts × seeds 7/17 × dynamic/static/CUDA Graph.
The Chinese row uses one text × the same two seeds × three execution modes.
Every invocation also exercises `infer()` once. Graph capture **and replay** are
checked; FA2 trials require active `valid` mode. Every Pro/Plus invocation checks
that the primary reference speaker embedding is computed once, then reused.

Assertions check architecture, sample rate, finite float32 PCM, non-silence,
semantic-generation statistics, and absence of logged inference errors. Unit
tests cover missing conditioning weights, unsupported checkpoint headers,
renamed checkpoint detection, multi-reference pairing/fallback, cache isolation,
error propagation, locale-independent API language labels, and HTTP/playback
sample rates. Passing these checks does not establish voice fidelity or naturalness.

## Reproduction and evidence

Use `benchmarks/model_smoke.py` with the local config described in
[the benchmark guide](../README.md#real-weight-model-compatibility). Run each
model in a fresh process, with the matching reference recording/transcript.
For Chinese, this run used the local package's existing G2PW assets and BERT
directory; no frontend or model downloads are needed when those assets exist.

```powershell
$env:GPT_SOVITS_HOME = 'F:\path\to\GPT-SoVITS'
python benchmarks/model_smoke.py --config F:\local\v2ProPlus.json --output F:\local\results
python benchmarks/model_smoke.py --config F:\local\v2ProPlus.json --output F:\local\results-fa --flash
python -m pytest tests -q
python -m ruff check aquatts tests benchmarks/model_smoke.py examples
python -m build
python -m twine check dist/*
```

[Machine-readable results](v2-model-support.json) preserve each trial, checkpoint
SHA-256 identities, rate, cache count, graph count, and FA2 selection. Timings
include cold frontend initialization and lazy graph capture where applicable;
they must not be compared directly with the warm v3 headline benchmarks.
Audio, absolute local paths, and reference transcripts are intentionally omitted.

V2-family streaming currently decodes each complete text segment before PCM
chunking. This validation does not claim upstream incremental `decode_streaming`
support. V1 compatibility is covered by unit contracts, not real-weight runs in
this matrix; v4 is explicitly unsupported.
