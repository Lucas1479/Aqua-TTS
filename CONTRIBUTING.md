# Contributing to Aqua-TTS

## Setup

```bash
git clone https://github.com/SiqiLiOcean/Aqua-TTS.git
cd Aqua-TTS
pip install -e ".[server]"
pip install -r requirements-dev.txt
```

## Running tests

```bash
python -m pytest tests/ -v
```

## Code style

This project uses [ruff](https://github.com/astral-sh/ruff) for linting:

```bash
ruff check Aqua/ tests/
```

## Project structure

```
Aqua/
  modeling/      Static KV cache + CUDA Graph patches for T2S decoder
  bigvgan/
    cuda/        Pre-compiled BigVGAN CUDA kernel loader
    torch/       Anti-alias activation (pure PyTorch fallback)
  inference/     Streaming inference helpers and parameter presets
```

## License

By contributing, you agree that your contributions will be licensed under the MIT License. Third-party code in this repository retains its original license; see `NOTICE` for details.
