# -*- coding: utf-8 -*-
"""T2S throughput comparison: Official vs Official CUDA Graph vs Spectralis.

Each variant runs in its own subprocess to avoid import conflicts.

Usage:
    python benchmarks/t2s_comparison_bench.py --gpt-model GPT_weights_v3/xxx-e15.ckpt
"""
from __future__ import annotations

import argparse, os, statistics, subprocess, sys, tempfile, textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFFICIAL_REPO = r"F:\Computer_Science\GPT-SoVITS-official"
MAIN_REPO = r"F:\BaiduNetdiskDownload\GPT-SoVITS\GPT-SoVITS-v3lora-20250401"
MAIN_GPT_SOVITS = os.path.join(MAIN_REPO, "GPT_SoVITS")
TOKEN_COUNT = 500
REPEATS = 5


def _run_official(gpt_model: str, timeout: int = 300) -> dict:
    """Benchmark official GPT-SoVITS (torch.cat KV cache, no CUDA Graph)."""
    script = textwrap.dedent(f"""\
import os, statistics, sys, time
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")
OFFICIAL = {OFFICIAL_REPO!r}
sys.path.insert(0, OFFICIAL)
sys.path.insert(0, os.path.join(OFFICIAL, "GPT_SoVITS"))
os.chdir(OFFICIAL)

# Suppress tqdm before importing torch/tqdm
from functools import partial as _partial
class _NoopTqdm:
    def __init__(self, iterable=None, *a, **kw):
        self._it = iter(iterable) if iterable is not None else iter([])
    def __iter__(self): return self
    def __next__(self): return next(self._it)
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def update(self, *a, **kw): pass
    def close(self): pass
    def set_description(self, *a): pass
    @staticmethod
    def write(*a, **kw): pass
import tqdm as _tqdm
_tqdm.tqdm = _NoopTqdm

import torch
from GPT_SoVITS.AR.models.t2s_lightning_module import Text2SemanticLightningModule

print("[official] Loading...", flush=True)
d = torch.load({gpt_model!r}, map_location="cpu")
cfg = d["config"]
m = Text2SemanticLightningModule(cfg, "****", is_train=False)
m.load_state_dict(d["weight"])
m = m.half().cuda().eval()

x = torch.randint(0, cfg["model"]["phoneme_vocab_size"], (1, 20)).long().cuda()
xl = torch.tensor([20]).long().cuda()
p = torch.randint(0, cfg["model"]["vocab_size"], (1, 3)).long().cuda()
b = torch.randn(1, 1024, 20).half().cuda()

# official infer_panel_naive is a generator — consume it to get (y, idx)
def _run_infer(model, *a, **kw):
    for _y, _steps in model.infer_panel_naive(*a, **kw):
        pass
    return _y, _steps

print("[official] Warmup...", flush=True)
for _ in range(3):
    _run_infer(m.model, x, xl, p, b, top_k=5, top_p=1, temperature=0.6)

print(f"[official] {REPEATS} trials...", flush=True)
rates = []
for trial in range({REPEATS}):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _, steps = _run_infer(m.model, x, xl, p, b, top_k=5, top_p=1, temperature=0.6)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    r = steps / dt if dt > 0 else 0
    rates.append(r)
    print(f"  trial {{trial+1}}: {{steps}} tok in {{dt:.3f}}s = {{r:.0f}} it/s", flush=True)

med = statistics.median(rates)
print(f"RESULT: median={{med:.0f}}", flush=True)
""")
    return _run_script(script, timeout)


def _run_official_cudagraph(gpt_model: str, timeout: int = 300) -> dict:
    """Benchmark official GPT-SoVITS CUDA Graph runner."""
    script = textwrap.dedent(f"""\
import os, statistics, sys, time
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")
OFFICIAL = {OFFICIAL_REPO!r}
sys.path.insert(0, OFFICIAL)
sys.path.insert(0, os.path.join(OFFICIAL, "GPT_SoVITS"))
os.chdir(OFFICIAL)

class _NoopTqdm:
    def __init__(self, iterable=None, *a, **kw):
        self._it = iter(iterable) if iterable is not None else iter([])
    def __iter__(self): return self
    def __next__(self): return next(self._it)
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def update(self, *a, **kw): pass
    def close(self): pass
    def set_description(self, *a): pass
    @staticmethod
    def write(*a, **kw): pass
import tqdm as _tqdm
_tqdm.tqdm = _NoopTqdm

import torch
from AR.models.t2s_model_cudagraph import CUDAGraphRunner
from AR.models.structs_cudagraph import T2SRequest

print("[official+cuda] Loading...", flush=True)
runner = CUDAGraphRunner(
    CUDAGraphRunner.load_decoder({gpt_model!r}),
    torch.device("cuda"),
    torch.float16,
)

d = torch.load({gpt_model!r}, map_location="cpu")
cfg = d["config"]
del d
x0 = torch.randint(0, cfg["model"]["phoneme_vocab_size"], (20,)).long().cuda()
xl = torch.tensor([20]).long().cuda()
prompt_tok = torch.randint(0, cfg["model"]["vocab_size"], (1, 3)).long().cuda()
bf = torch.randn(1024, 20).half().cuda()

request = T2SRequest(
    x=[x0], x_lens=xl, prompts=prompt_tok, bert_feature=[bf],
    valid_length=1, top_k=5, top_p=1.0, temperature=0.6,
    repetition_penalty=1.0, early_stop_num=-1, use_cuda_graph=True,
)

print("[official+cuda] Warmup...", flush=True)
for _ in range(3):
    runner.generate(request)

print(f"[official+cuda] {REPEATS} trials...", flush=True)
rates = []
for trial in range({REPEATS}):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = runner.generate(request)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    if result.exception is not None:
        print(f"  ERROR: {{result.exception}}", flush=True)
        if result.traceback:
            print(f"  TRACEBACK: {{result.traceback[-600:]}}", flush=True)
        rates.append(0.0)
        continue
    tokens = sum(r.size(0) for r in result.result if r is not None)
    rate = tokens / dt if dt > 0 else 0
    rates.append(rate)
    print(f"  trial {{trial+1}}: {{tokens}} tok in {{dt:.3f}}s = {{rate:.0f}} it/s", flush=True)

med = statistics.median(rates)
print(f"RESULT: median={{med:.0f}}", flush=True)
""")
    return _run_script(script, timeout)


def _run_spectralis(gpt_model: str, timeout: int = 300) -> dict:
    """Benchmark Spectralis-vendored model (static KV + pre-captured CUDA Graph)."""
    script = textwrap.dedent(f"""\
import os, statistics, sys, time
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ["ENABLE_CUDA_GRAPH"] = "1"
os.environ["ENABLE_CUDA_GRAPH_PRECAPTURE"] = "1"

ROOT = {ROOT!r}
MAIN_REPO = {MAIN_REPO!r}
MAIN_GPT_SOVITS = os.path.join(MAIN_REPO, "GPT_SoVITS")
# Vendored overrides must come BEFORE main GPT_SoVITS for t2s_model.py override
sys.path.insert(0, os.path.join(ROOT, "spectralis", "_vendor"))
sys.path.insert(0, ROOT)
sys.path.insert(0, MAIN_REPO)
sys.path.insert(0, MAIN_GPT_SOVITS)
os.chdir(MAIN_REPO)

class _NoopTqdm:
    def __init__(self, iterable=None, *a, **kw):
        self._it = iter(iterable) if iterable is not None else iter([])
    def __iter__(self): return self
    def __next__(self): return next(self._it)
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def update(self, *a, **kw): pass
    def close(self): pass
    def set_description(self, *a): pass
    @staticmethod
    def write(*a, **kw): pass
import tqdm as _tqdm
_tqdm.tqdm = _NoopTqdm

import torch
from GPT_SoVITS.AR.models.t2s_lightning_module import Text2SemanticLightningModule

print("[spectralis] Loading...", flush=True)
d = torch.load({gpt_model!r}, map_location="cpu")
cfg = d["config"]
m = Text2SemanticLightningModule(cfg, "****", is_train=False)
m.load_state_dict(d["weight"])
m = m.half().cuda().eval()

from spectralis.modeling import apply_cuda_graph_patch
apply_cuda_graph_patch(m.model)
m.model.precapture_cuda_graph()

x = torch.randint(0, cfg["model"]["phoneme_vocab_size"], (1, 20)).long().cuda()
xl = torch.tensor([20]).long().cuda()
p = torch.randint(0, cfg["model"]["vocab_size"], (1, 3)).long().cuda()
b = torch.randn(1, 1024, 20).half().cuda()

print("[spectralis] Warmup...", flush=True)
for _ in range(3):
    m.model.infer_panel_naive(x, xl, p, b, top_k=5, top_p=1, temperature=0.6)

print(f"[spectralis] {REPEATS} trials...", flush=True)
rates = []
for trial in range({REPEATS}):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _, steps = m.model.infer_panel_naive(x, xl, p, b, top_k=5, top_p=1, temperature=0.6)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    r = steps / dt if dt > 0 else 0
    rates.append(r)
    print(f"  trial {{trial+1}}: {{steps}} tok in {{dt:.3f}}s = {{r:.0f}} it/s", flush=True)

med = statistics.median(rates)
print(f"RESULT: median={{med:.0f}}", flush=True)
""")
    return _run_script(script, timeout)


def _run_script(script: str, timeout: int) -> dict:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(script)
        tmp_path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout,
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        for line in stdout.splitlines():
            print(f"  | {line}")
        if stderr.strip():
            print(f"  [stderr] {stderr[:2000]}", file=sys.stderr)

        result = {}
        for line in stdout.splitlines():
            if line.startswith("RESULT:"):
                for p in line.split()[1:]:
                    k, v = p.split("=")
                    result[k] = float(v)
        if "median" not in result:
            print(f"  WARNING: no RESULT line — check stderr above")
            result["median"] = 0.0
        return result
    finally:
        os.unlink(tmp_path)


def main():
    parser = argparse.ArgumentParser(description="T2S comparison benchmark")
    parser.add_argument("--gpt-model", required=True)
    parser.add_argument("--skip-official", action="store_true")
    parser.add_argument("--skip-official-cudagraph", action="store_true")
    parser.add_argument("--skip-spectralis", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.gpt_model):
        print(f"ERROR: model not found: {args.gpt_model}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"  T2S Throughput Comparison")
    print(f"  Model: {args.gpt_model}")
    print(f"  Repeats: {REPEATS}")
    print(f"{'='*70}")

    results = {}

    if not args.skip_official:
        print(f"\n--- Official (torch.cat KV, no CUDA Graph) ---")
        results["Official (no Graph)"] = _run_official(args.gpt_model)

    if not args.skip_official_cudagraph:
        print(f"\n--- Official CUDA Graph (static KV, lazy capture) ---")
        results["Official (CUDA Graph)"] = _run_official_cudagraph(args.gpt_model)

    if not args.skip_spectralis:
        print(f"\n--- Spectralis (static KV, pre-captured CUDA Graph) ---")
        results["Spectralis"] = _run_spectralis(args.gpt_model)

    print(f"\n{'='*70}")
    print(f"  Summary")
    print(f"{'='*70}")
    print(f"  {'Variant':<32} {'T2S Speed':>12}")
    print(f"  {'-'*44}")
    for name, r in results.items():
        speed = r.get("median", 0)
        print(f"  {name:<32} {speed:>8.0f} it/s")
    print()


if __name__ == "__main__":
    main()
