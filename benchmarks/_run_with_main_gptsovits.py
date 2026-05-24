"""Wrapper: run a benchmark script using the full GPT-SoVITS from the main repo.

Usage:
    python benchmarks/_run_with_main_gptsovits.py benchmarks/t2s_speed_bench.py [args...]
"""
import os
import sys

MAIN_REPO = os.environ.get("GPT_SOVITS_HOME")
if not MAIN_REPO:
    sys.exit("GPT_SOVITS_HOME must be set to your GPT-SoVITS repo root")
MAIN_GPT_SOVITS = os.path.join(MAIN_REPO, "GPT_SoVITS")
SPECTRALIS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDORED_GPT_SOVITS = os.path.join(SPECTRALIS_ROOT, "spectralis", "_vendor")

# Path order:
#   [0] SPECTRALIS_ROOT  — spectralis package
#   [1] MAIN_GPT_SOVITS  — bare AR / BigVGAN absolute imports resolve here
#   [2] MAIN_REPO        — GPT_SoVITS, config, tools packages
# Vendored overrides in spectralis/_vendor/ are managed by spectralis/__init__.py.
sys.path.insert(0, MAIN_GPT_SOVITS)
sys.path.insert(0, MAIN_REPO)
sys.path.insert(0, SPECTRALIS_ROOT)


class _GuardedPath(list):
    """A sys.path that keeps vendored overrides ahead of main-repo originals.

    Path order:
      [0] SPECTRALIS_ROOT      — spectralis package, benchmarks
      [1] VENDORED_GPT_SOVITS  — spectralis/_vendor (t2s_model.py + BigVGAN CUDA)
      [2] MAIN_GPT_SOVITS      — bare "AR.modules.*" fallback (not in vendored)
      [3] MAIN_REPO            — GPT_SoVITS.xxx, config, tools packages

    VENDORED_GPT_SOVITS contains vendored overrides: t2s_model.py (static KV +
    CUDA Graph) and BigVGAN CUDA kernel loader. Namespace __init__.py files
    (pkgutil.extend_path) merge with main GPT-SoVITS at import time.
    """

    _CANONICAL = (SPECTRALIS_ROOT, VENDORED_GPT_SOVITS, MAIN_GPT_SOVITS, MAIN_REPO)

    def _fixup(self) -> None:
        # Remove duplicates of entries we manage
        seen: set = set()
        i = 0
        while i < len(self):
            if self[i] in self._CANONICAL:
                if self[i] in seen:
                    self.pop(i)
                    continue
                seen.add(self[i])
            i += 1
        # Re-establish canonical order
        for p in self._CANONICAL:
            try:
                self.remove(p)
            except ValueError:
                pass
        super().insert(0, SPECTRALIS_ROOT)
        super().insert(1, VENDORED_GPT_SOVITS)
        super().insert(2, MAIN_GPT_SOVITS)
        super().insert(3, MAIN_REPO)

    def insert(self, index: int, value: str) -> None:
        super().insert(index, value)
        self._fixup()

    def append(self, value: str) -> None:
        super().append(value)
        self._fixup()

    def extend(self, values) -> None:
        super().extend(values)
        self._fixup()


sys.path = _GuardedPath(sys.path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python benchmarks/_run_with_main_gptsovits.py <benchmark_script> [args...]")
        sys.exit(1)

    script = sys.argv[1]
    sys.argv = sys.argv[1:]

    with open(script, "r", encoding="utf-8") as f:
        code = compile(f.read(), os.path.abspath(script), "exec")

    exec(code, {"__name__": "__main__", "__file__": os.path.abspath(script)})
