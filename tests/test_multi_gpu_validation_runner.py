"""The default plan/collection paths must not import Torch or probe a GPU."""
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
BLOCK_TORCH = """
import importlib.abc
import sys
class RejectTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('default validation path imported Torch')
sys.meta_path.insert(0, RejectTorch())
"""


def test_plan_never_imports_torch_and_records_pending_hardware(tmp_path):
    output = tmp_path / "plan"
    code = BLOCK_TORCH + """
import runpy
from pathlib import Path
script = Path.cwd() / 'benchmarks/multi_gpu_validation.py'
sys.argv = [str(script), '--output', sys.argv[1]]
runpy.run_path(str(script), run_name='__main__')
"""
    # The script receives only the result path; the child has a hard import guard.
    result = subprocess.run([sys.executable, "-c", code, str(output)],
                            cwd=ROOT, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    plan = json.loads((output / "plan.json").read_text())
    assert plan["status"] == "not_run" and plan["gpu_access"] is False
    assert plan["multi_gpu_cases_for_two_devices"] == 24
    assert not (output / "manifest.json").exists()


def test_default_multi_gpu_collection_skips_before_importing_torch():
    import os
    environment = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    environment.pop("AQUATTS_RUN_MULTI_GPU", None)
    code = BLOCK_TORCH + """
import pytest
code = pytest.main(['tests/test_multi_gpu_runtime.py', '-q'])
assert code == 5, code  # An entirely gated module collects no runnable tests.
assert 'torch' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=environment,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 skipped" in result.stdout


@pytest.mark.parametrize("failure", ["timeout", "missing_result", "worker_failed"])
def test_failed_or_incomplete_worker_cannot_pass_or_start_runtime(monkeypatch, tmp_path, failure):
    from benchmarks import multi_gpu_validation as runner

    root = tmp_path / "repo"
    root.mkdir()
    output = tmp_path / "results"
    output.mkdir()
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-test-A,GPU-test-B")
    monkeypatch.setattr(runner.subprocess, "check_output",
                        lambda command, **_: "tested-head\n" if command[1] == "rev-parse" else "")
    attempts = []

    def child(command, **options):
        attempts.append(command)
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, options["timeout"])
        if failure == "worker_failed":
            runner.write_json(output / "multi_gpu/result.json", {"status": "failed"})
            return subprocess.CompletedProcess(command, 1)
        return subprocess.CompletedProcess(command, 0)  # Exit zero alone is insufficient.

    monkeypatch.setattr(runner.subprocess, "run", child)
    assert runner.run(output, timeout=10) == 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed_or_incomplete"
    assert manifest["phases"]["multi_gpu"]["status"] in {"failed", "incomplete"}
    assert len(attempts) == 1 and not (output / "runtime").exists()
    assert not (root / ".tmp/multi-gpu-validation.lock").exists()


def test_runner_keeps_both_gpus_visible_and_reverses_runtime_order(monkeypatch, tmp_path):
    from benchmarks import multi_gpu_validation as runner

    root, output = tmp_path / "repo", tmp_path / "results"
    root.mkdir()
    output.mkdir()
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-test-A,GPU-test-B")
    monkeypatch.setattr(runner.subprocess, "check_output",
                        lambda command, **_: "tested-head\n" if command[1] == "rev-parse" else "")
    selections = []

    def child(command, **options):
        selections.append(options["env"]["CUDA_VISIBLE_DEVICES"])
        assert options["env"]["AQUATTS_RUN_MULTI_GPU"] == "1"
        phase = command[command.index("--worker") + 1]
        runner.write_json(output / phase / "result.json", {"status": "passed"})
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner.subprocess, "run", child)
    assert runner.run(output, timeout=10) == 0
    assert selections == ["GPU-test-A,GPU-test-B", "GPU-test-A,GPU-test-B", "GPU-test-B,GPU-test-A"]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "passed" and set(manifest["phases"]) == set(runner.PHASES)
