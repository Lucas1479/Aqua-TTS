"""Plan or explicitly run multi-GPU validation with durable per-test evidence.

Default invocation only writes a plan: no Torch import and no GPU discovery.
Use --execute after independently confirming that all visible GPUs are idle.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MULTI_GPU_TEST = "tests/test_multi_gpu_runtime.py"
PHASES = {
    "multi_gpu": [MULTI_GPU_TEST, "-q", "-x", "-ra"],
    "runtime": ["tests", f"--ignore={MULTI_GPU_TEST}", "-q", "-x", "-ra"],
    "runtime_reversed": ["tests", f"--ignore={MULTI_GPU_TEST}", "-q", "-x", "-ra"],
}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


class Journal:
    def __init__(self, directory):
        self.directory = directory
        self.path = directory / "events.jsonl"
        self.collected = 0
        self.calls = Counter()
        self.failed_reports = 0
        self.skipped_reports = 0

    def append(self, event, **details):
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps({"time": stamp(), "event": event, **details}, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())

    def pytest_collection_finish(self, session):
        self.collected = len(session.items)
        self.append("collected", nodeids=[item.nodeid for item in session.items])

    def pytest_runtest_logstart(self, nodeid, location):
        self.append("started", nodeid=nodeid)

    def pytest_runtest_logreport(self, report):
        self.failed_reports += int(report.failed)
        self.skipped_reports += int(report.skipped)
        if report.when == "call":
            self.calls[report.outcome] += 1
        self.append("report", nodeid=report.nodeid, when=report.when,
                    outcome=report.outcome, seconds=report.duration,
                    detail=str(report.longrepr) if report.failed or report.skipped else None)

    def pytest_sessionfinish(self, session, exitstatus):
        self.append("session_finished", exitstatus=int(exitstatus))


def worker(phase, directory):
    """Only called by the explicit execution path in a fresh child process."""
    journal = Journal(directory)
    journal.append("worker_started", phase=phase, pid=os.getpid())
    try:
        sys.path.insert(0, str(ROOT))
        from aquatts import TTSInferencer  # noqa: F401
        import pytest
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise RuntimeError("execution requires at least two visible NVIDIA CUDA GPUs")
        devices = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append({"index": index, "name": props.name,
                            "uuid": str(getattr(props, "uuid", "unavailable")),
                            "total_memory": props.total_memory})
        journal.append("devices", torch=torch.__version__, devices=devices,
                       visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
        code = int(pytest.main(PHASES[phase], plugins=[journal]))
        passed = code == 0 and journal.collected > 0 and journal.failed_reports == 0
        if phase == "multi_gpu":
            passed = passed and journal.skipped_reports == 0 and journal.calls["passed"] == journal.collected
        result = {"phase": phase, "status": "passed" if passed else "failed", "exitcode": code,
                  "collected": journal.collected, "calls": dict(journal.calls),
                  "failed_reports": journal.failed_reports, "skipped_reports": journal.skipped_reports,
                  "devices": devices, "finished_at": stamp()}
        write_json(directory / "result.json", result)
        return 0 if passed else (code or 1)
    except Exception as error:
        journal.append("worker_error", error=repr(error))
        write_json(directory / "result.json", {"phase": phase, "status": "failed", "error": repr(error)})
        raise


def run(output, timeout):
    selected = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    selected = [value.strip() for value in selected if value.strip()]
    if len(selected) != 2 or len(set(selected)) != 2 or not all(value.startswith("GPU-") for value in selected):
        raise ValueError("execution requires CUDA_VISIBLE_DEVICES to name exactly two distinct physical GPU UUIDs")
    # Prevent two invocations of this runner from competing for the same GPUs.
    lock = ROOT / ".tmp/multi-gpu-validation.lock"
    lock.parent.mkdir(exist_ok=True)
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"pid": os.getpid(), "started_at": stamp(), "output": str(output)}))
            handle.flush()
            os.fsync(handle.fileno())
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
        if dirty.strip():
            raise RuntimeError("commit the tested sources before hardware validation")
        manifest = {"status": "running", "head": head, "started_at": stamp(),
                    "python": sys.executable, "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "phases": {}, "scope": "Correctness only; no throughput or audio-quality acceptance."}
        write_json(output / "manifest.json", manifest)
        environment = dict(os.environ, AQUATTS_RUN_MULTI_GPU="1", PYTHONUNBUFFERED="1",
                           PYTHONIOENCODING="utf-8")
        for phase in PHASES:
            directory = output / phase
            directory.mkdir()
            manifest["phases"][phase] = {"status": "running", "started_at": stamp()}
            write_json(output / "manifest.json", manifest)
            started = time.monotonic()
            print(f"Running {phase}; durable progress: {directory / 'events.jsonl'}", flush=True)
            try:
                phase_environment = dict(environment)
                phase_environment["CUDA_VISIBLE_DEVICES"] = ",".join(
                    reversed(selected) if phase == "runtime_reversed" else selected)
                with (directory / "output.log").open("wb", buffering=0) as log:
                    completed = subprocess.run(
                        [sys.executable, str(Path(__file__).resolve()), "--worker", phase,
                         "--output", str(directory)],
                        cwd=ROOT, env=phase_environment, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
                path = directory / "result.json"
                result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
                    "status": "incomplete", "reason": "worker exited without a result"}
                result["process_exitcode"] = completed.returncode
                passed = completed.returncode == 0 and result["status"] == "passed"
            except subprocess.TimeoutExpired:
                result = {"status": "incomplete", "reason": f"worker exceeded {timeout} seconds"}
                passed = False
            result["wall_seconds"] = time.monotonic() - started
            manifest["phases"][phase] = result
            if not passed:
                manifest.update(status="failed_or_incomplete", finished_at=stamp())
                write_json(output / "manifest.json", manifest)
                print(f"{phase} did not pass; remaining phases were not started.", flush=True)
                return 1
            write_json(output / "manifest.json", manifest)
        manifest.update(status="passed", finished_at=stamp())
        write_json(output / "manifest.json", manifest)
        print(f"All requested correctness phases passed: {output / 'manifest.json'}", flush=True)
        return 0
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="actually use the visible GPUs; requires idle hardware")
    parser.add_argument("--timeout", type=int, default=900, help="maximum seconds per worker phase")
    parser.add_argument("--worker", choices=PHASES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    output = args.output.resolve()
    if args.worker:
        if os.environ.get("AQUATTS_RUN_MULTI_GPU") != "1":
            parser.error("worker execution requires the explicit multi-GPU environment gate")
        return worker(args.worker, output)
    if output.exists() and any(output.iterdir()):
        parser.error("--output must be a new or empty directory; previous evidence is never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    plan = {"status": "not_run", "created_at": stamp(), "gpu_access": False,
            "phases": PHASES, "multi_gpu_cases_for_two_devices": 24,
            "requirements": ["all selected GPUs idle", "two physical GPU UUIDs in CUDA_VISIBLE_DEVICES",
                             "runtime dependencies and upstream configured"],
            "outputs": ["manifest.json", "<phase>/events.jsonl", "<phase>/output.log", "<phase>/result.json"]}
    write_json(output / "plan.json", plan)
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0
    return run(output, args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
