"""Small durable benchmark records; all writes occur outside measured windows."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess


def file_identity(value):
    path = Path(value)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def source_identity(root):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    return {"commit": commit, "dirty": bool(dirty), "python": platform.python_version(),
            "platform": platform.platform()}


class BenchmarkJournal:
    def __init__(self, output):
        self.path = Path(output).with_suffix(".trials.jsonl") if output else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                raise FileExistsError(f"Refusing to overwrite prior trial evidence: {self.path}")

    def record(self, event, **fields):
        if self.path is None:
            return
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps({"time_utc": datetime.now(timezone.utc).isoformat(),
                                     "event": event, **fields}, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
