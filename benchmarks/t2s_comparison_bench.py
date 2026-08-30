# -*- coding: utf-8 -*-
"""Run deterministic T2S engines in isolated subprocesses.

The subprocess boundary prevents GPT-SoVITS module-path contamination while
keeping every engine on the same checkpoint, tensor shapes and step counts.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPEED_BENCH = ROOT / "benchmarks" / "t2s_speed_bench.py"


def _run_variant(args, name: str, engine: str, flash_attn: str) -> dict[str, Any]:
    handle = tempfile.NamedTemporaryFile(prefix="aqua-t2s-", suffix=".json", delete=False)
    json_path = Path(handle.name)
    handle.close()
    command = [
        sys.executable,
        str(SPEED_BENCH),
        "--upstream-home",
        str(Path(args.upstream_home).expanduser().resolve()),
        "--gpt-model",
        args.gpt_model,
        "--engine",
        engine,
        "--flash-attn",
        flash_attn,
        "--device",
        args.device,
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--json-output",
        str(json_path),
    ]
    print(f"\n--- {name} ---", flush=True)
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=args.timeout,
        )
        if completed.stdout:
            print(completed.stdout.rstrip())
        if completed.stderr:
            print(completed.stderr.rstrip(), file=sys.stderr)
        if completed.returncode:
            raise RuntimeError(f"{name} benchmark failed with exit code {completed.returncode}")
        return json.loads(json_path.read_text(encoding="utf-8"))
    finally:
        json_path.unlink(missing_ok=True)


def _format_rate(result: dict[str, Any], case: str) -> str:
    return f"{result['cases'][case]['median_sync_it_s']:.1f}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Isolated T2S throughput comparison")
    parser.add_argument("--upstream-home", default=os.environ.get("GPT_SOVITS_HOME"))
    parser.add_argument("--gpt-model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--flash-ab", action="store_true", help="also benchmark Aqua + FlashAttention2 valid mode")
    parser.add_argument("--skip-official", action="store_true")
    parser.add_argument("--skip-aqua-static", action="store_true")
    parser.add_argument("--skip-aqua", action="store_true")
    parser.add_argument("--json-output")
    parser.add_argument("--skip-official-cudagraph", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-Aqua", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.upstream_home:
        raise SystemExit("--upstream-home or GPT_SOVITS_HOME is required")
    if args.skip_Aqua:
        args.skip_aqua = True

    variants = []
    if not args.skip_official:
        variants.append(("Official upstream", "official", "off"))
    if not args.skip_aqua_static:
        variants.append(("Aqua static KV", "aqua-static", "off"))
    if not args.skip_aqua:
        variants.append(("Aqua CUDA Graph", "aqua", "off"))
        if args.flash_ab:
            variants.append(("Aqua CUDA Graph + FA2 valid", "aqua", "valid"))
    if not variants:
        raise SystemExit("all benchmark variants were skipped")

    results = {
        name: _run_variant(args, name, engine, flash_attn)
        for name, engine, flash_attn in variants
    }

    print("\n" + "=" * 92)
    print("Synchronized T2S throughput summary (median it/s)")
    print("=" * 92)
    print(f"{'Variant':<36} {'Cold':>10} {'Short/448':>12} {'Conv/512':>12} {'Long/768':>12}")
    print("-" * 92)
    for name, result in results.items():
        print(
            f"{name:<36} "
            f"{result['cold']['sync_it_s']:>10.1f} "
            f"{_format_rate(result, 'short'):>12} "
            f"{_format_rate(result, 'conversation'):>12} "
            f"{_format_rate(result, 'long'):>12}"
        )

    output = {
        "schema_version": 1,
        "upstream_home": str(Path(args.upstream_home).expanduser().resolve()),
        "gpt_model": str(Path(args.gpt_model).expanduser()),
        "warmup_runs": args.warmup,
        "measured_repeats": args.repeats,
        "results": results,
    }
    if args.json_output:
        destination = Path(args.json_output).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"json={destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
