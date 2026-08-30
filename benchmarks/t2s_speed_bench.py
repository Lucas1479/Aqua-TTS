# -*- coding: utf-8 -*-
"""Deterministic GPT-SoVITS T2S AR throughput benchmark.

The benchmark uses fixed tensor shapes and an early-stop boundary instead of
waiting for a randomly sampled EOS token. It reports both caller-visible wall
throughput and the authoritative CUDA-synchronized throughput.

Examples:
    python benchmarks/t2s_speed_bench.py \
        --upstream-home F:/src/GPT-SoVITS \
        --gpt-model F:/models/xxx-e15.ckpt

    python benchmarks/t2s_speed_bench.py ... --flash-attn valid
    python benchmarks/t2s_speed_bench.py ... --engine official
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    text_tokens: int
    prompt_tokens: int
    generated_steps: int
    expected_bucket: int

    @property
    def initial_len(self) -> int:
        return self.text_tokens + self.prompt_tokens


# These shapes cover the buckets that matter for conversational use. The
# conversation case reproduces the Amadeus trace with current_len=377,
# bucket=512 and roughly 120 generated steps.
DEFAULT_CASES = (
    BenchmarkCase("short", text_tokens=117, prompt_tokens=191, generated_steps=51, expected_bucket=448),
    BenchmarkCase(
        "conversation",
        text_tokens=186,
        prompt_tokens=191,
        generated_steps=123,
        expected_bucket=512,
    ),
    BenchmarkCase("long", text_tokens=257, prompt_tokens=191, generated_steps=180, expected_bucket=768),
)


class _NoopTqdm:
    """Minimal tqdm replacement so console rendering is outside the timing."""

    def __init__(self, iterable=None, *args, **kwargs):
        del args, kwargs
        self._iterator = iter(iterable) if iterable is not None else iter(())

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iterator)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        del args

    def update(self, *args, **kwargs):
        del args, kwargs

    def close(self):
        return None

    def set_description(self, *args, **kwargs):
        del args, kwargs

    @staticmethod
    def write(*args, **kwargs):
        del args, kwargs


def _disable_tqdm() -> None:
    import tqdm as tqdm_module

    tqdm_module.tqdm = _NoopTqdm
    try:
        import tqdm.auto as tqdm_auto

        tqdm_auto.tqdm = _NoopTqdm
    except ImportError:
        pass


def _resolve_file(value: str, upstream_home: Path) -> Path:
    requested = Path(value).expanduser()
    candidates = [requested] if requested.is_absolute() else [Path.cwd() / requested, upstream_home / requested]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    attempted = ", ".join(str(candidate.resolve()) for candidate in candidates)
    raise FileNotFoundError(f"GPT checkpoint not found; tried: {attempted}")


def _parse_buckets(value: str) -> list[int]:
    buckets = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not buckets or any(bucket <= 0 for bucket in buckets):
        raise argparse.ArgumentTypeError("buckets must be a comma-separated list of positive integers")
    return buckets


def _summarize(rows: Iterable[dict[str, float]]) -> dict[str, Any]:
    materialized = list(rows)
    if not materialized:
        raise ValueError("cannot summarize an empty benchmark result")
    sync_rates = [row["sync_it_s"] for row in materialized]
    wall_rates = [row["wall_it_s"] for row in materialized]
    return {
        "repeats": len(materialized),
        "median_sync_it_s": statistics.median(sync_rates),
        "median_wall_it_s": statistics.median(wall_rates),
        "min_sync_it_s": min(sync_rates),
        "max_sync_it_s": max(sync_rates),
        "trials": materialized,
    }


def _make_inputs(torch, config: dict[str, Any], case: BenchmarkCase, device):
    phoneme_vocab = int(config["model"]["phoneme_vocab_size"])
    semantic_vocab = int(config["model"]["vocab_size"])
    x = torch.zeros((1, case.text_tokens), dtype=torch.long, device=device)
    x.remainder_(phoneme_vocab)
    x_lens = torch.tensor([case.text_tokens], dtype=torch.long, device=device)
    prompts = torch.zeros((1, case.prompt_tokens), dtype=torch.long, device=device)
    prompts.remainder_(semantic_vocab)
    bert = torch.zeros((1, 1024, case.text_tokens), dtype=torch.float16, device=device)
    return x, x_lens, prompts, bert


def _run_once(torch, decoder, inputs, case: BenchmarkCase, *, graph_enabled: bool, seed: int):
    x, x_lens, prompts, bert = inputs
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        _semantic, reported_steps = decoder.infer_panel(
            x,
            x_lens,
            prompts,
            bert,
            top_k=5,
            top_p=1,
            early_stop_num=case.generated_steps,
            temperature=0.6,
            enable_cuda_graph=graph_enabled,
            enable_static_kv=True,
        )
    wall_elapsed = time.perf_counter() - started
    torch.cuda.synchronize()
    sync_elapsed = time.perf_counter() - started
    steps = int(reported_steps)
    if steps != case.generated_steps:
        raise RuntimeError(
            f"{case.name}: expected {case.generated_steps} reported steps, got {steps}; "
            "the benchmark did not reach its deterministic early-stop boundary"
        )
    return {
        "reported_steps": steps,
        "wall_seconds": wall_elapsed,
        "sync_seconds": sync_elapsed,
        "wall_it_s": steps / wall_elapsed,
        "sync_it_s": steps / sync_elapsed,
    }


def _load_decoder(args, upstream_home: Path, checkpoint: Path, buckets: list[int]):
    graph_enabled = args.engine == "aqua"
    os.environ["ENABLE_CUDA_GRAPH"] = "1" if graph_enabled else "0"
    os.environ["ENABLE_CUDA_GRAPH_PRECAPTURE"] = "0"
    os.environ.pop("CUDA_GRAPH_REPLAY_SYNC", None)

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from aquatts import configure_gpt_sovits

    configure_gpt_sovits(upstream_home, require=True)
    _disable_tqdm()

    import torch
    from AR.models.t2s_lightning_module import Text2SemanticLightningModule

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    model = Text2SemanticLightningModule(config, "****", is_train=False)
    model.load_state_dict(payload["weight"])
    model = model.half().to(args.device).eval()
    decoder = model.model

    if args.engine.startswith("aqua"):
        from aquatts.modeling import apply_cuda_graph_patch

        apply_cuda_graph_patch(decoder, buckets=buckets)
        if args.flash_attn != "off":
            from aquatts.modeling.t2s_flash_attn import (
                apply_flash_attn_patch,
                get_flash_attn_error,
            )

            if not apply_flash_attn_patch(decoder, mode=args.flash_attn):
                raise RuntimeError(f"FlashAttention2 is unavailable: {get_flash_attn_error()}")
        if graph_enabled:
            results = decoder.precapture_cuda_graph(buckets=buckets)
            failed = [bucket for bucket, ok in results.items() if not ok]
            if failed:
                raise RuntimeError(f"CUDA Graph pre-capture failed for buckets: {failed}")
    elif args.flash_attn != "off":
        raise ValueError("--flash-attn requires --engine aqua or aqua-static")

    return torch, model, decoder, config, graph_enabled


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic T2S AR throughput benchmark")
    parser.add_argument("--upstream-home", default=os.environ.get("GPT_SOVITS_HOME"))
    parser.add_argument("--gpt-model", required=True)
    parser.add_argument("--engine", choices=("aqua", "aqua-static", "official"), default="aqua")
    parser.add_argument("--flash-attn", choices=("off", "valid", "bucket"), default="off")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--buckets", type=_parse_buckets, default=_parse_buckets("448,512,768"))
    parser.add_argument(
        "--warmup",
        type=int,
        default=15,
        help="Conversation-shape warmup calls before steady-state trials; 15 also raises the GPU to a stable P-state",
    )
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--json-output")
    parser.add_argument("--bench-official", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.bench_official:
        args.engine = "official"
    if not args.upstream_home:
        raise SystemExit("--upstream-home or GPT_SOVITS_HOME is required")
    if args.warmup < 0 or args.repeats < 1:
        raise SystemExit("--warmup must be >= 0 and --repeats must be >= 1")

    upstream_home = Path(args.upstream_home).expanduser().resolve()
    checkpoint = _resolve_file(args.gpt_model, upstream_home)
    required_buckets = {case.expected_bucket for case in DEFAULT_CASES}
    missing = required_buckets.difference(args.buckets)
    if args.engine == "aqua" and missing:
        raise SystemExit(f"--buckets is missing benchmark buckets: {sorted(missing)}")

    print("=" * 88)
    print("Aqua deterministic T2S AR benchmark")
    print(f"engine={args.engine} flash_attn={args.flash_attn} device={args.device}")
    print(f"upstream={upstream_home}")
    print(f"checkpoint={checkpoint}")
    print("=" * 88)

    torch, model, decoder, config, graph_enabled = _load_decoder(
        args, upstream_home, checkpoint, args.buckets
    )
    del model  # decoder retains all references required for inference
    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        raise RuntimeError("this benchmark requires a CUDA device")

    device = next(decoder.parameters()).device
    inputs = {case.name: _make_inputs(torch, config, case, device) for case in DEFAULT_CASES}
    original_eos = decoder.EOS
    decoder.EOS = -1
    try:
        conversation = next(case for case in DEFAULT_CASES if case.name == "conversation")
        cold = _run_once(
            torch,
            decoder,
            inputs[conversation.name],
            conversation,
            graph_enabled=graph_enabled,
            seed=args.seed,
        )
        print(
            f"cold/conversation  initial={conversation.initial_len:>3} "
            f"bucket={conversation.expected_bucket:>3} steps={cold['reported_steps']:>3} "
            f"sync={cold['sync_it_s']:>7.1f} it/s wall={cold['wall_it_s']:>7.1f} it/s"
        )

        for index in range(args.warmup):
            _run_once(
                torch,
                decoder,
                inputs[conversation.name],
                conversation,
                graph_enabled=graph_enabled,
                seed=args.seed + index,
            )

        summaries: dict[str, Any] = {}
        for case in DEFAULT_CASES:
            rows = []
            for repeat in range(args.repeats):
                row = _run_once(
                    torch,
                    decoder,
                    inputs[case.name],
                    case,
                    graph_enabled=graph_enabled,
                    seed=args.seed + repeat,
                )
                rows.append(row)
            summary = _summarize(rows)
            summaries[case.name] = {
                **asdict(case),
                "initial_len": case.initial_len,
                **summary,
            }
            print(
                f"{case.name:<18} initial={case.initial_len:>3} bucket={case.expected_bucket:>3} "
                f"steps={case.generated_steps:>3} sync={summary['median_sync_it_s']:>7.1f} it/s "
                f"wall={summary['median_wall_it_s']:>7.1f} it/s "
                f"range={summary['min_sync_it_s']:.1f}-{summary['max_sync_it_s']:.1f}"
            )
    finally:
        decoder.EOS = original_eos

    output = {
        "schema_version": 1,
        "engine": args.engine,
        "flash_attn": args.flash_attn,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "upstream_home": str(upstream_home),
        "checkpoint": str(checkpoint),
        "warmup_runs": args.warmup,
        "measured_repeats": args.repeats,
        "cold": {**asdict(conversation), "initial_len": conversation.initial_len, **cold},
        "cases": summaries,
    }
    if args.json_output:
        destination = Path(args.json_output).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"json={destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
