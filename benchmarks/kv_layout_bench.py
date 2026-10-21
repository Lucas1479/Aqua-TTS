"""Measure prepared Graph KV storage in a fresh process, without voice assets.

Run separately for bucket/nested. For a legacy control, export the pre-change
t2s_streaming.py and pass --layout legacy --baseline-source that_file.py.
This measures construction and storage, not inference latency or speech quality.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpt-model", type=Path, required=True)
    parser.add_argument("--layout", choices=("legacy", "bucket", "nested"), required=True)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--preset", choices=("full", "minimal", "lazy", "off"), default="full")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precapture-buckets", type=int, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.layout == "legacy") != (args.baseline_source is not None):
        parser.error("legacy requires --baseline-source; bucket/nested use the installed source")

    import torch
    from aquatts import TTSInferencer  # noqa: F401 - initialize upstream import context
    from aquatts.inferencer import Text2SemanticLightningModule
    from aquatts.inference.presets import resolve_cuda_graph_config
    import aquatts.modeling.t2s_streaming as runtime

    if args.baseline_source:
        spec = importlib.util.spec_from_file_location("kv_layout_baseline", args.baseline_source)
        runtime = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = runtime
        spec.loader.exec_module(runtime)
    config = resolve_cuda_graph_config(args.preset, {})
    if args.precapture_buckets is not None:
        config["buckets"] = args.precapture_buckets
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available() or torch.version.hip:
        parser.error("this benchmark requires NVIDIA CUDA")
    with torch.cuda.device(device):
        checkpoint = torch.load(args.gpt_model, map_location="cpu")
        model = Text2SemanticLightningModule(checkpoint["config"], "benchmark", is_train=False)
        model.load_state_dict(checkpoint["weight"])
        del checkpoint
        model = model.half().to(device).eval()
        kwargs = {} if args.layout == "legacy" else {"kv_cache_layout": args.layout}
        decoder = runtime.apply_cuda_graph_patch(model.model, graph_config=config, **kwargs)
        torch.cuda.synchronize(device)
        before = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        prepared = {}
        if config["enable"] and config["precapture"]:
            prepared = decoder.precapture_cuda_graph(config["buckets"])
            if not all(prepared.values()):
                raise RuntimeError(f"Incomplete capture: {prepared}")
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        storages = {}
        for inputs in decoder.bucket_static_inputs.values():
            for tensor in inputs["k_cache"] + inputs["v_cache"]:
                storage = tensor.untyped_storage()
                storages[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
        report = {
            "layout": args.layout, "preset": args.preset,
            "python": sys.version.split()[0], "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(device), "dtype": str(next(model.parameters()).dtype),
            "layers": decoder.num_layers, "hidden": decoder.model_dim,
            "buckets": decoder.kv_cache_buckets, "prepared": prepared,
            "requested_capture_buckets": config["buckets"],
            "graph_count": len(decoder.bucket_graphs),
            "graph_keys": sorted(decoder.bucket_graphs),
            "unique_kv_storage_bytes": sum(storages.values()),
            "unique_kv_storage_count": len(storages),
            "allocated_before_capture": before,
            "allocated_after_capture": torch.cuda.memory_allocated(device),
            "reserved_after_capture": torch.cuda.memory_reserved(device),
            "peak_allocated_capture": torch.cuda.max_memory_allocated(device),
            "preparation_seconds": elapsed,
            "checkpoint_sha256": hashlib.sha256(args.gpt_model.read_bytes()).hexdigest(),
            "source_sha256": hashlib.sha256(Path(runtime.__file__).read_bytes()).hexdigest(),
            "scope": "One fresh process; SDPA synthetic capture; no inference latency or voice quality claim",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
