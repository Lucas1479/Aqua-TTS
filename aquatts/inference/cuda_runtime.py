"""CUDA selection and bounded BigVGAN preparation; Torch stays optional at import."""
import os
import time


def resolve_cuda_device(requested, *, torch_module=None):
    if torch_module is None:
        import torch as torch_module
    device = torch_module.device(requested)
    if device.type != "cuda":
        return device
    if not torch_module.cuda.is_available() or getattr(torch_module.version, "hip", None):
        raise RuntimeError("Requested CUDA device requires an available NVIDIA CUDA runtime")
    index = torch_module.cuda.current_device() if device.index is None else device.index
    count = torch_module.cuda.device_count()
    if index < 0 or index >= count:
        raise ValueError(f"CUDA device index {index} is unavailable; visible device count is {count}")
    return torch_module.device("cuda", index)


def resolve_vocoder_warmup(shapes=None, repeats=None, *, environ=None):
    env = os.environ if environ is None else environ
    if shapes is None:
        shapes = env.get("AQUATTS_BIGVGAN_WARMUP_SHAPES", "40,70,128")
    if repeats is None:
        repeats = env.get("AQUATTS_BIGVGAN_WARMUP_REPEATS", "5")
    def integer(value, name):
        if isinstance(value, str):
            value = value.strip()
            if value.isdecimal():
                value = int(value)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must contain positive integers")
        return value
    if isinstance(shapes, str):
        shapes = [part.strip() for part in shapes.split(",") if part.strip()]
    try:
        shapes = list(dict.fromkeys(integer(value, "warmup shapes") for value in shapes))
    except TypeError as error:
        raise ValueError("warmup shapes must be a sequence of positive integers") from error
    repeats = integer(repeats, "warmup repeats")
    if len(shapes) > 16 or any(size > 4096 for size in shapes) or repeats > 20:
        raise ValueError("warmup budget: at most 16 shapes, 4096 frames per shape and 20 repeats")
    if sum(shapes) * repeats > 65536:
        raise ValueError("warmup budget exceeds 65536 total mel frames")
    return tuple(shapes), repeats


def warmup_vocoder(model, shapes, repeats):
    import torch
    stats = {"status": "disabled", "shapes": list(shapes), "repeats": repeats,
             "forward_calls": 0, "seconds": 0.0}
    if model is None or not shapes:
        return stats
    parameter = next(model.parameters())
    device, dtype = parameter.device, parameter.dtype
    stats.update(device=str(device), dtype=str(dtype), mel_channels=int(model.h.num_mels))
    if device.type != "cuda":
        stats.update(status="skipped", reason="CUDA warmup only")
        return stats
    started = time.perf_counter()
    try:
        with torch.cuda.device(device), torch.random.fork_rng(devices=[device.index]), torch.inference_mode():
            for frames in shapes:
                mel = torch.zeros(1, model.h.num_mels, frames, device=device, dtype=dtype)
                for _ in range(repeats):
                    model(mel)
                    stats["forward_calls"] += 1
            torch.cuda.synchronize(device)
        stats["status"] = "completed"
    except Exception as error:
        stats.update(status="failed", error=str(error))
    stats["seconds"] = time.perf_counter() - started
    return stats
