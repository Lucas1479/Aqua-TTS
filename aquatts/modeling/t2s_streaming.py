"""
Static KV Cache and CUDA Graph acceleration for GPT-SoVITS T2S AR decoder.

Key optimizations:
1. T2SBlockWithStaticCache — scatter_ instead of torch.cat for KV cache,
   maintaining fixed-size buffers so CUDA Graph can capture the decode step.
2. Bucketed CUDA Graph — 6 pre-captured graphs (128, 256, 448, 512, 768, 1024)
   with complete-request workspace ownership and optional shared nested roots.
3. apply_cuda_graph_patch() — monkey-patches Text2SemanticDecoder to use
   static cache + optional CUDA Graph during infer_panel.

Graceful degradation: CUDA Graph -> static KV -> dynamic KV fallback chain.
"""

import time
import inspect as _inspect
import os
import logging
from typing import List, Optional, Tuple

import torch
from torch.nn import functional as F
from aquatts.inference.presets import (
    resolve_cuda_graph_config,
    resolve_cuda_graph_request,
)
from aquatts.inference.workspace import GraphWorkspaceRegistry, resolve_kv_cache_layout
from aquatts.inference.cuda_execution import cuda_execution, cuda_work

# ---------------------------------------------------------------------------
# Upstream GPT-SoVITS imports. Aqua patches these objects in-place and does not
# replace the upstream t2s_model module.
# ---------------------------------------------------------------------------
from AR.models.t2s_model import (
    T2SMLP,
    Text2SemanticDecoder,
    scaled_dot_product_attention,
)

logger = logging.getLogger(__name__)


_GRAPH_INITIAL_LEN_STRIDE = 32
_SUPPORTED_PATCH_KWARGS = frozenset({"enable_cuda_graph", "enable_static_kv"})


class IncompatibleGPTSoVITSError(RuntimeError):
    """The selected upstream decoder no longer satisfies Aqua's patch contract."""


def _env_flag_enabled(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _should_stop_on_eos(logits: torch.Tensor, samples: torch.Tensor, eos: int) -> bool:
    """Evaluate both EOS conditions with one device-to-host synchronization."""

    greedy_is_eos = torch.argmax(logits, dim=-1)[0].eq(eos)
    sampled_is_eos = samples[0, 0].eq(eos)
    return bool(torch.logical_or(greedy_is_eos, sampled_is_eos).item())


def _replay_cuda_graph(decoder, cuda_graph, device) -> None:
    cuda_graph.replay()
    if decoder.cuda_graph_replay_sync:
        torch.cuda.synchronize(device)


def _validate_decoder_contract(decoder) -> None:
    required = (
        "num_layers",
        "num_head",
        "model_dim",
        "h",
        "ar_predict_layer",
        "infer_panel",
        "infer_panel_naive",
    )
    missing = [name for name in required if not hasattr(decoder, name)]
    if missing:
        raise IncompatibleGPTSoVITSError(
            "upstream Text2SemanticDecoder is missing required attributes: "
            + ", ".join(missing)
        )

__all__ = [
    "T2SBlockWithStaticCache",
    "T2STransformerWithStaticCache",
    "apply_cuda_graph_patch",
    "IncompatibleGPTSoVITSError",
    "_GRAPH_INITIAL_LEN_STRIDE",
]


# ===========================================================================
# Static KV Cache blocks
# ===========================================================================

@torch.jit.script
class T2SBlockWithStaticCache:
    """
    T2SBlock variant that writes into a fixed-size KV buffer via scatter_.

    Unlike the original decode_next_token which uses torch.cat (changing tensor
    shapes every step), this keeps the buffer shape constant so the decode step
    can be captured by CUDA Graph.
    """
    def __init__(
            self,
            num_heads,
            hidden_dim: int,
            mlp: T2SMLP,
            qkv_w,
            qkv_b,
            out_w,
            out_b,
            norm_w1,
            norm_b1,
            norm_eps1,
            norm_w2,
            norm_b2,
            norm_eps2,
    ):
        self.num_heads = num_heads
        self.mlp = mlp
        self.hidden_dim: int = hidden_dim
        self.qkv_w = qkv_w
        self.qkv_b = qkv_b
        self.out_w = out_w
        self.out_b = out_b
        self.norm_w1 = norm_w1
        self.norm_b1 = norm_b1
        self.norm_eps1 = norm_eps1
        self.norm_w2 = norm_w2
        self.norm_b2 = norm_b2
        self.norm_eps2 = norm_eps2

    def process_prompt(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        torch_sdpa: bool = True
    ):
        if padding_mask is not None:
            if padding_mask.dim() == 2:
                padding_mask = padding_mask.unsqueeze(-1)
            x_masked = x * padding_mask
        else:
            x_masked = x

        q, k, v = F.linear(x_masked, self.qkv_w, self.qkv_b).chunk(3, dim=-1)

        batch_size = q.shape[0]
        q_len = q.shape[1]
        kv_len = k.shape[1]

        if padding_mask is not None:
            q = q * padding_mask
            k_cache = k * padding_mask
            v_cache = v * padding_mask
        else:
            k_cache = k
            v_cache = v

        q = q.view(batch_size, q_len, self.num_heads, -1).transpose(1, 2)
        k = k_cache.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)
        v = v_cache.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)

        if torch_sdpa:
            attn = F.scaled_dot_product_attention(q, k, v, ~attn_mask)
        else:
            attn = scaled_dot_product_attention(q, k, v, attn_mask)

        attn = attn.transpose(1, 2).reshape(batch_size, q_len, -1)

        if padding_mask is not None:
            attn = F.linear(attn * padding_mask, self.out_w, self.out_b)
        else:
            attn = F.linear(attn, self.out_w, self.out_b)

        x = x + attn
        x = F.layer_norm(
            x, [self.hidden_dim], self.norm_w1, self.norm_b1, self.norm_eps1
        )
        x = x + self.mlp.forward(x)
        x = F.layer_norm(
            x,
            [self.hidden_dim],
            self.norm_w2,
            self.norm_b2,
            self.norm_eps2,
        )
        return x, k_cache, v_cache

    def decode_next_token_with_static_cache(
        self,
        x: torch.Tensor,
        k_cache: torch.Tensor,   # fixed [B, bucket_size, hidden]
        v_cache: torch.Tensor,   # fixed [B, bucket_size, hidden]
        pos_idx: torch.Tensor,   # [B, 1, hidden] long, persistent GPU tensor
        key_valid_mask: torch.Tensor,  # [B, bucket_size] bool
        valid_kv_len: int = -1,
        torch_sdpa: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        scatter_ writes current-step KV into pos_idx position.
        Attention masks unwritten and graph-alignment gap positions. Zero KV
        does not mean zero softmax weight, so padding must never be visible.
        pos_idx is a persistent GPU tensor updated via fill_() outside the graph,
        so the replay sees the correct write position without shape changes.
        """
        q, k, v = F.linear(x, self.qkv_w, self.qkv_b).chunk(3, dim=-1)

        k_cache.scatter_(1, pos_idx, k)
        v_cache.scatter_(1, pos_idx, v)

        batch_size = q.shape[0]
        kv_len = valid_kv_len if valid_kv_len > 0 else k_cache.shape[1]

        q = q.view(batch_size, 1, self.num_heads, -1).transpose(1, 2)
        k_full = k_cache[:, :kv_len, :].view(
            batch_size, kv_len, self.num_heads, -1
        ).transpose(1, 2)
        v_full = v_cache[:, :kv_len, :].view(
            batch_size, kv_len, self.num_heads, -1
        ).transpose(1, 2)

        if valid_kv_len > 0 and torch_sdpa:
            attn = F.scaled_dot_product_attention(q, k_full, v_full)
        elif valid_kv_len > 0:
            attn = scaled_dot_product_attention(q, k_full, v_full, None)
        elif torch_sdpa:
            valid_attn_mask = key_valid_mask.unsqueeze(1).unsqueeze(1)
            attn = F.scaled_dot_product_attention(
                q, k_full, v_full, attn_mask=valid_attn_mask
            )
        else:
            valid_attn_mask = key_valid_mask.unsqueeze(1).unsqueeze(1)
            attn = scaled_dot_product_attention(
                q, k_full, v_full, ~valid_attn_mask
            )

        attn = attn.transpose(1, 2).reshape(batch_size, 1, -1)
        attn = F.linear(attn, self.out_w, self.out_b)

        x = x + attn
        x = F.layer_norm(
            x, [self.hidden_dim], self.norm_w1, self.norm_b1, self.norm_eps1
        )
        x = x + self.mlp.forward(x)
        x = F.layer_norm(
            x,
            [self.hidden_dim],
            self.norm_w2,
            self.norm_b2,
            self.norm_eps2,
        )
        return x, k_cache, v_cache


@torch.jit.script
class T2STransformerWithStaticCache:
    """Transformer wrapper that delegates to T2SBlockWithStaticCache blocks."""

    def __init__(self, num_blocks: int, blocks: List[T2SBlockWithStaticCache]):
        self.num_blocks: int = num_blocks
        self.blocks = blocks

    def process_prompt(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        torch_sdpa: bool = True
    ):
        k_cache: List[torch.Tensor] = []
        v_cache: List[torch.Tensor] = []
        for i in range(self.num_blocks):
            x, k_cache_, v_cache_ = self.blocks[i].process_prompt(
                x, attn_mask, padding_mask, torch_sdpa
            )
            k_cache.append(k_cache_)
            v_cache.append(v_cache_)
        return x, k_cache, v_cache

    def decode_next_token_with_static_cache(
        self,
        x: torch.Tensor,
        k_cache: List[torch.Tensor],
        v_cache: List[torch.Tensor],
        pos_idx: torch.Tensor,
        key_valid_mask: torch.Tensor,
        valid_kv_len: int = -1,
        torch_sdpa: bool = True
    ):
        """All layers share the same write position and validity mask."""
        key_valid_mask.scatter_(1, pos_idx[:, :, 0], True)
        for i in range(self.num_blocks):
            x, k_cache[i], v_cache[i] = self.blocks[i].decode_next_token_with_static_cache(
                x, k_cache[i], v_cache[i], pos_idx, key_valid_mask,
                valid_kv_len, torch_sdpa
            )
        return x, k_cache, v_cache


# ===========================================================================
# Bucketed CUDA Graph capture
# ===========================================================================

# Minimum generation slots reserved when selecting a bucket.
# Must satisfy aligned_kv + _BUCKET_MIN_GEN_SLOTS < bucket_size.
# 96 slots (~1.92s @ 50Hz) keeps single-sentence routing at bucket 512,
# while multi-sentence (kv >= 420) routes to bucket 768.
_BUCKET_MIN_GEN_SLOTS: int = 96

# Default bucket sizes covering typical prompt lengths.
_DEFAULT_BUCKETS = [128, 256, 448, 512, 768, 1024]

# Warmup steps before capturing each bucket.
_CUDA_GRAPH_WARMUP_STEPS = 3


def _select_bucket(decoder, kv_cache_len):
    """Pick the smallest bucket that fits kv_cache_len with room to generate."""
    buckets = getattr(decoder, "kv_cache_buckets", _DEFAULT_BUCKETS)
    if not buckets:
        return None

    _s = _GRAPH_INITIAL_LEN_STRIDE
    aligned = ((kv_cache_len + _s - 1) // _s) * _s
    min_needed = aligned + _BUCKET_MIN_GEN_SLOTS

    for bucket_size in sorted(buckets):
        if min_needed < bucket_size:
            return bucket_size
    return None


def _get_bucket_lock(decoder, bucket_key):
    return decoder._graph_workspaces.lock_for(_workspace_owner(decoder, bucket_key))


def _workspace_owner(decoder, bucket):
    return None if decoder.kv_cache_layout == "nested" else bucket


def _check_graph_storage(decoder, device, dtype):
    signature = (tuple(decoder.kv_cache_buckets), torch.device(device), dtype,
                 decoder.num_layers, decoder.model_dim)
    if decoder._graph_storage_signature is not None and decoder._graph_storage_signature != signature:
        raise RuntimeError("KV device/dtype/capacity changes require a new decoder")
    return signature


def _graph_kv_buffers(decoder, bucket, device, dtype):
    """Return short computation views without replacing captured storage."""
    if tuple(decoder.kv_cache_buckets) != decoder._kv_bucket_signature:
        raise RuntimeError("KV capacity changes require a new decoder")
    if bucket not in decoder.kv_cache_buckets:
        raise ValueError(f"Unknown KV bucket: {bucket}")
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    signature = _check_graph_storage(decoder, device, dtype)
    owner = _workspace_owner(decoder, bucket)
    capacity = max(decoder.kv_cache_buckets) if decoder.kv_cache_layout == "nested" else bucket
    shape = (decoder.num_layers, 1, capacity, decoder.model_dim)
    roots = decoder._graph_kv_roots.get(owner)
    if roots is None:
        roots = (torch.zeros(shape, device=device, dtype=dtype),
                 torch.zeros(shape, device=device, dtype=dtype))
        decoder._graph_kv_roots[owner] = roots
        decoder._graph_storage_signature = signature
    if any(tuple(root.shape) != shape or root.device != device or root.dtype != dtype for root in roots):
        raise RuntimeError("KV device/dtype/capacity changes require a new decoder")
    return tuple([layer[:, :bucket, :] for layer in root.unbind(0)] for root in roots)


def _warmup_and_capture_bucket(decoder, bucket_size, device):
    owner = _workspace_owner(decoder, bucket_size)
    with decoder._graph_workspaces.lease(owner, device):
        if bucket_size in decoder.bucket_graphs:
            return True
        with cuda_execution.try_capture() as admitted:
            if not admitted:
                logger.debug("Capture deferred: other CUDA work is active (bucket=%s)", bucket_size)
                return False
            return _capture_bucket(decoder, bucket_size, device)


@torch.inference_mode(False)
@torch.no_grad()
def _capture_bucket(decoder, bucket_size, device):
    """Capture one graph per bucket; real prefix length is mutable GPU data."""
    graph_key = bucket_size
    initial_len = bucket_size // 2
    print(f"Capturing CUDA Graph for bucket={bucket_size}, initial_len={initial_len}...")
    warmup_start = time.perf_counter()

    try:
        batch_size = 1
        hidden_dim = decoder.model_dim

        model_dtype = next(decoder.ar_predict_layer.parameters()).dtype

        k_cache, v_cache = _graph_kv_buffers(decoder, bucket_size, device, model_dtype)

        for i in range(decoder.num_layers):
            k_cache[i].zero_()
            v_cache[i].zero_()
            # Capture is an implementation detail and must not consume the
            # caller's sampling RNG.
            k_cache[i][:, :initial_len, :].fill_(0.01)
            v_cache[i][:, :initial_len, :].fill_(0.02)

        key_valid_mask = torch.zeros(
            batch_size, bucket_size, dtype=torch.bool, device=device
        )
        key_valid_mask[:, :initial_len] = True
        xy_pos = torch.full(
            (batch_size, 1, hidden_dim), 0.03,
            dtype=model_dtype, device=device,
        )
        pos_idx = torch.full(
            (batch_size, 1, hidden_dim), initial_len, dtype=torch.long, device=device
        )

        # Torch's implicit capture stream can belong to another GPU. Warm up
        # and capture on a stream owned by this decoder's actual device.
        capture_stream = torch.cuda.Stream(device=device)
        capture_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(capture_stream):
            for _ in range(_CUDA_GRAPH_WARMUP_STEPS):
                xy_dec, k_cache, v_cache = decoder.t2s_transformer_static.decode_next_token_with_static_cache(
                    xy_pos, k_cache, v_cache, pos_idx, key_valid_mask
                )
                logits = decoder.ar_predict_layer(xy_dec[:, -1])
        torch.cuda.current_stream(device).wait_stream(capture_stream)

        warmup_time = time.perf_counter() - warmup_start
        decoder.cuda_graph_stats["warmup_time"][graph_key] = warmup_time
        print(f"  Warmup complete: {warmup_time:.4f}s")

        # Reset buffers to fixed state
        for i in range(decoder.num_layers):
            k_cache[i].zero_()
            v_cache[i].zero_()
            k_cache[i][:, :initial_len, :].fill_(0.01)
            v_cache[i][:, :initial_len, :].fill_(0.02)
        key_valid_mask = torch.zeros(
            batch_size, bucket_size, dtype=torch.bool, device=device
        )
        key_valid_mask[:, :initial_len] = True
        xy_pos = torch.full(
            (batch_size, 1, hidden_dim), 0.03,
            dtype=model_dtype, device=device,
        )
        pos_idx = torch.full(
            (batch_size, 1, hidden_dim), initial_len, dtype=torch.long, device=device
        )

        # Capture
        capture_start = time.perf_counter()
        try:
            _has_capture_error_mode = (
                "capture_error_mode"
                in _inspect.signature(torch.cuda.graph).parameters
            )
        except (ValueError, TypeError):
            _has_capture_error_mode = False  # PyTorch < 2.5
        # Admission excludes other participating work, including its RNG and
        # synchronization calls. Keep CUDA's global safety checks enabled.
        _graph_extra = {"capture_error_mode": "global"} if _has_capture_error_mode else {}
        with torch.cuda.device(device):
            cuda_graph = torch.cuda.CUDAGraph()
            capture_stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.graph(cuda_graph, stream=capture_stream, **_graph_extra):
                xy_dec, k_cache_out, v_cache_out = decoder.t2s_transformer_static.decode_next_token_with_static_cache(
                    xy_pos, k_cache, v_cache, pos_idx, key_valid_mask
                )
                logits = decoder.ar_predict_layer(xy_dec[:, -1])
            torch.cuda.current_stream(device).wait_stream(capture_stream)

        capture_time = time.perf_counter() - capture_start
        decoder.cuda_graph_stats["capture_time"][graph_key] = capture_time

        decoder.bucket_graphs[graph_key] = cuda_graph
        decoder.bucket_static_inputs[graph_key] = {
            "xy_pos": xy_pos,
            "k_cache": k_cache,
            "v_cache": v_cache,
            "pos_idx": pos_idx,
            "key_valid_mask": key_valid_mask,
        }
        decoder.bucket_static_outputs[graph_key] = {
            "xy_dec": xy_dec,
            "logits": logits,
        }
        decoder.cuda_graph_stats["bucket_hits"][graph_key] = 0

        print(f"  Graph captured: {capture_time:.4f}s")
        return True

    except Exception as e:
        print(f"  Capture failed for bucket={bucket_size}, initial_len={initial_len}: {e}")
        import traceback
        traceback.print_exc()
        return False


def _precapture_cuda_graph(decoder, buckets=None, kv_len_range=None):
    """Prepare at most one graph per bucket; prefix ranges are unnecessary."""
    if not getattr(decoder, "cuda_graph_enabled", False):
        print("CUDA Graph not enabled, skipping precapture")
        return {}

    available = getattr(decoder, "kv_cache_buckets", _DEFAULT_BUCKETS)
    if not available:
        return {}

    device = next(decoder.parameters()).device
    target_buckets = available if buckets is None else buckets
    results = {}

    for bucket in dict.fromkeys(target_buckets):
        if bucket not in available:
            results[bucket] = False
            continue

        results[bucket] = _warmup_and_capture_bucket(decoder, bucket, device)

    return results


def _promote_prepared_bucket(decoder, current_bucket, valid_length):
    """Expose a larger prepared root view; valid KV is never copied or reset."""
    if decoder.kv_cache_layout != "nested" or valid_length != current_bucket:
        return None
    for target in sorted(decoder.bucket_graphs):
        if target <= current_bucket or target <= valid_length:
            continue
        inputs = decoder.bucket_static_inputs[target]
        # SDPA can propagate NaN/Inf even from masked values. Only newly
        # exposed, unwritten slots are cleared; the live prefix is retained.
        for key, value in zip(inputs["k_cache"], inputs["v_cache"]):
            key[:, current_bucket:].zero_()
            value[:, current_bucket:].zero_()
        inputs["key_valid_mask"].zero_()
        inputs["key_valid_mask"][:, :valid_length] = True
        inputs["pos_idx"].fill_(valid_length)
        return target
    return None


def _ensure_directories_exist(path):
    """Create parent directories for path if needed."""
    import os
    os.makedirs(path, exist_ok=True)


# ===========================================================================
# Patched infer_panel_naive — static KV + optional CUDA Graph
# ===========================================================================

@torch.no_grad()
def _patched_infer_panel_naive(decoder, x, x_lens, prompts, bert_feature,
                                top_k=-100, top_p=100, early_stop_num=-1,
                                temperature=1.0, repetition_penalty=1.35,
                                _workspace_wait_seconds=0.0, **kwargs):
    """
    Replacement for Text2SemanticDecoder.infer_panel_naive with static KV cache
    and optional CUDA Graph acceleration.
    """
    from tqdm import tqdm
    from AR.models.t2s_model import sample

    x = decoder.ar_text_embedding(x)
    x = x + decoder.bert_proj(bert_feature.transpose(1, 2))
    x = decoder.ar_text_position(x)

    y = prompts
    x_len = x.shape[1]
    x_attn_mask = torch.zeros((x_len, x_len), dtype=torch.bool)
    stop = False

    k_cache = None
    v_cache = None

    if y is not None:
        y_emb = decoder.ar_audio_embedding(y)
        y_len = y_emb.shape[1]
        prefix_len = y.shape[1]
        y_pos = decoder.ar_audio_position(y_emb)
        xy_pos = torch.concat([x, y_pos], dim=1)
        ref_free = False
    else:
        y_emb = None
        y_len = 0
        prefix_len = 0
        y_pos = None
        xy_pos = x
        y = torch.zeros(x.shape[0], 0, dtype=torch.int, device=x.device)
        ref_free = True

    bsz = x.shape[0]
    src_len = x_len + y_len
    x_attn_mask_pad = F.pad(
        x_attn_mask,
        (0, y_len),
        value=True,
    )
    y_attn_mask = F.pad(
        torch.triu(torch.ones(y_len, y_len, dtype=torch.bool), diagonal=1),
        (x_len, 0),
        value=False,
    )
    xy_attn_mask = torch.concat([x_attn_mask_pad, y_attn_mask], dim=0) \
        .unsqueeze(0) \
        .expand(bsz * decoder.num_head, -1, -1) \
        .view(bsz, decoder.num_head, src_len, src_len) \
        .to(device=x.device, dtype=torch.bool)

    enable_static_kv = kwargs.get("enable_static_kv",
                                   getattr(decoder, "use_static_kv_cache", False))
    enable_static_kv = bool(enable_static_kv and getattr(decoder, "use_static_kv_cache", False))
    enable_cuda_graph = resolve_cuda_graph_request(
        kwargs.get("enable_cuda_graph"),
        default=getattr(decoder, "cuda_graph_enabled", False),
        available=getattr(decoder, "cuda_graph_available", False),
        static_kv=enable_static_kv,
    )
    logger.debug("Graph requested=%s default=%s available=%s enabled=%s",
                 kwargs.get("enable_cuda_graph"), decoder.cuda_graph_enabled,
                 decoder.cuda_graph_available, enable_cuda_graph)
    graph_run_enabled = enable_cuda_graph and bsz == 1 and bool(getattr(decoder, "kv_cache_buckets", []))

    current_bucket = None
    bucket_captured = False
    current_lens = None
    graph_key = None
    pos_idx_static: Optional[torch.Tensor] = None
    key_valid_mask: Optional[torch.Tensor] = None

    static_transformer = getattr(decoder, "t2s_transformer_static", None)
    dynamic_transformer = decoder.t2s_transformer
    static_mode_active = enable_static_kv and static_transformer is not None

    request_stats = {
        "total_steps": 0, "graph_replay_steps": 0, "initial_kv": src_len,
        "start_bucket": None, "final_bucket": None, "promotions": 0,
        "static_steps": 0, "dynamic_steps": 0, "capture_attempts": 0,
        "workspace_wait_seconds": _workspace_wait_seconds,
    }

    if static_mode_active:
        transformer = static_transformer
    else:
        transformer = dynamic_transformer

    for idx in tqdm(range(1500)):
        if xy_attn_mask is not None:
            xy_dec, k_cache, v_cache = transformer.process_prompt(
                xy_pos, xy_attn_mask, None
            )
            logits = decoder.ar_predict_layer(xy_dec[:, -1])

            if static_mode_active and k_cache is not None:
                kv_cache_len = k_cache[0].shape[1]
                selected_bucket = _select_bucket(decoder, kv_cache_len)

                if selected_bucket is not None:
                    current_bucket = selected_bucket
                    request_stats["start_bucket"] = request_stats["final_bucket"] = current_bucket
                    batch_size = k_cache[0].shape[0]
                    hidden_dim = k_cache[0].shape[2]
                    device = k_cache[0].device
                    dtype = k_cache[0].dtype

                    if graph_run_enabled:
                        graph_key = current_bucket
                        if graph_key not in decoder.bucket_graphs:
                            request_stats["capture_attempts"] += 1
                            _warmup_and_capture_bucket(decoder, current_bucket, x.device)
                        bucket_captured = graph_key in decoder.bucket_graphs
                    if bucket_captured:
                        static_in = decoder.bucket_static_inputs[graph_key]
                        k_cache_static, v_cache_static = static_in["k_cache"], static_in["v_cache"]
                        key_valid_mask, pos_idx_static = static_in["key_valid_mask"], static_in["pos_idx"]
                    else:
                        k_cache_static = [torch.zeros(batch_size, current_bucket, hidden_dim,
                                                     dtype=dtype, device=device) for _ in k_cache]
                        v_cache_static = [torch.zeros_like(item) for item in k_cache_static]
                        key_valid_mask = torch.zeros(batch_size, current_bucket, dtype=torch.bool, device=device)
                        pos_idx_static = torch.empty(batch_size, 1, hidden_dim, dtype=torch.long, device=device)
                    for k_target, v_target, k_prompt, v_prompt in zip(k_cache_static, v_cache_static, k_cache, v_cache):
                        k_target[:, :kv_cache_len].copy_(k_prompt)
                        v_target[:, :kv_cache_len].copy_(v_prompt)
                        k_target[:, kv_cache_len:].zero_()
                        v_target[:, kv_cache_len:].zero_()
                    k_cache, v_cache = k_cache_static, v_cache_static
                    current_lens = [kv_cache_len] * len(k_cache)
                    key_valid_mask.zero_()
                    key_valid_mask[:, :kv_cache_len] = True
                    pos_idx_static.fill_(kv_cache_len)
                else:
                    static_mode_active = False
                    transformer = dynamic_transformer
                    graph_run_enabled = False

        else:
            if static_mode_active and current_bucket is not None and current_lens is not None:
                if current_lens[0] >= current_bucket:
                    target = (_promote_prepared_bucket(decoder, current_bucket, current_lens[0])
                              if graph_run_enabled and bucket_captured else None)
                    if target is not None:
                        current_bucket = graph_key = target
                        inputs = decoder.bucket_static_inputs[target]
                        k_cache, v_cache = inputs["k_cache"], inputs["v_cache"]
                        key_valid_mask, pos_idx_static = inputs["key_valid_mask"], inputs["pos_idx"]
                        request_stats["promotions"] += 1
                        request_stats["final_bucket"] = target
                    else:
                        length = current_lens[0]
                        k_cache = [item[:, :length].clone() for item in k_cache]
                        v_cache = [item[:, :length].clone() for item in v_cache]
                        static_mode_active = graph_run_enabled = False

            if static_mode_active and current_bucket is not None and current_lens is not None:
                position = current_lens[0]
                replayed = False
                if graph_run_enabled and bucket_captured:
                    with _get_bucket_lock(decoder, graph_key):
                        inputs = decoder.bucket_static_inputs[graph_key]
                        inputs["xy_pos"].copy_(xy_pos)
                        inputs["pos_idx"].fill_(position)
                        try:
                            _replay_cuda_graph(decoder, decoder.bucket_graphs[graph_key], xy_pos.device)
                            logits = decoder.bucket_static_outputs[graph_key]["logits"]
                            replayed = True
                        except RuntimeError as error:
                            graph_run_enabled = bucket_captured = False
                            logger.warning("CUDA Graph replay failed; continuing eager: %s", error)
                if replayed:
                    request_stats["graph_replay_steps"] += 1
                else:
                    pos_idx_static.fill_(position)
                    xy_dec, k_cache, v_cache = static_transformer.decode_next_token_with_static_cache(
                        xy_pos, k_cache, v_cache, pos_idx_static, key_valid_mask, position + 1)
                    logits = decoder.ar_predict_layer(xy_dec[:, -1])
                    request_stats["static_steps"] += 1
                current_lens = [position + 1] * len(k_cache)
            else:
                xy_dec, k_cache, v_cache = dynamic_transformer.decode_next_token(xy_pos, k_cache, v_cache)
                logits = decoder.ar_predict_layer(xy_dec[:, -1])
                request_stats["dynamic_steps"] += 1

        if idx == 0:
            xy_attn_mask = None
        if idx < 11:
            logits = logits[:, :-1]

        samples = sample(
            logits, y, top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty, temperature=temperature
        )[0]

        y = torch.concat([y, samples], dim=1)

        if early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num:
            stop = True

        if _should_stop_on_eos(logits, samples, decoder.EOS):
            stop = True

        if stop:
            if y.shape[1] == 0:
                y = torch.concat([y, torch.zeros_like(samples)], dim=1)
            print(f"T2S Decoding EOS [{prefix_len} -> {y.shape[1]}]")

            if graph_run_enabled and bucket_captured and current_bucket is not None:
                request_stats["total_steps"] = idx + 1
                graph_ratio = (request_stats["graph_replay_steps"] /
                               request_stats["total_steps"] * 100)
                print(f"CUDA Graph stats: "
                      f"total={request_stats['total_steps']}, "
                      f"replay={request_stats['graph_replay_steps']} "
                      f"({graph_ratio:.1f}%), bucket={current_bucket}")
            break

        request_stats["total_steps"] = idx + 1
        y_emb = decoder.ar_audio_embedding(y[:, -1:])
        xy_pos = (y_emb * decoder.ar_audio_position.x_scale
                  + decoder.ar_audio_position.alpha
                  * decoder.ar_audio_position.pe[:, y_len + idx].to(
                      dtype=y_emb.dtype, device=y_emb.device))

    request_stats["total_steps"] = idx + 1
    request_stats["generated_tokens"] = max(0, y.shape[1] - prefix_len - 1)
    request_stats["final_kv"] = src_len + request_stats["graph_replay_steps"] + request_stats["static_steps"] + request_stats["dynamic_steps"]
    decoder.cuda_graph_stats.update(request_stats)
    logger.info("T2S request layout=%s initial_kv=%d start_bucket=%s final_bucket=%s "
                "promotions=%d generated_tokens=%d graph_steps=%d static_steps=%d "
                "dynamic_steps=%d capture_attempts=%d final_kv=%d workspace_wait_seconds=%.6f",
                decoder.kv_cache_layout, src_len, request_stats["start_bucket"],
                request_stats["final_bucket"], request_stats["promotions"],
                request_stats["generated_tokens"], request_stats["graph_replay_steps"],
                request_stats["static_steps"], request_stats["dynamic_steps"],
                request_stats["capture_attempts"], request_stats["final_kv"], _workspace_wait_seconds)
    if ref_free:
        return y[:, :-1], 0
    return y[:, :-1], idx


# ===========================================================================
# Public API: apply_cuda_graph_patch
# ===========================================================================

def apply_cuda_graph_patch(decoder: Text2SemanticDecoder, buckets=None, *, graph_config=None,
                           kv_cache_layout=None):
    """
    Monkey-patch a Text2SemanticDecoder instance for static KV cache + CUDA Graph.

    This is the main entry point. Call it once after loading the T2S model::

        from aquatts.modeling import apply_cuda_graph_patch
        apply_cuda_graph_patch(t2s_model.model)

    Args:
        decoder: The Text2SemanticDecoder submodule of the T2S Lightning module.
        buckets: Optional list of bucket sizes. Defaults to [128, 256, 448, 512, 768, 1024].

    Returns:
        The same decoder instance (mutated in-place).
    """
    _validate_decoder_contract(decoder)
    if getattr(decoder, "_aquatts_cuda_graph_patched", False):
        if kv_cache_layout is not None and resolve_kv_cache_layout(kv_cache_layout) != decoder.kv_cache_layout:
            raise RuntimeError("Changing KV layout requires a new decoder")
        return decoder

    if buckets is None:
        buckets = _DEFAULT_BUCKETS

    # --- Build static-cache blocks from existing layers ---
    blocks_static = []
    for i in range(decoder.num_layers):
        layer = decoder.h.layers[i]
        t2smlp = T2SMLP(
            layer.linear1.weight,
            layer.linear1.bias,
            layer.linear2.weight,
            layer.linear2.bias,
        )
        block_static = T2SBlockWithStaticCache(
            decoder.num_head,
            decoder.model_dim,
            t2smlp,
            layer.self_attn.in_proj_weight,
            layer.self_attn.in_proj_bias,
            layer.self_attn.out_proj.weight,
            layer.self_attn.out_proj.bias,
            layer.norm1.weight,
            layer.norm1.bias,
            layer.norm1.eps,
            layer.norm2.weight,
            layer.norm2.bias,
            layer.norm2.eps,
        )
        blocks_static.append(block_static)

    decoder.t2s_transformer_static = T2STransformerWithStaticCache(
        decoder.num_layers, blocks_static
    )

    # --- Bucket configuration ---
    decoder.kv_cache_buckets = list(buckets)
    decoder._kv_bucket_signature = tuple(buckets)
    decoder.kv_cache_layout = resolve_kv_cache_layout(kv_cache_layout)
    decoder._graph_kv_roots = {}
    decoder._graph_storage_signature = None
    decoder.bucket_graphs = {}
    decoder.bucket_static_inputs = {}
    decoder.bucket_static_outputs = {}
    decoder.bucket_k_cache_buffers = {}
    decoder.bucket_v_cache_buffers = {}

    # Shared graph state is owned for the complete AR request, not one replay.
    decoder._graph_workspaces = GraphWorkspaceRegistry()
    decoder.cuda_graph_lock = decoder._graph_workspaces.lock_for(None)

    # --- Stats ---
    if not hasattr(decoder, "cuda_graph_stats"):
        decoder.cuda_graph_stats = {
            "total_steps": 0,
            "graph_replay_steps": 0,
            "bucket_hits": {},
            "bucket_misses": 0,
            "capture_time": {},
            "warmup_time": {},
        }

    # --- Enable flags ---
    try:
        decoder_device = next(decoder.parameters()).device
    except (StopIteration, AttributeError):
        decoder_device = torch.device("cpu")
    decoder.use_static_kv_cache = (
        torch.cuda.is_available() and decoder_device.type == "cuda"
    )
    config = resolve_cuda_graph_config("full") if graph_config is None else graph_config
    decoder.cuda_graph_available = decoder.use_static_kv_cache and not bool(torch.version.hip)
    decoder.cuda_graph_enabled = bool(config["enable"] and decoder.cuda_graph_available)
    # CUDA stream dependencies already order replay before sampling. A
    # device-wide synchronize after every token is useful only for diagnosis.
    decoder.cuda_graph_replay_sync = _env_flag_enabled(
        "CUDA_GRAPH_REPLAY_SYNC", False
    )

    # --- Bind methods ---
    decoder.precapture_cuda_graph = lambda buckets=None, kv_range=None: \
        _precapture_cuda_graph(decoder, buckets, kv_range)
    decoder._warmup_and_capture_bucket = lambda bs, dev: \
        _warmup_and_capture_bucket(decoder, bs, dev)

    # --- Replace only infer_panel. Upstream infer_panel_naive is a generator
    # used by its batching/streaming entry points and must retain that contract.
    original_infer_panel = decoder.infer_panel
    decoder._aquatts_original_infer_panel = original_infer_panel
    decoder._aquatts_original_infer_panel_naive = decoder.infer_panel_naive

    @cuda_work
    def patched_infer_panel(x, x_lens, prompts, bert_feature,
                            top_k=-100, top_p=100, early_stop_num=-1,
                            temperature=1.0, repetition_penalty=1.35, **kwargs):
        unsupported = set(kwargs) - _SUPPORTED_PATCH_KWARGS
        if unsupported:
            return original_infer_panel(
                x,
                x_lens,
                prompts,
                bert_feature,
                top_k,
                top_p,
                early_stop_num,
                temperature,
                repetition_penalty,
                **kwargs,
            )
        def run_request(wait_seconds=0.0):
            return _patched_infer_panel_naive(
                decoder, x, x_lens, prompts, bert_feature,
                top_k, top_p, early_stop_num, temperature, repetition_penalty,
                _workspace_wait_seconds=wait_seconds, **kwargs
            )

        graph_enabled = resolve_cuda_graph_request(
            kwargs.get("enable_cuda_graph"), default=decoder.cuda_graph_enabled,
            available=decoder.cuda_graph_available,
            static_kv=bool(kwargs.get("enable_static_kv", decoder.use_static_kv_cache)
                           and decoder.use_static_kv_cache),
        )
        bucket = None
        if graph_enabled and x.is_cuda and x.shape[0] == 1:
            prefix = x.shape[1] + (0 if prompts is None else prompts.shape[1])
            bucket = _select_bucket(decoder, prefix)
        if bucket is None:
            return run_request()
        parameter = next(decoder.parameters())
        _check_graph_storage(decoder, parameter.device, parameter.dtype)
        with decoder._graph_workspaces.lease(_workspace_owner(decoder, bucket), x.device) as wait_seconds:
            logger.debug("Graph workspace request wait=%.6fs bucket=%s layout=%s",
                         wait_seconds, bucket, decoder.kv_cache_layout)
            return run_request(wait_seconds)

    decoder.infer_panel = patched_infer_panel
    decoder._aquatts_cuda_graph_patched = True

    # --- Pre-capture all buckets if CUDA Graph is enabled ---
    if decoder.cuda_graph_enabled:
        print(f"Aqua: static KV cache + CUDA Graph patch applied. "
              f"Buckets: {buckets}")
    else:
        print("Aqua: static KV cache patch applied (CUDA Graph disabled).")

    return decoder
