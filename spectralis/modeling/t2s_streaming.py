"""
Static KV Cache and CUDA Graph acceleration for GPT-SoVITS T2S AR decoder.

Key optimizations:
1. T2SBlockWithStaticCache — scatter_ instead of torch.cat for KV cache,
   maintaining fixed-size buffers so CUDA Graph can capture the decode step.
2. Bucketed CUDA Graph — 6 pre-captured graphs (128, 256, 448, 512, 768, 1024)
   with per-bucket threading locks for concurrency-safe replay.
3. apply_cuda_graph_patch() — monkey-patches Text2SemanticDecoder to use
   static cache + optional CUDA Graph during infer_panel_naive.

Graceful degradation: CUDA Graph -> static KV -> dynamic KV fallback chain.
"""

import math
import threading
import time
import inspect as _inspect
from typing import List, Optional, Tuple

import torch
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# Upstream GPT-SoVITS imports (vendored alongside this package)
# ---------------------------------------------------------------------------
from AR.models.t2s_model import (
    T2SBlock,
    T2SMLP,
    T2STransformer,
    Text2SemanticDecoder,
    scaled_dot_product_attention,
    _GRAPH_INITIAL_LEN_STRIDE,
)

__all__ = [
    "T2SBlockWithStaticCache",
    "T2STransformerWithStaticCache",
    "apply_cuda_graph_patch",
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
        torch_sdpa: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        scatter_ writes current-step KV into pos_idx position.
        Attention looks at the full bucket — unwritten positions are 0,
        yielding near-zero softmax weight and negligible impact.
        pos_idx is a persistent GPU tensor updated via fill_() outside the graph,
        so the replay sees the correct write position without shape changes.
        """
        q, k, v = F.linear(x, self.qkv_w, self.qkv_b).chunk(3, dim=-1)

        k_cache.scatter_(1, pos_idx, k)
        v_cache.scatter_(1, pos_idx, v)

        batch_size = q.shape[0]
        kv_len = k_cache.shape[1]  # always bucket_size (static shape, graph-friendly)

        q = q.view(batch_size, 1, self.num_heads, -1).transpose(1, 2)
        k_full = k_cache.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)
        v_full = v_cache.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)

        if torch_sdpa:
            attn = F.scaled_dot_product_attention(q, k_full, v_full)
        else:
            attn = scaled_dot_product_attention(q, k_full, v_full, None)

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
        torch_sdpa: bool = True
    ):
        """All layers share the same pos_idx."""
        for i in range(self.num_blocks):
            x, k_cache[i], v_cache[i] = self.blocks[i].decode_next_token_with_static_cache(
                x, k_cache[i], v_cache[i], pos_idx, torch_sdpa
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
    """Per-bucket lock for concurrency-safe graph capture and replay."""
    if bucket_key is None:
        return decoder.cuda_graph_lock
    with decoder._bucket_locks_guard:
        lock = decoder._bucket_locks.get(bucket_key)
        if lock is None:
            lock = threading.Lock()
            decoder._bucket_locks[bucket_key] = lock
    return lock


def _warmup_and_capture_bucket(decoder, bucket_size, initial_len, device):
    """Warm up and capture a CUDA Graph for a specific (bucket_size, initial_len) pair."""
    graph_key = (bucket_size, initial_len)
    print(f"Capturing CUDA Graph for bucket={bucket_size}, initial_len={initial_len}...")
    warmup_start = time.perf_counter()

    try:
        batch_size = 1
        hidden_dim = decoder.model_dim

        model_dtype = next(decoder.ar_predict_layer.parameters()).dtype

        k_cache = [
            torch.zeros(batch_size, bucket_size, hidden_dim, dtype=model_dtype, device=device)
            for _ in range(decoder.num_layers)
        ]
        v_cache = [
            torch.zeros(batch_size, bucket_size, hidden_dim, dtype=model_dtype, device=device)
            for _ in range(decoder.num_layers)
        ]

        for i in range(decoder.num_layers):
            k_cache[i][:, :initial_len, :] = torch.randn(
                batch_size, initial_len, hidden_dim, dtype=model_dtype, device=device
            )
            v_cache[i][:, :initial_len, :] = torch.randn(
                batch_size, initial_len, hidden_dim, dtype=model_dtype, device=device
            )

        xy_pos = torch.randn(batch_size, 1, hidden_dim, dtype=model_dtype, device=device)
        pos_idx = torch.full(
            (batch_size, 1, hidden_dim), initial_len, dtype=torch.long, device=device
        )

        # Warmup
        for _ in range(_CUDA_GRAPH_WARMUP_STEPS):
            xy_dec, k_cache, v_cache = decoder.t2s_transformer_static.decode_next_token_with_static_cache(
                xy_pos, k_cache, v_cache, pos_idx
            )
            logits = decoder.ar_predict_layer(xy_dec[:, -1])

        warmup_time = time.perf_counter() - warmup_start
        decoder.cuda_graph_stats["warmup_time"][graph_key] = warmup_time
        print(f"  Warmup complete: {warmup_time:.4f}s")

        # Reset buffers to fixed state
        k_cache = [
            torch.zeros(batch_size, bucket_size, hidden_dim, dtype=model_dtype, device=device)
            for _ in range(decoder.num_layers)
        ]
        v_cache = [
            torch.zeros(batch_size, bucket_size, hidden_dim, dtype=model_dtype, device=device)
            for _ in range(decoder.num_layers)
        ]
        for i in range(decoder.num_layers):
            k_cache[i][:, :initial_len, :] = torch.randn(
                batch_size, initial_len, hidden_dim, dtype=model_dtype, device=device
            )
            v_cache[i][:, :initial_len, :] = torch.randn(
                batch_size, initial_len, hidden_dim, dtype=model_dtype, device=device
            )
        xy_pos = torch.randn(batch_size, 1, hidden_dim, dtype=model_dtype, device=device)
        pos_idx = torch.full(
            (batch_size, 1, hidden_dim), initial_len, dtype=torch.long, device=device
        )

        # Capture
        capture_start = time.perf_counter()
        try:
            _has_capture_error_mode = (
                "capture_error_mode"
                in _inspect.signature(torch.cuda.CUDAGraph.__init__).parameters
            )
        except (ValueError, TypeError):
            _has_capture_error_mode = False  # PyTorch < 2.5
        _graph_extra = {"capture_error_mode": "relaxed"} if _has_capture_error_mode else {}
        with torch.cuda.device(device):
            cuda_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cuda_graph, **_graph_extra):
                xy_dec, k_cache_out, v_cache_out = decoder.t2s_transformer_static.decode_next_token_with_static_cache(
                    xy_pos, k_cache, v_cache, pos_idx
                )
                logits = decoder.ar_predict_layer(xy_dec[:, -1])

        capture_time = time.perf_counter() - capture_start
        decoder.cuda_graph_stats["capture_time"][graph_key] = capture_time

        decoder.bucket_graphs[graph_key] = cuda_graph
        decoder.bucket_static_inputs[graph_key] = {
            "xy_pos": xy_pos,
            "k_cache": k_cache,
            "v_cache": v_cache,
            "pos_idx": pos_idx,
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
    """Pre-capture CUDA Graphs for expected kv_cache_len ranges."""
    if not getattr(decoder, "cuda_graph_enabled", False):
        print("CUDA Graph not enabled, skipping precapture")
        return {}

    available = getattr(decoder, "kv_cache_buckets", _DEFAULT_BUCKETS)
    if not available:
        return {}

    device = next(decoder.parameters()).device
    target_buckets = buckets or [available[0]]
    results = {}

    _s = _GRAPH_INITIAL_LEN_STRIDE

    for bucket in target_buckets:
        if bucket not in available:
            results[bucket] = False
            continue

        if kv_len_range:
            lo, hi = kv_len_range
        else:
            G = _BUCKET_MIN_GEN_SLOTS
            _sb = sorted(available)
            _bi = _sb.index(bucket)
            prev_b = _sb[_bi - 1] if _bi > 0 else 0
            lo = max(_s * 8, prev_b - G)
            hi = min(bucket - G - 1, int(bucket * 0.75))

        min_il = ((lo + _s - 1) // _s) * _s
        max_il = (hi // _s) * _s
        initial_lens = [il for il in range(min_il, max_il + 1, _s) if il < bucket]

        bucket_ok = True
        for initial_len in initial_lens:
            graph_key = (bucket, initial_len)
            lock = _get_bucket_lock(decoder, graph_key)
            with lock:
                if graph_key in decoder.bucket_graphs:
                    continue
                success = _warmup_and_capture_bucket(decoder, bucket, initial_len, device)
                if not success:
                    bucket_ok = False
        results[bucket] = bucket_ok

    return results


def _ensure_directories_exist(path):
    """Create parent directories for path if needed."""
    import os
    os.makedirs(path, exist_ok=True)


# ===========================================================================
# Patched infer_panel_naive — static KV + optional CUDA Graph
# ===========================================================================

def _patched_infer_panel_naive(decoder, x, x_lens, prompts, bert_feature,
                                top_k=-100, top_p=100, early_stop_num=-1,
                                temperature=1.0, repetition_penalty=1.35, **kwargs):
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

    enable_cuda_graph = kwargs.get("enable_cuda_graph",
                                    getattr(decoder, "cuda_graph_enabled", False))
    enable_static_kv = kwargs.get("enable_static_kv",
                                   getattr(decoder, "use_static_kv_cache", False))
    if enable_cuda_graph and not getattr(decoder, "use_static_kv_cache", False):
        enable_cuda_graph = False
    if not enable_static_kv:
        enable_cuda_graph = False
    graph_run_enabled = enable_cuda_graph and bool(getattr(decoder, "kv_cache_buckets", []))

    current_bucket = None
    bucket_captured = False
    current_lens = None
    graph_key = None
    graph_initial_len = None
    graph_step_count = 0
    pos_idx_static: Optional[torch.Tensor] = None

    static_transformer = getattr(decoder, "t2s_transformer_static", None)
    dynamic_transformer = decoder.t2s_transformer
    static_mode_active = enable_static_kv and static_transformer is not None

    decoder.cuda_graph_stats["total_steps"] = 0
    decoder.cuda_graph_stats["graph_replay_steps"] = 0

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
                    batch_size = k_cache[0].shape[0]
                    hidden_dim = k_cache[0].shape[2]
                    device = k_cache[0].device
                    dtype = k_cache[0].dtype

                    k_cache_static = [
                        torch.zeros(batch_size, current_bucket, hidden_dim, dtype=dtype, device=device)
                        for _ in range(len(k_cache))
                    ]
                    v_cache_static = [
                        torch.zeros(batch_size, current_bucket, hidden_dim, dtype=dtype, device=device)
                        for _ in range(len(v_cache))
                    ]

                    for i in range(len(k_cache)):
                        k_cache_static[i][:, :kv_cache_len, :] = k_cache[i]
                        v_cache_static[i][:, :kv_cache_len, :] = v_cache[i]

                    k_cache = k_cache_static
                    v_cache = v_cache_static
                    current_lens = [kv_cache_len] * len(k_cache)

                    pos_idx_static = torch.full(
                        (batch_size, 1, hidden_dim), kv_cache_len,
                        dtype=torch.long, device=device
                    )

                    if graph_run_enabled:
                        if kv_cache_len >= current_bucket - 1:
                            graph_run_enabled = False
                        else:
                            _aligned = ((kv_cache_len + _GRAPH_INITIAL_LEN_STRIDE - 1)
                                        // _GRAPH_INITIAL_LEN_STRIDE * _GRAPH_INITIAL_LEN_STRIDE)
                            if _aligned >= current_bucket:
                                graph_run_enabled = False
                            else:
                                graph_initial_len = _aligned
                                graph_key = (current_bucket, graph_initial_len)
                                if graph_key not in decoder.bucket_graphs:
                                    bucket_lock = _get_bucket_lock(decoder, graph_key)
                                    with bucket_lock:
                                        if graph_key not in decoder.bucket_graphs:
                                            success = _warmup_and_capture_bucket(
                                                decoder, current_bucket, graph_initial_len, x.device
                                            )
                                            if success:
                                                bucket_captured = True
                                        else:
                                            bucket_captured = True
                                else:
                                    bucket_captured = True

                                if bucket_captured and graph_key in decoder.bucket_graphs:
                                    static_in = decoder.bucket_static_inputs[graph_key]
                                    for _i in range(len(static_in["k_cache"])):
                                        static_in["k_cache"][_i][:, :kv_cache_len, :].copy_(
                                            k_cache[_i][:, :kv_cache_len, :]
                                        )
                                        static_in["v_cache"][_i][:, :kv_cache_len, :].copy_(
                                            v_cache[_i][:, :kv_cache_len, :]
                                        )
                                        static_in["k_cache"][_i][:, kv_cache_len:, :].zero_()
                                        static_in["v_cache"][_i][:, kv_cache_len:, :].zero_()
                                    static_in["pos_idx"].fill_(graph_initial_len)
                else:
                    static_mode_active = False
                    transformer = dynamic_transformer
                    graph_run_enabled = False

        elif static_mode_active and current_bucket is not None and current_lens is not None:
            if current_lens[0] >= current_bucket - 1:
                keep_len = current_bucket - 1
                if not hasattr(decoder, "_sliding_window_triggered"):
                    decoder._sliding_window_triggered = True
                    print(f"Sliding window: keeping last {keep_len} tokens")
                for i in range(len(k_cache)):
                    k_cache[i][:, :keep_len, :] = k_cache[i][:, -keep_len:, :].clone()
                    v_cache[i][:, :keep_len, :] = v_cache[i][:, -keep_len:, :].clone()
                    k_cache[i][:, keep_len:, :].zero_()
                    v_cache[i][:, keep_len:, :].zero_()
                current_lens = [keep_len] * len(current_lens)

            if graph_run_enabled and bucket_captured and graph_key is not None and graph_key in decoder.bucket_graphs:
                replay_failed = False
                bucket_lock = _get_bucket_lock(decoder, graph_key)
                with bucket_lock:
                    try:
                        cuda_graph = decoder.bucket_graphs[graph_key]
                        static_inputs = decoder.bucket_static_inputs[graph_key]
                        static_outputs = decoder.bucket_static_outputs[graph_key]

                        static_inputs["xy_pos"].copy_(xy_pos)
                        static_inputs["pos_idx"].fill_(graph_initial_len + graph_step_count)

                        cuda_graph.replay()
                        torch.cuda.synchronize(xy_pos.device)

                        logits = static_outputs["logits"]
                        decoder.cuda_graph_stats["graph_replay_steps"] += 1
                        graph_step_count += 1

                        if not hasattr(decoder, "_cuda_graph_replay_started"):
                            decoder._cuda_graph_replay_started = True
                            print(f"CUDA Graph replay active (key={graph_key})")

                        if graph_initial_len + graph_step_count >= current_bucket:
                            graph_run_enabled = False
                            print(f"Graph write position at bucket boundary, falling back to static path")
                            _fallback_len = graph_initial_len + graph_step_count
                            static_in_fb = decoder.bucket_static_inputs[graph_key]
                            for _fi in range(len(k_cache)):
                                k_cache[_fi].copy_(static_in_fb["k_cache"][_fi])
                                v_cache[_fi].copy_(static_in_fb["v_cache"][_fi])
                            current_lens = [_fallback_len] * len(k_cache)
                            if pos_idx_static is not None:
                                pos_idx_static.fill_(_fallback_len)
                            else:
                                pos_idx_static = k_cache[0].new_full(
                                    (k_cache[0].shape[0], 1, k_cache[0].shape[2]),
                                    _fallback_len, dtype=torch.long)
                    except RuntimeError as e:
                        replay_failed = True
                        graph_run_enabled = False
                        bucket_captured = False
                        print(f"CUDA Graph replay failed, falling back to static: {e!r}")
                if replay_failed:
                    static_inputs = decoder.bucket_static_inputs[graph_key]
                    for i in range(len(k_cache)):
                        k_cache[i].copy_(static_inputs["k_cache"][i])
                        v_cache[i].copy_(static_inputs["v_cache"][i])
                    _fallback_len = graph_initial_len + graph_step_count
                    current_lens = [_fallback_len] * len(k_cache)
                    if pos_idx_static is not None:
                        pos_idx_static.fill_(_fallback_len)
                    else:
                        pos_idx_static = k_cache[0].new_full(
                            (k_cache[0].shape[0], 1, k_cache[0].shape[2]),
                            _fallback_len, dtype=torch.long)
                    xy_dec, k_cache, v_cache = static_transformer.decode_next_token_with_static_cache(
                        xy_pos, k_cache, v_cache, pos_idx_static
                    )
                    current_lens = [l + 1 for l in current_lens]
                    logits = decoder.ar_predict_layer(xy_dec[:, -1])
            else:
                if pos_idx_static is not None:
                    pos_idx_static.fill_(current_lens[0])
                else:
                    pos_idx_static = k_cache[0].new_full(
                        (k_cache[0].shape[0], 1, k_cache[0].shape[2]),
                        current_lens[0], dtype=torch.long)
                xy_dec, k_cache, v_cache = static_transformer.decode_next_token_with_static_cache(
                    xy_pos, k_cache, v_cache, pos_idx_static
                )
                current_lens = [l + 1 for l in current_lens]
                logits = decoder.ar_predict_layer(xy_dec[:, -1])
        else:
            if transformer is not dynamic_transformer:
                transformer = dynamic_transformer
            xy_dec, k_cache, v_cache = dynamic_transformer.decode_next_token(
                xy_pos, k_cache, v_cache
            )
            logits = decoder.ar_predict_layer(xy_dec[:, -1])

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

        if torch.argmax(logits, dim=-1)[0] == decoder.EOS or samples[0, 0] == decoder.EOS:
            stop = True

        if stop:
            if y.shape[1] == 0:
                y = torch.concat([y, torch.zeros_like(samples)], dim=1)
            print(f"T2S Decoding EOS [{prefix_len} -> {y.shape[1]}]")

            if graph_run_enabled and bucket_captured and current_bucket is not None:
                decoder.cuda_graph_stats["total_steps"] = idx + 1
                graph_ratio = (decoder.cuda_graph_stats["graph_replay_steps"] /
                               decoder.cuda_graph_stats["total_steps"] * 100)
                print(f"CUDA Graph stats: "
                      f"total={decoder.cuda_graph_stats['total_steps']}, "
                      f"replay={decoder.cuda_graph_stats['graph_replay_steps']} "
                      f"({graph_ratio:.1f}%), bucket={current_bucket}")
            break

        decoder.cuda_graph_stats["total_steps"] = idx + 1
        y_emb = decoder.ar_audio_embedding(y[:, -1:])
        xy_pos = (y_emb * decoder.ar_audio_position.x_scale
                  + decoder.ar_audio_position.alpha
                  * decoder.ar_audio_position.pe[:, y_len + idx].to(
                      dtype=y_emb.dtype, device=y_emb.device))

    if ref_free:
        return y[:, :-1], 0
    return y[:, :-1], idx


# ===========================================================================
# Public API: apply_cuda_graph_patch
# ===========================================================================

def apply_cuda_graph_patch(decoder: Text2SemanticDecoder, buckets=None):
    """
    Monkey-patch a Text2SemanticDecoder instance for static KV cache + CUDA Graph.

    This is the main entry point. Call it once after loading the T2S model::

        from spectralis.modeling import apply_cuda_graph_patch
        apply_cuda_graph_patch(t2s_model.model)

    Args:
        decoder: The Text2SemanticDecoder submodule of the T2S Lightning module.
        buckets: Optional list of bucket sizes. Defaults to [128, 256, 448, 512, 768, 1024].

    Returns:
        The same decoder instance (mutated in-place).
    """
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
    decoder.kv_cache_buckets = buckets
    decoder.bucket_graphs = {}
    decoder.bucket_static_inputs = {}
    decoder.bucket_static_outputs = {}
    decoder.bucket_k_cache_buffers = {}
    decoder.bucket_v_cache_buffers = {}

    # --- Threading ---
    if not hasattr(decoder, "cuda_graph_lock"):
        decoder.cuda_graph_lock = threading.Lock()
    decoder._bucket_locks = {}
    decoder._bucket_locks_guard = threading.Lock()

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
    decoder.use_static_kv_cache = torch.cuda.is_available()
    import os
    cuda_graph_env = os.environ.get("ENABLE_CUDA_GRAPH", "1")
    decoder.cuda_graph_enabled = decoder.use_static_kv_cache and (cuda_graph_env == "1")

    # --- Bind methods ---
    decoder.precapture_cuda_graph = lambda buckets=None, kv_range=None: \
        _precapture_cuda_graph(decoder, buckets, kv_range)
    decoder._warmup_and_capture_bucket = lambda bs, il, dev: \
        _warmup_and_capture_bucket(decoder, bs, il, dev)

    # --- Replace infer_panel (which delegates to infer_panel_naive) ---
    original_infer_panel_naive = decoder.infer_panel_naive

    def patched_infer_panel_naive(x, x_lens, prompts, bert_feature,
                                   top_k=-100, top_p=100, early_stop_num=-1,
                                   temperature=1.0, repetition_penalty=1.35, **kwargs):
        return _patched_infer_panel_naive(
            decoder, x, x_lens, prompts, bert_feature,
            top_k, top_p, early_stop_num, temperature, repetition_penalty, **kwargs
        )

    decoder.infer_panel_naive = patched_infer_panel_naive
    decoder.infer_panel = patched_infer_panel_naive

    # --- Pre-capture all buckets if CUDA Graph is enabled ---
    if decoder.cuda_graph_enabled:
        print(f"Spectralis: static KV cache + CUDA Graph patch applied. "
              f"Buckets: {buckets}")
    else:
        print(f"Spectralis: static KV cache patch applied (CUDA Graph disabled).")

    return decoder
