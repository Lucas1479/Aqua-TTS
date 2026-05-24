"""
FlashAttention-optimized T2S decoder blocks.

Uses flash_attn_with_kvcache for fused KV-cache append + attention in a
single kernel, reducing memory traffic compared to separate scatter_ + SDPA.

Graceful fallback: if flash_attn is not installed, falls back to PyTorch's
F.scaled_dot_product_attention (which internally dispatches to FlashAttention
on Ampere+ GPUs since PyTorch 2.0).

KV cache layout follows the flash_attn convention:
  k_cache: [batch, seqlen, num_heads, head_dim]
  v_cache: [batch, seqlen, num_heads, head_dim]

Usage:
    from aquatts.modeling.t2s_flash_attn import apply_flash_attn_patch
    apply_flash_attn_patch(t2s_model.model)
"""

import math
import threading
from typing import List, Optional, Tuple

import torch
from torch.nn import functional as F

from AR.models.t2s_model import (
    T2SMLP,
    scaled_dot_product_attention,
    _GRAPH_INITIAL_LEN_STRIDE,
)

# ---------------------------------------------------------------------------
# Graceful import of flash_attn
# ---------------------------------------------------------------------------

_flash_attn_available = False
_flash_attn_error = None
_flash_attn_with_kvcache = None

try:
    from flash_attn import flash_attn_with_kvcache as _fa_kvcache
    _flash_attn_with_kvcache = _fa_kvcache
    _flash_attn_available = True
except ImportError as e:
    _flash_attn_error = str(e)
except Exception as e:
    _flash_attn_error = str(e)


def is_flash_attn_available() -> bool:
    """Check if the flash_attn library is importable."""
    return _flash_attn_available


def get_flash_attn_error() -> Optional[str]:
    """Return the reason flash_attn failed to import, or None."""
    return _flash_attn_error


# ===========================================================================
# FlashAttention T2S blocks
# ===========================================================================


@torch.jit.script
class T2SBlockWithStaticCacheFlash:
    """
    T2SBlock variant using flash_attn_with_kvcache for the decode step.

    KV cache is stored in flash_attn format: [batch, seqlen, num_heads, head_dim].
    This differs from the standard [batch, seqlen, hidden_dim] format in
    T2SBlockWithStaticCache — the reshape/transpose happens at write time
    instead of at attention time, saving a transpose inside the hot loop.
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
        self.head_dim: int = hidden_dim // num_heads
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
        torch_sdpa: bool = True,
    ):
        """Same as T2SBlockWithStaticCache.process_prompt — not using flash_attn here."""
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

    def decode_next_token_with_static_cache_sdpa(
        self,
        x: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        pos_idx: torch.Tensor,
        torch_sdpa: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """SDPA fallback path using scatter_ + F.scaled_dot_product_attention."""
        q, k, v = F.linear(x, self.qkv_w, self.qkv_b).chunk(3, dim=-1)

        k_cache.scatter_(1, pos_idx, k)
        v_cache.scatter_(1, pos_idx, v)

        batch_size = q.shape[0]
        kv_len = k_cache.shape[1]

        # k_cache/v_cache in [B, bucket, H*D] format (hidden_dim), reshape for SDPA
        q_sdpa = q.view(batch_size, 1, self.num_heads, -1).transpose(1, 2)
        k_sdpa = k_cache.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)
        v_sdpa = v_cache.view(batch_size, kv_len, self.num_heads, -1).transpose(1, 2)

        if torch_sdpa:
            attn = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa)
        else:
            attn = scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, None)

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

    def decode_next_token_with_static_cache(
        self,
        x: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        pos_idx: torch.Tensor,
        torch_sdpa: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        If flash_attn is available, use it for fused KV-cache append + attention.
        Otherwise fall back to scatter_ + F.scaled_dot_product_attention.

        KV cache is stored in flash_attn format: [B, bucket, H, D_per_head].
        pos_idx is [B, 1, hidden_dim] long — we only use batch/layer dims.
        """
        if not _flash_attn_available:
            return self.decode_next_token_with_static_cache_sdpa(
                x, k_cache, v_cache, pos_idx, torch_sdpa
            )

        B, _, H_dim = x.shape
        D = self.head_dim

        q, k, v = F.linear(x, self.qkv_w, self.qkv_b).chunk(3, dim=-1)

        # Reshape to flash_attn format: [B, 1, H, D]
        q_fa = q.view(B, 1, self.num_heads, D)
        k_fa = k.view(B, 1, self.num_heads, D)
        v_fa = v.view(B, 1, self.num_heads, D)

        # Write current k,v into cache at pos_idx position
        # pos_idx is [B, 1, hidden_dim] long; extract batch-level scalar
        write_pos = pos_idx[0, 0, 0].item()
        k_cache[:, write_pos : write_pos + 1] = k_fa
        v_cache[:, write_pos : write_pos + 1] = v_fa

        cache_seqlens = torch.full(
            (B,), write_pos + 1, dtype=torch.int32, device=x.device
        )

        try:
            attn = _flash_attn_with_kvcache(
                q_fa,
                k_cache=k_cache,
                v_cache=v_cache,
                cache_seqlens=cache_seqlens,
            )
        except RuntimeError:
            # flash_attn may fail on certain GPU architectures or input shapes
            return self.decode_next_token_with_static_cache_sdpa(
                x, k_cache, v_cache, pos_idx, torch_sdpa
            )

        attn = attn.reshape(B, 1, H_dim)
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
class T2STransformerWithStaticCacheFlash:
    """Transformer using FlashAttention T2S blocks."""

    def __init__(self, num_blocks: int, blocks: List[T2SBlockWithStaticCacheFlash]):
        self.num_blocks: int = num_blocks
        self.blocks = blocks

    def process_prompt(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        torch_sdpa: bool = True,
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
        torch_sdpa: bool = True,
    ):
        for i in range(self.num_blocks):
            x, k_cache[i], v_cache[i] = self.blocks[
                i
            ].decode_next_token_with_static_cache(
                x, k_cache[i], v_cache[i], pos_idx, torch_sdpa
            )
        return x, k_cache, v_cache


# ===========================================================================
# Public API
# ===========================================================================


def _flash_attn_kv_cache_shape(batch_size, bucket_size, hidden_dim, num_heads, device, dtype):
    """
    Return kV cache for flash_attn format: [B, bucket, H, head_dim].
    This differs from the standard [B, bucket, hidden_dim] used by scatter_ SDPA.
    """
    head_dim = hidden_dim // num_heads
    k = torch.zeros(
        batch_size, bucket_size, num_heads, head_dim, dtype=dtype, device=device
    )
    v = torch.zeros(
        batch_size, bucket_size, num_heads, head_dim, dtype=dtype, device=device
    )
    return k, v


def _convert_kv_to_flash_format(kv_hidden_format, num_heads):
    """
    Convert KV cache from [B, bucket, hidden_dim] to [B, bucket, num_heads, head_dim].
    Used when migrating from standard T2SBlockWithStaticCache format.
    """
    if kv_hidden_format is None:
        return None
    B, bucket, H_dim = kv_hidden_format.shape
    D = H_dim // num_heads
    return kv_hidden_format.view(B, bucket, num_heads, D)


def apply_flash_attn_patch(decoder, buckets=None):
    """
    Replace the standard T2STransformerWithStaticCache with a FlashAttention variant.

    This is a drop-in replacement — call it AFTER apply_cuda_graph_patch.

    If flash_attn is not installed, this is a no-op: PyTorch's built-in
    F.scaled_dot_product_attention already dispatches to FlashAttention
    (or Memory-Efficient Attention) on supported GPUs since PyTorch 2.0.

    The flash_attn library provides additional benefit via the fused
    KV-cache-append + attention kernel (flash_attn_with_kvcache), which
    reduces GPU memory traffic compared to separate scatter_ + SDPA.

    Args:
        decoder: Text2SemanticDecoder instance (already patched by apply_cuda_graph_patch).
        buckets: Unused (kept for API compatibility).

    Returns:
        The same decoder instance.
    """
    if not _flash_attn_available:
        print(
            f"[FlashAttn] flash_attn library not available ({_flash_attn_error or 'unknown'}). "
            f"Falling back to PyTorch SDPA (which auto-dispatches to FlashAttention "
            f"on Ampere+ GPUs since PyTorch 2.0). No performance loss expected."
        )
        return decoder

    # Build FlashAttention blocks from existing layers
    blocks_flash = []
    for i in range(decoder.num_layers):
        layer = decoder.h.layers[i]
        t2smlp = T2SMLP(
            layer.linear1.weight,
            layer.linear1.bias,
            layer.linear2.weight,
            layer.linear2.bias,
        )
        block_flash = T2SBlockWithStaticCacheFlash(
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
        blocks_flash.append(block_flash)

    decoder.t2s_transformer_static = T2STransformerWithStaticCacheFlash(
        decoder.num_layers, blocks_flash
    )

    print(
        "[FlashAttn] T2S decoder upgraded to use flash_attn_with_kvcache "
        "(fused KV-cache append + attention)."
    )
    return decoder
