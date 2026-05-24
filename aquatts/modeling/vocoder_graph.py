"""
Bucketed CUDA Graph wrapper for the SoVITS/BigVGAN vocoder forward pass.

Captures the flow + decoder step (self.flow + self.dec) inside CUDA Graphs
pre-sized for common mel-T lengths, eliminating kernel launch overhead on
each decode call.

This follows the same pattern as GSV-TTS-Lite's SoVITS CUDA Graph:
  ┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
  │  quantizer   │ →   │  flow + decoder  │ →   │    audio     │
  │  decode      │     │  ← CUDA Graph →  │     │              │
  └──────────────┘     └─────────────────┘     └──────────────┘

Usage:
    from aquatts.modeling.vocoder_graph import VocoderGraphManager

    # After loading the SoVITS model:
    vgm = VocoderGraphManager(vq_model)
    vgm.precapture()  # pre-capture graphs for common mel-T buckets

    # During inference, the decode() method automatically uses graph replay.
"""

import threading
import time
from typing import Dict, List, Optional

import torch
from torch.nn import functional as F


# Default mel-T bucket sizes for CUDA Graph capture.
# These cover typical first-chunk to full-utterance mel sizes
# after quantizer decode + interpolation (x2).
#  - 70  → ~0.35s audio (common first chunk)
#  - 128 → ~0.64s audio
#  - 256 → ~1.28s audio
#  - 512 → ~2.56s audio (typical full utterance)
#  - 768 → ~3.84s audio
#  - 1024 → ~5.12s audio (long utterance)
_DEFAULT_VOCODER_BUCKETS = [70, 128, 256, 512, 768, 1024]


class _VocoderBucket:
    """Holds pre-allocated buffers and CUDA Graph for one mel-T size bucket."""

    __slots__ = (
        "mel_t",
        "z_p_padded", "y_mask_padded", "ge_padded",
        "dec_o", "cuda_graph", "lock",
    )

    def __init__(self):
        self.mel_t: int = 0
        self.z_p_padded: Optional[torch.Tensor] = None
        self.y_mask_padded: Optional[torch.Tensor] = None
        self.ge_padded: Optional[torch.Tensor] = None
        self.dec_o: Optional[torch.Tensor] = None
        self.cuda_graph: Optional[torch.cuda.CUDAGraph] = None
        self.lock = threading.Lock()


class VocoderGraphManager:
    """Manager for bucketed CUDA Graph capture on a SoVITS/BigVGAN vocoder."""

    def __init__(
        self,
        vq_model: torch.nn.Module,
        bucket_sizes: Optional[List[int]] = None,
        warmup_steps: int = 3,
    ):
        """
        Args:
            vq_model: SoVITS vocoder model (SynthesizerTrn or SynthesizerTrnV3).
            bucket_sizes: List of mel-T bucket sizes. Defaults cover streaming chunks.
            warmup_steps: Number of warmup runs before each graph capture.
        """
        self.vq_model = vq_model
        self.bucket_sizes = bucket_sizes or _DEFAULT_VOCODER_BUCKETS
        self.warmup_steps = warmup_steps

        # Per-bucket state
        self.buckets: Dict[int, _VocoderBucket] = {}
        self._captured = False

        # Reference to original decode
        self._original_decode = None

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def precapture(self, device: Optional[torch.device] = None) -> bool:
        """
        Pre-capture CUDA Graphs for all bucket sizes.

        Returns True if all buckets captured successfully.
        """
        if device is None:
            device = next(self.vq_model.parameters()).device
        if device.type != "cuda":
            print("[VocoderGraph] Not CUDA device, skipping capture")
            return False

        # Auto-detect model attributes
        try:
            latent_channels = self.vq_model.enc_p.latent_channels
        except AttributeError:
            latent_channels = self.vq_model.generator.latent_channels  # v3

        try:
            gin_channels = self.vq_model.gin_channels
        except AttributeError:
            gin_channels = self.vq_model.gen_channels  # v3 fallback

        batch_size = 1
        dtype = next(self.vq_model.parameters()).dtype

        success_count = 0

        for mel_t in sorted(self.bucket_sizes):
            bucket = _VocoderBucket()
            bucket.mel_t = mel_t

            try:
                # Pre-allocate padded tensors
                bucket.z_p_padded = torch.zeros(
                    batch_size, latent_channels, mel_t,
                    dtype=dtype, device=device
                )
                bucket.y_mask_padded = torch.zeros(
                    batch_size, 1, mel_t,
                    dtype=dtype, device=device
                )
                bucket.ge_padded = torch.zeros(
                    batch_size, gin_channels, 1,
                    dtype=dtype, device=device
                )

                # Fill with random data to avoid zero-only paths
                bucket.z_p_padded.normal_()
                bucket.y_mask_padded.fill_(1.0)
                bucket.ge_padded.normal_()

                # Warmup
                for _ in range(self.warmup_steps):
                    z = self.vq_model.flow(
                        bucket.z_p_padded, bucket.y_mask_padded, bucket.ge_padded
                    )
                    self.vq_model.dec(
                        z * bucket.y_mask_padded, g=bucket.ge_padded
                    )

                if device.type == "cuda":
                    torch.cuda.synchronize(device)

                    # Capture CUDA Graph
                    import inspect as _inspect
                    try:
                        _has_cem = (
                            "capture_error_mode"
                            in _inspect.signature(
                                torch.cuda.CUDAGraph.__init__
                            ).parameters
                        )
                    except (ValueError, TypeError):
                        _has_cem = False

                    _graph_extra = (
                        {"capture_error_mode": "relaxed"}
                        if _has_cem else {}
                    )

                    with torch.cuda.device(device):
                        bucket.cuda_graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(
                            bucket.cuda_graph, **_graph_extra
                        ):
                            z = self.vq_model.flow(
                                bucket.z_p_padded,
                                bucket.y_mask_padded,
                                bucket.ge_padded,
                            )
                            bucket.dec_o = self.vq_model.dec(
                                z * bucket.y_mask_padded,
                                g=bucket.ge_padded,
                            )

                self.buckets[mel_t] = bucket
                success_count += 1
                print(
                    f"[VocoderGraph] Captured bucket mel_T={mel_t} "
                    f"(latent={latent_channels}, gin={gin_channels})"
                )

            except Exception as e:
                print(
                    f"[VocoderGraph] Failed to capture bucket mel_T={mel_t}: {e}"
                )
                import traceback
                traceback.print_exc()

        self._captured = success_count > 0
        print(
            f"[VocoderGraph] Captured {success_count}/{len(self.bucket_sizes)} "
            f"vocoder buckets"
        )
        return self._captured

    # ------------------------------------------------------------------
    # Graph replay
    # ------------------------------------------------------------------

    def decode(self, z_p, y_mask, ge, samples_per_frame=2):
        """
        Decode z_p → audio using CUDA Graph replay when a bucket fits.

        Args:
            z_p: [B, latent_C, mel_T]
            y_mask: [B, 1, mel_T]
            ge: [B, gin_C, 1]
            samples_per_frame: Audio samples per mel frame (usually 2).

        Returns:
            audio: [B, 1, mel_T * samples_per_frame]
        """
        if not self._captured:
            z = self.vq_model.flow(z_p, y_mask, ge)
            return self.vq_model.dec(z * y_mask, g=ge)

        z_current_length = z_p.size(-1)

        # Find smallest bucket >= current length
        matched = None
        for mel_t in sorted(self.bucket_sizes):
            if mel_t >= z_current_length:
                matched = mel_t
                break

        if matched is None:
            # No bucket fits — fallback to normal execution
            z = self.vq_model.flow(z_p, y_mask, ge)
            return self.vq_model.dec(z * y_mask, g=ge)

        bucket = self.buckets.get(matched)
        if bucket is None or bucket.cuda_graph is None:
            z = self.vq_model.flow(z_p, y_mask, ge)
            return self.vq_model.dec(z * y_mask, g=ge)

        with bucket.lock:
            # Copy real data into padded buffers
            bucket.z_p_padded.zero_()
            bucket.y_mask_padded.zero_()
            bucket.z_p_padded[:, :, :z_current_length].copy_(z_p)
            bucket.y_mask_padded[:, :, :z_current_length].copy_(y_mask)
            bucket.ge_padded.copy_(ge)

            bucket.cuda_graph.replay()
            torch.cuda.synchronize(z_p.device)

            # Slice output to actual length
            audio_len = z_current_length * samples_per_frame
            return bucket.dec_o[:, :, :audio_len]

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------

    def patch(self):
        """Replace vq_model.decode with bucket.graph.replay path."""
        if self._original_decode is None:
            self._original_decode = self.vq_model.decode

        manager = self

        def patched_decode(codes, text, ge, noise_scale=0.5, speed=1,
                          cuda_graph=True, stream_mode=False,
                          valid_start_idx=None, overlap_len=None,
                          slice_indices=None):
            """Patched decode: original pipeline with flow+decoder via graph."""
            # Run the upstream decode up to the flow+decoder step
            # This handles quantizer, enc_p.infer, etc.
            # Then use our CUDA Graph for the flow + dec step.

            import inspect
            orig = manager._original_decode
            sig = inspect.signature(orig)

            # For v3 models, the decode signature may differ.
            # We extract the flow+dec step and replace it.
            # For now, delegate to the original and wrap only if cuda_graph=True.
            if not cuda_graph or not manager._captured:
                return orig(
                    codes, text, ge, noise_scale, speed,
                    stream_mode=stream_mode,
                    valid_start_idx=valid_start_idx,
                    overlap_len=overlap_len,
                    slice_indices=slice_indices,
                )

            # Run the quantizer + enc_p.infer part (same as original)
            quantized = manager.vq_model.quantizer.decode(codes)
            quantized = F.interpolate(
                quantized,
                size=quantized.shape[-1] * 2,
                mode="nearest",
            )
            if ge.shape[-1] != 1:
                ge = F.interpolate(
                    ge, size=ge.shape[-1] * 2, mode="nearest"
                )

            m_p, logs_p, y_mask = manager.vq_model.enc_p.infer(
                quantized,
                text,
                manager.vq_model.ge_to512(ge.transpose(2, 1)).transpose(2, 1)
                if getattr(manager.vq_model, "is_v2pro", False)
                else ge,
                speed,
                stream_mode,
                valid_start_idx,
                overlap_len,
                slice_indices,
            )

            if speed != 1 and ge.shape[-1] != 1:
                ge = F.interpolate(
                    ge, size=m_p.shape[-1], mode="nearest"
                )

            z_p = (
                m_p
                + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
            )

            # Use our CUDA Graph for the flow+decoder step
            o = manager.decode(z_p, y_mask, ge)

            try:
                attn = manager.vq_model.enc_p.mrte.cross_attention.attn
                return o, attn[0, ...]
            except AttributeError:
                return o

        self.vq_model.decode = patched_decode

    def unpatch(self):
        """Restore the original decode method."""
        if self._original_decode is not None:
            self.vq_model.decode = self._original_decode
            self._original_decode = None
