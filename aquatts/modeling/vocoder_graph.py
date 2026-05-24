"""
Bucketed CUDA Graph wrapper for the SoVITS/BigVGAN vocoder forward pass.

v2 (SynthesizerTrn): Captures flow + decoder inside CUDA Graphs.
v3 (SynthesizerTrnV3): Captures BigVGAN.forward() inside CUDA Graphs.

Usage:
    from aquatts.modeling.vocoder_graph import VocoderGraphManager

    # After loading the vocoder model (and BigVGAN for v3):
    vgm = VocoderGraphManager(vq_model, bigvgan_model=bigvgan_model)
    vgm.precapture()

    # During inference, decode() or decode_bigvgan() uses graph replay.
"""

import threading
import time
from typing import Dict, List, Optional

import torch
from torch.nn import functional as F


_DEFAULT_VOCODER_BUCKETS = [70, 128, 256, 512, 768, 1024]


class _VocoderBucket:
    """Holds pre-allocated buffers and CUDA Graph for one mel-T size bucket."""

    __slots__ = (
        "mel_t",
        "z_p_padded", "y_mask_padded", "ge_padded",
        "mel_padded",
        "dec_o", "cuda_graph", "lock",
    )

    def __init__(self):
        self.mel_t: int = 0
        self.z_p_padded: Optional[torch.Tensor] = None
        self.y_mask_padded: Optional[torch.Tensor] = None
        self.ge_padded: Optional[torch.Tensor] = None
        self.mel_padded: Optional[torch.Tensor] = None
        self.dec_o: Optional[torch.Tensor] = None
        self.cuda_graph: Optional[torch.cuda.CUDAGraph] = None
        self.lock = threading.Lock()


class VocoderGraphManager:
    """Manager for bucketed CUDA Graph capture on a SoVITS/BigVGAN vocoder."""

    def __init__(
        self,
        vq_model: torch.nn.Module,
        bigvgan_model: Optional[torch.nn.Module] = None,
        bucket_sizes: Optional[List[int]] = None,
        warmup_steps: int = 3,
    ):
        self.vq_model = vq_model
        self.bigvgan_model = bigvgan_model
        self.bucket_sizes = bucket_sizes or _DEFAULT_VOCODER_BUCKETS
        self.warmup_steps = warmup_steps

        self.buckets: Dict[int, _VocoderBucket] = {}
        self._captured = False
        self._is_v3 = bigvgan_model is not None

        self._original_decode = None
        self._original_bigvgan_forward = None

    @property
    def is_v3(self) -> bool:
        return self._is_v3

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def _get_device_and_dtype(self, model):
        """Get device and dtype from model parameters."""
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        return device, dtype

    def _capture_graph_context(self, device):
        """Create a CUDA Graph capture context with relaxed error mode if available."""
        import inspect as _inspect
        try:
            _has_cem = (
                "capture_error_mode"
                in _inspect.signature(torch.cuda.CUDAGraph.__init__).parameters
            )
        except (ValueError, TypeError):
            _has_cem = False

        _graph_extra = {"capture_error_mode": "relaxed"} if _has_cem else {}
        bucket_graph = torch.cuda.CUDAGraph()
        ctx = torch.cuda.graph(bucket_graph, **_graph_extra)
        return bucket_graph, ctx

    def _precapture_v2(self, device):
        """Capture v2 flow + decoder CUDA Graphs."""
        if not hasattr(self.vq_model, "flow"):
            print("[VocoderGraph] vq_model has no 'flow' — likely v3, but no bigvgan_model provided")
            return 0

        try:
            latent_channels = self.vq_model.enc_p.latent_channels
        except AttributeError:
            latent_channels = self.vq_model.generator.latent_channels

        try:
            gin_channels = self.vq_model.gin_channels
        except AttributeError:
            gin_channels = self.vq_model.gen_channels

        batch_size = 1
        dtype = next(self.vq_model.parameters()).dtype
        success_count = 0

        for mel_t in sorted(self.bucket_sizes):
            bucket = _VocoderBucket()
            bucket.mel_t = mel_t

            try:
                bucket.z_p_padded = torch.zeros(
                    batch_size, latent_channels, mel_t, dtype=dtype, device=device
                )
                bucket.y_mask_padded = torch.zeros(
                    batch_size, 1, mel_t, dtype=dtype, device=device
                )
                bucket.ge_padded = torch.zeros(
                    batch_size, gin_channels, 1, dtype=dtype, device=device
                )
                bucket.z_p_padded.normal_()
                bucket.y_mask_padded.fill_(1.0)
                bucket.ge_padded.normal_()

                for _ in range(self.warmup_steps):
                    z = self.vq_model.flow(bucket.z_p_padded, bucket.y_mask_padded, bucket.ge_padded)
                    self.vq_model.dec(z * bucket.y_mask_padded, g=bucket.ge_padded)

                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    bucket_graph, ctx = self._capture_graph_context(device)
                    with torch.cuda.device(device):
                        bucket.cuda_graph = bucket_graph
                        with ctx:
                            z = self.vq_model.flow(
                                bucket.z_p_padded, bucket.y_mask_padded, bucket.ge_padded
                            )
                            bucket.dec_o = self.vq_model.dec(z * bucket.y_mask_padded, g=bucket.ge_padded)

                self.buckets[mel_t] = bucket
                success_count += 1
                print(f"[VocoderGraph] v2 bucket mel_T={mel_t} captured")

            except Exception as e:
                print(f"[VocoderGraph] Failed v2 bucket mel_T={mel_t}: {e}")
                import traceback
                traceback.print_exc()

        return success_count

    def _precapture_v3(self, device):
        """Capture v3 BigVGAN.forward() CUDA Graphs.

        BigVGAN forward takes mel input [B, 100, mel_T] → audio [B, 1, mel_T * 256].
        """
        if self.bigvgan_model is None:
            print("[VocoderGraph] No BigVGAN model for v3, skipping capture")
            return 0

        model = self.bigvgan_model
        try:
            num_mels = model.h.num_mels
        except AttributeError:
            num_mels = 100

        batch_size = 1
        dtype = next(model.parameters()).dtype
        success_count = 0

        for mel_t in sorted(self.bucket_sizes):
            bucket = _VocoderBucket()
            bucket.mel_t = mel_t

            try:
                bucket.mel_padded = torch.zeros(
                    batch_size, num_mels, mel_t, dtype=dtype, device=device
                )
                bucket.mel_padded.normal_(0, 0.5)

                for _ in range(self.warmup_steps):
                    _ = model(bucket.mel_padded)

                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    bucket_graph, ctx = self._capture_graph_context(device)
                    with torch.cuda.device(device):
                        bucket.cuda_graph = bucket_graph
                        with ctx:
                            bucket.dec_o = model(bucket.mel_padded)

                self.buckets[mel_t] = bucket
                success_count += 1
                print(f"[VocoderGraph] v3 BigVGAN bucket mel_T={mel_t} captured")

            except Exception as e:
                print(f"[VocoderGraph] Failed v3 bucket mel_T={mel_t}: {e}")
                import traceback
                traceback.print_exc()

        return success_count

    def precapture(self, device: Optional[torch.device] = None) -> bool:
        """Pre-capture CUDA Graphs for all bucket sizes.

        Returns True if any buckets captured successfully.
        """
        if device is None:
            model = self.bigvgan_model if self._is_v3 else self.vq_model
            device = next(model.parameters()).device
        if device.type != "cuda":
            print("[VocoderGraph] Not CUDA device, skipping capture")
            return False

        if self._is_v3:
            success_count = self._precapture_v3(device)
        else:
            success_count = self._precapture_v2(device)

        self._captured = success_count > 0
        print(f"[VocoderGraph] Captured {success_count}/{len(self.bucket_sizes)} vocoder buckets")
        return self._captured

    # ------------------------------------------------------------------
    # Graph replay
    # ------------------------------------------------------------------

    def decode(self, z_p, y_mask, ge, samples_per_frame=2):
        """v2 path: decode z_p → audio using CUDA Graph replay."""
        if not self._captured:
            z = self.vq_model.flow(z_p, y_mask, ge)
            return self.vq_model.dec(z * y_mask, g=ge)

        z_current_length = z_p.size(-1)
        matched = self._find_bucket(z_current_length)
        if matched is None:
            z = self.vq_model.flow(z_p, y_mask, ge)
            return self.vq_model.dec(z * y_mask, g=ge)

        bucket = self.buckets[matched]
        with bucket.lock:
            bucket.z_p_padded.zero_()
            bucket.y_mask_padded.zero_()
            bucket.z_p_padded[:, :, :z_current_length].copy_(z_p)
            bucket.y_mask_padded[:, :, :z_current_length].copy_(y_mask)
            bucket.ge_padded.copy_(ge)

            bucket.cuda_graph.replay()
            torch.cuda.synchronize(z_p.device)

            audio_len = z_current_length * samples_per_frame
            return bucket.dec_o[:, :, :audio_len]

    def decode_bigvgan(self, mel: torch.Tensor) -> torch.Tensor:
        """v3 path: BigVGAN forward → audio, using CUDA Graph replay when bucket fits.

        Args:
            mel: [B, num_mels, mel_T]  mel spectrogram features

        Returns:
            audio: [B, 1, mel_T * 256]
        """
        if not self._captured:
            return self.bigvgan_model(mel)

        mel_t = mel.size(-1)
        matched = self._find_bucket(mel_t)
        if matched is None:
            return self.bigvgan_model(mel)

        bucket = self.buckets[matched]
        with bucket.lock:
            bucket.mel_padded.zero_()
            bucket.mel_padded[:, :, :mel_t].copy_(mel)

            bucket.cuda_graph.replay()
            torch.cuda.synchronize(mel.device)

            audio_len = mel_t * 256
            return bucket.dec_o[:, :, :audio_len]

    def _find_bucket(self, current_len: int) -> Optional[int]:
        """Find smallest bucket >= current length, or None."""
        for mel_t in sorted(self.bucket_sizes):
            if mel_t >= current_len:
                return mel_t
        return None

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------

    def patch(self):
        """Patch the vocoder path.

        For v2: replaces vq_model.decode with graph replay.
        For v3: replaces bigvgan_model.forward with graph replay.
        """
        if self._is_v3:
            self._patch_bigvgan()
        else:
            self._patch_v2_decode()

    def _patch_v2_decode(self):
        """Replace vq_model.decode with graph-replay path."""
        if self._original_decode is None:
            self._original_decode = self.vq_model.decode

        manager = self

        def patched_decode(codes, text, ge, noise_scale=0.5, speed=1,
                          cuda_graph=True, stream_mode=False,
                          valid_start_idx=None, overlap_len=None,
                          slice_indices=None):
            if not cuda_graph or not manager._captured:
                return manager._original_decode(
                    codes, text, ge, noise_scale, speed,
                    stream_mode=stream_mode,
                    valid_start_idx=valid_start_idx,
                    overlap_len=overlap_len,
                    slice_indices=slice_indices,
                )

            quantized = manager.vq_model.quantizer.decode(codes)
            quantized = F.interpolate(quantized, size=quantized.shape[-1] * 2, mode="nearest")
            if ge.shape[-1] != 1:
                ge = F.interpolate(ge, size=ge.shape[-1] * 2, mode="nearest")

            m_p, logs_p, y_mask = manager.vq_model.enc_p.infer(
                quantized, text,
                manager.vq_model.ge_to512(ge.transpose(2, 1)).transpose(2, 1)
                if getattr(manager.vq_model, "is_v2pro", False) else ge,
                speed, stream_mode, valid_start_idx, overlap_len, slice_indices,
            )

            if speed != 1 and ge.shape[-1] != 1:
                ge = F.interpolate(ge, size=m_p.shape[-1], mode="nearest")

            z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
            o = manager.decode(z_p, y_mask, ge)

            try:
                attn = manager.vq_model.enc_p.mrte.cross_attention.attn
                return o, attn[0, ...]
            except AttributeError:
                return o

        self.vq_model.decode = patched_decode

    def _patch_bigvgan(self):
        """Replace bigvgan_model.forward with CUDA Graph-backed version."""
        if self._original_bigvgan_forward is None:
            self._original_bigvgan_forward = self.bigvgan_model.forward

        manager = self

        def graph_forward(x):
            if not manager._captured:
                return manager._original_bigvgan_forward(x)
            return manager.decode_bigvgan(x)

        self.bigvgan_model.forward = graph_forward

    def unpatch(self):
        """Restore original methods."""
        if self._original_decode is not None:
            self.vq_model.decode = self._original_decode
            self._original_decode = None
        if self._original_bigvgan_forward is not None:
            self.bigvgan_model.forward = self._original_bigvgan_forward
            self._original_bigvgan_forward = None
