"""Compatibility bridge to Aqua's canonical BigVGAN CUDA activation."""

from aquatts.bigvgan.cuda.activation1d import (  # noqa: F401
    Activation1d,
    FusedAntiAliasActivation,
)


__all__ = ["Activation1d", "FusedAntiAliasActivation"]
