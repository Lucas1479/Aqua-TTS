from aquatts.bigvgan.cuda.load import load

__all__ = ["Activation1d", "FusedAntiAliasActivation", "load"]


def __getattr__(name):
    if name in {"Activation1d", "FusedAntiAliasActivation"}:
        try:
            from aquatts.bigvgan.cuda.activation1d import (
                Activation1d,
                FusedAntiAliasActivation,
            )
        except ImportError:
            return None
        return {
            "Activation1d": Activation1d,
            "FusedAntiAliasActivation": FusedAntiAliasActivation,
        }[name]
    raise AttributeError(name)
