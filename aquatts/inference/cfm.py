"""Opt-in, instance-scoped mask preparation for upstream CFM/DiT.

The diffusion loop, conditioner caches, CFG and training branch stay upstream.
Only DiT's single mask assignment is adapted in memory, without changing any
external checkout file, class or existing function globals.
"""
from __future__ import annotations

import ast
import __future__
from contextvars import ContextVar
import inspect
import textwrap
from types import MethodType

from aquatts.upstream import GPTSoVITSConfigurationError


class UnpaddedCFM:
    def __init__(self, cfm):
        self.cfm = cfm
        self.last_stats = None
        self._prepared = ContextVar("aqua_cfm_prepared_mask", default=None)
        estimator = cfm.estimator
        original = estimator.forward.__func__
        original = getattr(original, "_aqua_original", original)
        required = {"infer", "text_cache", "dt_cache", "x_lens"}
        if not required.issubset(inspect.signature(original).parameters):
            raise GPTSoVITSConfigurationError("CFM experiment requires the cached upstream DiT interface")
        if not {"text_cfg_cache", "dt_cache", "text_cache"}.issubset(
                inspect.unwrap(cfm.inference).__code__.co_varnames):
            raise GPTSoVITSConfigurationError("CFM experiment requires the upstream conditioner-cache loop")
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
        except (OSError, TypeError, SyntaxError) as error:
            raise GPTSoVITSConfigurationError("CFM experiment requires inspectable DiT source") from error
        expected = ast.parse("mask = sequence_mask(x_lens, max_length=x.size(1)).to(x.device)").body[0]
        assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                       and ast.dump(node) == ast.dump(expected)]
        references = [node for node in ast.walk(tree) if isinstance(node, ast.Name) and node.id == "sequence_mask"]
        if len(assignments) != 1 or len(references) != 1:
            raise GPTSoVITSConfigurationError("Unsupported DiT mask construction; disable cfm_unpadded")
        sequence_mask = original.__globals__.get("sequence_mask")
        if not callable(sequence_mask):
            raise GPTSoVITSConfigurationError("DiT sequence_mask is unavailable")
        self._sequence_mask = sequence_mask

        def prepared_mask(lengths, x, infer):
            prepared = self._prepared.get() if infer else None
            if (prepared is not None and lengths is prepared[0]
                    and tuple(x.shape[:2]) == prepared[1]):
                return prepared[2]
            return sequence_mask(lengths, max_length=x.size(1)).to(x.device)

        assignments[0].value = ast.parse("_aqua_prepared_mask(x_lens, x, infer)", mode="eval").body
        ast.fix_missing_locations(tree)
        namespace = dict(original.__globals__, _aqua_prepared_mask=prepared_mask)
        exec(compile(tree, inspect.getfile(original), "exec", flags=__future__.annotations.compiler_flag), namespace)
        adapted = namespace[original.__name__]
        adapted._aqua_original = original
        estimator.forward = MethodType(adapted, estimator)

    def __call__(self, mu, x_lens, prompt, n_timesteps, *, unpadded_length=None,
                 temperature=1.0, inference_cfg_rate=0):
        """Use Aqua-owned host length metadata; unknown/batched/padded inputs keep a mask."""
        eligible = (type(unpadded_length) is int and mu.size(0) == 1
                    and tuple(x_lens.shape) == (1,) and unpadded_length == mu.size(1))
        mask = None if eligible else self._sequence_mask(x_lens, max_length=mu.size(1)).to(mu.device)
        token = self._prepared.set((x_lens, tuple(mu.shape[:2]), mask))
        try:
            result = self.cfm.inference(mu, x_lens, prompt, n_timesteps,
                                        temperature=temperature, inference_cfg_rate=inference_cfg_rate)
            self.last_stats = {"path": "unpadded" if eligible else "masked",
                               "mask_builds": 0 if eligible else 1,
                               "steps": n_timesteps, "cfg_rate": inference_cfg_rate}
            return result
        finally:
            self._prepared.reset(token)
