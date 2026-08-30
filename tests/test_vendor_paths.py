# -*- coding: utf-8 -*-
"""Tests for vendor path setup and package structure."""
import os
import sys
import importlib
from pathlib import Path

import pytest

import aquatts


class TestVendorPathSetup:
    def test_vendor_dir_in_sys_path(self):
        """_vendor/ is added to sys.path after import aquatts."""
        _vendor = os.path.join(os.path.dirname(aquatts.__file__), "_vendor")
        assert os.path.isdir(_vendor), f"_vendor/ not found at {_vendor}"
        assert _vendor in sys.path, "_vendor/ not in sys.path"

    def test_vendor_precedes_main(self):
        """_vendor/ should be before any GPT_SoVITS main repo path on sys.path."""
        _vendor = os.path.join(os.path.dirname(aquatts.__file__), "_vendor")
        _vendor_gpt_sovits = os.path.join(_vendor, "GPT_SoVITS")
        vendor_idx = sys.path.index(_vendor)
        vendor_gpt_sovits_idx = sys.path.index(_vendor_gpt_sovits)
        for p in sys.path:
            if (
                "GPT_SoVITS" in p
                and not p.startswith(_vendor)
                and "pretrained_models" not in p
            ):
                main_idx = sys.path.index(p)
                assert vendor_idx < main_idx, (
                    f"_vendor/ (idx {vendor_idx}) must precede {p} (idx {main_idx})"
                )
                assert vendor_gpt_sovits_idx < main_idx, (
                    f"_vendor/GPT_SoVITS (idx {vendor_gpt_sovits_idx}) must precede "
                    f"{p} (idx {main_idx})"
                )

    def test_t2s_model_from_upstream(self):
        """Aqua patches the selected upstream model instead of vendoring it."""
        if not os.environ.get("GPT_SOVITS_HOME"):
            pytest.skip("upstream checkout is not configured")
        import GPT_SoVITS.AR.models.t2s_model as m

        assert Path(m.__file__).resolve() == aquatts.upstream_t2s_model_path()
        assert "_vendor" not in m.__file__
        assert hasattr(m.Text2SemanticDecoder, "infer_panel_naive"), (
            "upstream t2s_model missing infer_panel_naive"
        )

    def test_top_level_ar_model_from_upstream(self):
        """Top-level AR imports resolve to the same selected upstream source."""
        if not os.environ.get("GPT_SOVITS_HOME"):
            pytest.skip("upstream checkout is not configured")
        import AR.models.t2s_model as m

        assert Path(m.__file__).resolve() == aquatts.upstream_t2s_model_path()

    def test_no_vendored_t2s_model_copy(self):
        vendor_copy = (
            Path(aquatts.__file__).parent
            / "_vendor/GPT_SoVITS/AR/models/t2s_model.py"
        )
        assert not vendor_copy.exists()

    def test_gpt_sovits_home_optional_for_submodules(self):
        """Submodules (params, streaming) import without GPT_SOVITS_HOME."""
        from aquatts.inference.params import get_sovits_params
        from aquatts.inference.streaming import (
            apply_fade_in,
            apply_fade_out,
            finalize_stream_chunk,
        )
        params = get_sovits_params("test", is_first_sentence=True)
        assert isinstance(params, dict)
        assert "speed" in params

    def test_tts_inferencer_import_needs_gpt_sovits_home(self):
        """TTSInferencer import requires GPT_SOVITS_HOME to be set."""
        if os.environ.get("GPT_SOVITS_HOME"):
            pytest.skip("covered by the configured-upstream integration test")
        with pytest.raises(ImportError):
            from aquatts import TTSInferencer

    def test_package_version(self):
        """aquatts.__version__ is set."""
        assert aquatts.__version__ == "0.2.1"

    def test_import_does_not_change_working_directory(self):
        """Reloading the package must not change the caller's cwd."""
        original_cwd = os.getcwd()
        importlib.reload(aquatts)
        assert os.getcwd() == original_cwd

    def test_all_exports(self):
        """__all__ lists expected public API."""
        assert "__version__" in aquatts.__all__
        assert "TTSInferencer" in aquatts.__all__
        assert "configure_gpt_sovits" in aquatts.__all__
