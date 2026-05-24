# -*- coding: utf-8 -*-
"""Tests for vendor path setup and package structure."""
import os
import sys

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
        vendor_idx = sys.path.index(_vendor)
        for p in sys.path:
            if "GPT_SoVITS" in p and p != _vendor and "pretrained_models" not in p:
                main_idx = sys.path.index(p)
                assert vendor_idx < main_idx, (
                    f"_vendor/ (idx {vendor_idx}) must precede {p} (idx {main_idx})"
                )

    def test_t2s_model_from_vendor(self):
        """Vendored t2s_model.py is loaded instead of main repo version."""
        import GPT_SoVITS.AR.models.t2s_model as m
        assert "_vendor" in m.__file__, f"Wrong t2s_model source: {m.__file__}"
        # Verify vendored features exist
        assert hasattr(m.Text2SemanticDecoder, "infer_panel_naive"), (
            "vendored t2s_model missing infer_panel_naive"
        )

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
            from aquatts import TTSInferencer
            assert TTSInferencer.__module__ == "aquatts.inferencer"
        else:
            # Without GPT_SOVITS_HOME, importing TTSInferencer should fail
            import pytest
            with pytest.raises(ImportError):
                from aquatts import TTSInferencer

    def test_package_version(self):
        """aquatts.__version__ is set."""
        assert aquatts.__version__ == "1.0.0"

    def test_all_exports(self):
        """__all__ lists expected public API."""
        assert "__version__" in aquatts.__all__
        assert "TTSInferencer" in aquatts.__all__
