from aquatts.inference.params import get_sovits_params


class TestGetSovitsParams:
    def test_first_sentence(self):
        p = get_sovits_params("hello", is_first_sentence=True)
        assert p["speed"] == 1.1
        assert p["enable_static_kv"] is True
        assert "max_sec_override" in p

    def test_short_text(self):
        p = get_sovits_params("short text", is_first_sentence=False)
        assert p["speed"] == 1
        assert p["sample_steps"] == 16
        assert p["enable_static_kv"] is True

    def test_long_text(self):
        p = get_sovits_params("a" * 100, is_first_sentence=False)
        assert p["sample_steps"] == 32
        assert p["speed"] == 1
        assert p["enable_static_kv"] is True

    def test_empty_text(self):
        p = get_sovits_params("   ", is_first_sentence=False)
        # Treated as short text (length 0 < 45)
        assert isinstance(p, dict)
        assert "top_k" in p
