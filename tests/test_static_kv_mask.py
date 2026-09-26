import pytest


torch = pytest.importorskip("torch", reason="Torch runtime is optional in unit CI")


try:
    import aquatts  # noqa: F401 - configures the upstream overlay
    import AR.models.t2s_model as upstream_t2s
    import aquatts.modeling.t2s_flash_attn as flash_t2s
    import aquatts.modeling.t2s_streaming as patched_t2s
except ImportError as exc:  # external GPT-SoVITS runtime is optional in unit CI
    pytest.skip(f"GPT-SoVITS runtime unavailable: {exc}", allow_module_level=True)


def _blocks(hidden=4, heads=2):
    torch.manual_seed(17)
    mlp_w1 = torch.randn(hidden * 2, hidden)
    mlp_b1 = torch.randn(hidden * 2)
    mlp_w2 = torch.randn(hidden, hidden * 2)
    mlp_b2 = torch.randn(hidden)
    qkv_w = torch.randn(hidden * 3, hidden)
    qkv_b = torch.randn(hidden * 3)
    out_w = torch.randn(hidden, hidden)
    out_b = torch.randn(hidden)
    norm_w1 = torch.randn(hidden)
    norm_b1 = torch.randn(hidden)
    norm_w2 = torch.randn(hidden)
    norm_b2 = torch.randn(hidden)
    common = (
        heads,
        hidden,
        qkv_w,
        qkv_b,
        out_w,
        out_b,
        norm_w1,
        norm_b1,
        1e-5,
        norm_w2,
        norm_b2,
        1e-5,
    )
    dynamic = upstream_t2s.T2SBlock(
        common[0],
        common[1],
        upstream_t2s.T2SMLP(mlp_w1, mlp_b1, mlp_w2, mlp_b2),
        *common[2:],
    )
    static = patched_t2s.T2SBlockWithStaticCache(
        common[0],
        common[1],
        upstream_t2s.T2SMLP(mlp_w1, mlp_b1, mlp_w2, mlp_b2),
        *common[2:],
    )
    return dynamic, static


def test_static_mask_matches_dynamic_with_gap_and_future_garbage():
    dynamic, static = _blocks()
    torch.manual_seed(23)
    x = torch.randn(1, 1, 4)
    prefix_k = torch.randn(1, 3, 4)
    prefix_v = torch.randn(1, 3, 4)
    expected, _, _ = dynamic.decode_next_token(
        x.clone(), prefix_k.clone(), prefix_v.clone()
    )

    static_k = torch.randn(1, 8, 4) * 100
    static_v = torch.randn(1, 8, 4) * 100
    static_k[:, :3] = prefix_k
    static_v[:, :3] = prefix_v
    pos_idx = torch.full((1, 1, 4), 5, dtype=torch.long)
    valid = torch.zeros(1, 8, dtype=torch.bool)
    valid[:, :3] = True
    transformer = patched_t2s.T2STransformerWithStaticCache(1, [static])

    actual, _, _ = transformer.decode_next_token_with_static_cache(
        x.clone(), [static_k], [static_v], pos_idx, valid
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert valid.tolist() == [
        [True, True, True, False, False, True, False, False]
    ]


def test_non_graph_prefix_matches_dynamic_without_full_bucket_mask():
    dynamic, static = _blocks()
    torch.manual_seed(31)
    x = torch.randn(1, 1, 4)
    prefix_k = torch.randn(1, 3, 4)
    prefix_v = torch.randn(1, 3, 4)
    expected, _, _ = dynamic.decode_next_token(
        x.clone(), prefix_k.clone(), prefix_v.clone()
    )
    k_cache = torch.randn(1, 8, 4) * 100
    v_cache = torch.randn(1, 8, 4) * 100
    k_cache[:, :3] = prefix_k
    v_cache[:, :3] = prefix_v
    pos_idx = torch.full((1, 1, 4), 3, dtype=torch.long)
    valid = torch.zeros(1, 8, dtype=torch.bool)
    valid[:, :3] = True
    transformer = patched_t2s.T2STransformerWithStaticCache(1, [static])

    actual, _, _ = transformer.decode_next_token_with_static_cache(
        x.clone(), [k_cache], [v_cache], pos_idx, valid, 4
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_flash_bucket_mode_never_exposes_the_full_allocation(monkeypatch):
    _, static = _blocks()
    observed = {}

    def fake_flash(q, _k_cache, _v_cache, **kwargs):
        observed["cache_seqlens"] = kwargs["cache_seqlens"].detach().cpu()
        return torch.zeros_like(q)

    monkeypatch.setattr(flash_t2s, "_flash_attn_with_kvcache", fake_flash)
    flash = flash_t2s.T2SBlockWithStaticCacheFlash(
        2,
        4,
        static,
        static.qkv_w,
        static.qkv_b,
        static.out_w,
        static.out_b,
        static.norm_w1,
        static.norm_b1,
        static.norm_eps1,
        static.norm_w2,
        static.norm_b2,
        static.norm_eps2,
        mode="bucket",
    )
    transformer = flash_t2s.T2STransformerWithStaticCacheFlash(1, [flash])
    x = torch.randn(1, 1, 4)
    k_cache = torch.randn(1, 8, 4)
    v_cache = torch.randn(1, 8, 4)
    pos_idx = torch.full((1, 1, 4), 3, dtype=torch.long)
    valid = torch.zeros(1, 8, dtype=torch.bool)
    valid[:, :3] = True

    transformer.decode_next_token_with_static_cache(
        x, [k_cache], [v_cache], pos_idx, valid, 4
    )

    assert observed["cache_seqlens"].tolist() == [4]
    assert valid.tolist() == [[True, True, True, True, False, False, False, False]]
