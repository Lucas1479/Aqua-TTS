from aquatts.inference.semantic_stability import assess_semantic_candidate


def test_accepts_long_diverse_candidate_without_a_hard_length_cap():
    result = assess_semantic_candidate(
        [index % 37 for index in range(180)],
        target_phone_count=20,
        max_generation_tokens=400,
    )

    assert result.accepted is True
    assert result.soft_token_budget == 96


def test_rejects_equal_token_collapse_without_token_id_rules():
    result = assess_semantic_candidate(
        list(range(24)) + [937] * 48,
        target_phone_count=48,
        max_generation_tokens=400,
    )

    assert result.accepted is False
    assert "equal_token_collapse" in result.reasons


def test_rejects_periodic_collapse():
    result = assess_semantic_candidate(
        [11, 29, 47, 53] * 18,
        target_phone_count=6,
        max_generation_tokens=400,
    )

    assert result.accepted is False
    assert "periodic_token_collapse" in result.reasons
    assert result.periodic_run_period == 4


def test_rejects_over_budget_filler_only_with_repetition_evidence():
    result = assess_semantic_candidate(
        list(range(84)) + [311] * 20,
        target_phone_count=6,
        max_generation_tokens=400,
    )

    assert result.accepted is False
    assert result.reasons == ("length_with_repetition",)


def test_rejects_empty_candidate():
    result = assess_semantic_candidate(
        [],
        target_phone_count=8,
        max_generation_tokens=400,
    )

    assert result.accepted is False
    assert result.reasons == ("empty_candidate",)
