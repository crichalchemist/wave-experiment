"""Tests for VaryBalance annotation consistency scorer."""
from src.core.providers import MockProvider


def test_identical_rewrites_score_zero():
    """Identical rewrite lengths produce MSD of 0."""
    from src.data.annotation_filter import annotation_consistency_score
    # MockProvider returns same response for both calls
    provider = MockProvider(response="x" * 100)
    score = annotation_consistency_score("Analysis A.", "Analysis B.", provider)
    assert score == 0.0


def test_different_rewrite_lengths_score_nonzero():
    """Different rewrite lengths produce non-zero MSD."""
    from src.data.annotation_filter import annotation_consistency_score

    responses = iter(["short", "a much longer rewrite response here"])

    class _VaryingProvider:
        def complete(self, prompt, **_): return next(responses)
        def embed(self, text): return []

    score = annotation_consistency_score("A.", "B.", _VaryingProvider())
    assert score > 0.0


def test_score_is_squared_length_difference():
    """Verify the MSD formula: (len_a - len_b)^2."""
    from src.data.annotation_filter import annotation_consistency_score

    rewrite_a = "x" * 10
    rewrite_b = "y" * 15
    responses = iter([rewrite_a, rewrite_b])

    class _DeterministicProvider:
        def complete(self, prompt, **_): return next(responses)
        def embed(self, text): return []

    score = annotation_consistency_score("A.", "B.", _DeterministicProvider())
    expected = float((10 - 15) ** 2)
    assert score == expected


def test_score_is_nonnegative():
    """MSD is always non-negative."""
    from src.data.annotation_filter import annotation_consistency_score

    responses = iter(["longer rewrite here", "short"])

    class _Provider:
        def complete(self, prompt, **_): return next(responses)
        def embed(self, text): return []

    score = annotation_consistency_score("A.", "B.", _Provider())
    assert score >= 0.0


def test_is_consistent_below_threshold():
    from src.data.annotation_filter import is_consistent
    provider = MockProvider(response="same")  # identical length → MSD=0
    assert is_consistent("A.", "B.", provider, threshold=0.1)


def test_is_consistent_above_threshold():
    from src.data.annotation_filter import is_consistent

    responses = iter(["x" * 100, "y" * 200])

    class _Provider:
        def complete(self, prompt, **_): return next(responses)
        def embed(self, text): return []

    assert not is_consistent("A.", "B.", _Provider(), threshold=0.1)


def test_consistency_threshold_constant():
    from src.data.annotation_filter import CONSISTENCY_THRESHOLD
    assert CONSISTENCY_THRESHOLD == 0.1


def test_rewrite_prompt_includes_analysis():
    """The rewrite prompt sent to provider must include the original analysis text."""
    from src.data.annotation_filter import annotation_consistency_score
    prompts_seen = []

    class _CapturingProvider:
        def complete(self, prompt, **_):
            prompts_seen.append(prompt)
            return "rewrite"
        def embed(self, text): return []

    annotation_consistency_score("UNIQUE_MARKER_TEXT", "other", _CapturingProvider())
    assert any("UNIQUE_MARKER_TEXT" in p for p in prompts_seen)
