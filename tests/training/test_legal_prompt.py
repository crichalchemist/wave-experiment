def test_legal_analysis_prompt_contains_doctrinal_framing():
    from src.training.constitutional_warmup import build_legal_analysis_prompt
    prompt = build_legal_analysis_prompt("Some statute text here")
    assert "law as written" in prompt.lower()
    assert "law as applied" in prompt.lower()
    assert "doctrinal" in prompt.lower()
    assert "Some statute text here" in prompt


def test_legal_analysis_prompt_names_gap_types():
    from src.training.constitutional_warmup import build_legal_analysis_prompt
    prompt = build_legal_analysis_prompt("text")
    assert "enforcement" in prompt.lower()
    assert "statute" in prompt.lower()


def test_legal_analysis_prompt_includes_domain_tag():
    from src.training.constitutional_warmup import build_legal_analysis_prompt
    prompt = build_legal_analysis_prompt("text", domain="criminal_justice")
    assert "criminal_justice" in prompt or "criminal justice" in prompt.lower()


def test_original_analysis_prompt_still_works():
    """Regression: generic prompt unchanged."""
    from src.training.constitutional_warmup import _ANALYSIS_PROMPT
    prompt = _ANALYSIS_PROMPT.format(text="test doc")
    assert "information gaps" in prompt.lower()