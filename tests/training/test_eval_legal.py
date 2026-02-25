def test_eval_legal_prompt_checks_doctrinal_detection():
    from src.training.eval_legal import build_eval_prompt
    prompt = build_eval_prompt(
        statute="18 U.S.C. § 3553(a) requires individualized sentencing",
        enforcement="Mandatory minimums applied uniformly regardless of circumstances",
    )
    assert "statute" in prompt.lower() or "18 U.S.C." in prompt
    assert "enforcement" in prompt.lower() or "mandatory" in prompt.lower()


def test_parse_eval_response_detects_doctrinal():
    from src.training.eval_legal import parse_eval_response
    response = "DOCTRINAL GAP: The statute requires individualized sentencing but mandatory minimums are applied uniformly."
    result = parse_eval_response(response)
    assert result["detected_doctrinal"] is True


def test_parse_eval_response_no_gap():
    from src.training.eval_legal import parse_eval_response
    response = "The enforcement pattern is consistent with the statutory text."
    result = parse_eval_response(response)
    assert result["detected_doctrinal"] is False
