"""
Evaluation for legal written-vs-applied gap detection.

Tests whether the model can identify DOCTRINAL gaps when given
paired statute text and enforcement reality.
"""
from __future__ import annotations


def build_eval_prompt(statute: str, enforcement: str) -> str:
    """Build an evaluation prompt pairing statute with enforcement data."""
    return (
        "You are an investigative analyst. Compare the following legal provision "
        "with the documented enforcement reality. Identify any DOCTRINAL gaps — "
        "places where the law as written diverges from the law as applied.\n\n"
        f"Law as written:\n{statute}\n\n"
        f"Law as applied:\n{enforcement}\n\n"
        "If a doctrinal gap exists, begin your response with 'DOCTRINAL GAP:' "
        "followed by your analysis. If no gap exists, explain why the enforcement "
        "is consistent with the statute."
    )


def parse_eval_response(response: str) -> dict[str, bool]:
    """Parse model response to determine if it detected a doctrinal gap."""
    text = response.lower()
    detected = (
        "doctrinal gap" in text
        or ("doctrinal" in text and "gap" in text)
        or ("diverge" in text and ("statute" in text or "enforcement" in text))
        or ("law as written" in text and "law as applied" in text)
    )
    return {"detected_doctrinal": detected}
