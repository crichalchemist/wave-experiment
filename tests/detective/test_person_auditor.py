"""Tests for person auditor — claim decomposition and verification."""

from __future__ import annotations

import pytest

from src.core.providers import MockProvider
from src.detective.person_auditor import (
    AtomicClaim,
    PersonAudit,
    VerificationResult,
    audit_person,
    compute_severity,
    decompose_claims,
    verify_claim,
    _classify_claim_type,
)


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


class TestAtomicClaim:
    def test_frozen(self):
        c = AtomicClaim(
            text="Met with Person B in 2010",
            source_text="original paragraph",
            claim_type="temporal",
            person="Person A",
        )
        with pytest.raises(AttributeError):
            c.text = "mutated"  # type: ignore[misc]

    def test_fields(self):
        c = AtomicClaim(
            text="Transferred $50,000",
            source_text="source",
            claim_type="financial",
            person="Person A",
        )
        assert c.text == "Transferred $50,000"
        assert c.claim_type == "financial"
        assert c.person == "Person A"


class TestVerificationResult:
    def test_frozen(self):
        v = VerificationResult(
            claim=AtomicClaim(
                text="claim", source_text="src", claim_type="other", person="X"
            ),
            status="supported",
            confidence=0.9,
            supporting_evidence=("doc1",),
            contradicting_evidence=(),
        )
        with pytest.raises(AttributeError):
            v.status = "contradicted"  # type: ignore[misc]

    def test_fields(self):
        claim = AtomicClaim(
            text="Was in New York",
            source_text="src",
            claim_type="location",
            person="X",
        )
        v = VerificationResult(
            claim=claim,
            status="contradicted",
            confidence=0.8,
            supporting_evidence=(),
            contradicting_evidence=("doc2",),
        )
        assert v.status == "contradicted"
        assert v.confidence == 0.8
        assert v.contradicting_evidence == ("doc2",)

    def test_confidence_bounds(self):
        claim = AtomicClaim(
            text="c", source_text="s", claim_type="other", person="X"
        )
        with pytest.raises(ValueError, match="confidence must be in"):
            VerificationResult(
                claim=claim,
                status="supported",
                confidence=1.5,
                supporting_evidence=(),
                contradicting_evidence=(),
            )


class TestPersonAudit:
    def test_frozen(self):
        a = PersonAudit(
            person="Jeffrey Epstein",
            claims=(),
            verifications=(),
            overall_confidence=0.5,
            contradiction_count=0,
            severity_score=0.0,
        )
        with pytest.raises(AttributeError):
            a.person = "mutated"  # type: ignore[misc]

    def test_supported_count(self):
        claim = AtomicClaim(
            text="c", source_text="s", claim_type="other", person="X"
        )
        v1 = VerificationResult(
            claim=claim, status="supported", confidence=0.9,
            supporting_evidence=("d",), contradicting_evidence=(),
        )
        v2 = VerificationResult(
            claim=claim, status="contradicted", confidence=0.8,
            supporting_evidence=(), contradicting_evidence=("d",),
        )
        v3 = VerificationResult(
            claim=claim, status="unverified", confidence=0.3,
            supporting_evidence=(), contradicting_evidence=(),
        )
        a = PersonAudit(
            person="X",
            claims=(claim, claim, claim),
            verifications=(v1, v2, v3),
            overall_confidence=0.6,
            contradiction_count=1,
            severity_score=0.3,
        )
        assert a.supported_count == 1
        assert a.contradicted_count == 1
        assert a.unverified_count == 1


# ---------------------------------------------------------------------------
# _classify_claim_type
# ---------------------------------------------------------------------------


class TestClassifyClaimType:
    def test_temporal(self):
        assert _classify_claim_type("Met in January 2010") == "temporal"
        assert _classify_claim_type("Between 1995 and 2003") == "temporal"

    def test_financial(self):
        assert _classify_claim_type("Transferred $50,000 to the account") == "financial"
        assert _classify_claim_type("Received funding from the foundation") == "financial"

    def test_location(self):
        assert _classify_claim_type("Traveled to New York") == "location"
        assert _classify_claim_type("Visited the island") == "location"

    def test_association(self):
        assert _classify_claim_type("Was associated with Person B") == "association"
        assert _classify_claim_type("Met with the director") == "association"

    def test_role(self):
        assert _classify_claim_type("Served as director of the foundation") == "role"
        assert _classify_claim_type("Employed as a financial advisor") == "role"

    def test_other(self):
        assert _classify_claim_type("Something happened") == "other"


# ---------------------------------------------------------------------------
# decompose_claims
# ---------------------------------------------------------------------------


class TestDecomposeClaims:
    def test_fallback_sentence_splitting(self):
        """Without a provider, falls back to sentence splitting."""
        text = "Person A met with Person B. Person A traveled to Paris."
        claims = decompose_claims("Person A", text)
        assert len(claims) >= 2
        assert all(isinstance(c, AtomicClaim) for c in claims)
        assert all(c.person == "Person A" for c in claims)

    def test_empty_text(self):
        claims = decompose_claims("Person A", "")
        assert claims == ()

    def test_llm_decomposition(self):
        """With a provider that returns 'claim:' lines, parses them."""
        provider = MockProvider(
            response="claim: Met with Person B in 2010\nclaim: Transferred funds to account"
        )
        claims = decompose_claims("Person A", "Some compound text", provider=provider)
        assert len(claims) == 2
        assert "Met with Person B" in claims[0].text
        assert "Transferred funds" in claims[1].text

    def test_llm_fallback_on_empty_response(self):
        """If LLM returns no 'claim:' lines, falls back to sentence splitting."""
        provider = MockProvider(response="I cannot decompose this text.")
        text = "Person A met Person B. Person A went to London."
        claims = decompose_claims("Person A", text, provider=provider)
        assert len(claims) >= 2  # fell back to sentence splitting


# ---------------------------------------------------------------------------
# verify_claim
# ---------------------------------------------------------------------------


class TestVerifyClaim:
    def test_fallback_keyword_overlap(self):
        """Without a provider, uses keyword overlap heuristic."""
        claim = AtomicClaim(
            text="Met with Person B in New York",
            source_text="src",
            claim_type="association",
            person="Person A",
        )
        result = verify_claim(
            claim,
            evidence_texts=["Person A met Person B in New York at the hotel"],
        )
        assert isinstance(result, VerificationResult)
        assert result.status in ("supported", "partially_supported", "unverified")

    def test_no_evidence(self):
        claim = AtomicClaim(
            text="Made a payment",
            source_text="src",
            claim_type="financial",
            person="Person A",
        )
        result = verify_claim(claim, evidence_texts=[])
        assert result.status == "unverified"
        assert result.confidence == 0.0

    def test_llm_verification(self):
        """With a provider, parses status and confidence from LLM response."""
        claim = AtomicClaim(
            text="Met Person B",
            source_text="src",
            claim_type="association",
            person="Person A",
        )
        provider = MockProvider(response="status: supported\nconfidence: 0.85")
        result = verify_claim(claim, evidence_texts=["Some evidence"], provider=provider)
        assert result.status == "supported"
        assert result.confidence == 0.85


# ---------------------------------------------------------------------------
# compute_severity
# ---------------------------------------------------------------------------


class TestComputeSeverity:
    def test_no_contradictions(self):
        score = compute_severity(
            verification_count=5, contradiction_count=0
        )
        assert 0.0 <= score <= 1.0
        assert score < 0.5  # no contradictions → low severity

    def test_all_contradictions(self):
        score = compute_severity(
            verification_count=5, contradiction_count=5
        )
        assert score > 0.3  # significant severity

    def test_with_threatened_constructs(self):
        low = compute_severity(
            verification_count=5, contradiction_count=2
        )
        high = compute_severity(
            verification_count=5,
            contradiction_count=2,
            threatened_constructs=("justice", "truth", "autonomy"),
        )
        assert high > low  # threatened constructs increase severity

    def test_zero_verifications(self):
        score = compute_severity(verification_count=0, contradiction_count=0)
        assert score == 0.0

    def test_bounded(self):
        score = compute_severity(
            verification_count=100,
            contradiction_count=100,
            threatened_constructs=("a", "b", "c", "d", "e", "f", "g", "h"),
        )
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# audit_person (integration)
# ---------------------------------------------------------------------------


class TestAuditPerson:
    def test_full_audit_no_provider(self):
        """Full audit pipeline with heuristic fallbacks."""
        result = audit_person(
            person="Person A",
            finding_texts=[
                "Person A met Person B in New York.",
                "Person A transferred funds to an offshore account.",
            ],
            evidence_texts=[
                "Documents show Person A met Person B in New York in 2010.",
                "Bank records indicate Person A made wire transfers.",
            ],
        )
        assert isinstance(result, PersonAudit)
        assert result.person == "Person A"
        assert len(result.claims) >= 2
        assert len(result.verifications) == len(result.claims)
        assert 0.0 <= result.severity_score <= 1.0

    def test_empty_findings(self):
        result = audit_person(
            person="Person A",
            finding_texts=[],
            evidence_texts=[],
        )
        assert result.person == "Person A"
        assert result.claims == ()
        assert result.severity_score == 0.0

    def test_with_threatened_constructs(self):
        result = audit_person(
            person="Person A",
            finding_texts=["Person A was involved in trafficking."],
            evidence_texts=["Evidence of Person A involvement."],
            threatened_constructs=("justice", "autonomy"),
        )
        assert isinstance(result, PersonAudit)
        assert result.severity_score >= 0.0
