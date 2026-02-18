"""Tests for Module C: geopolitical presumption detection."""
import pytest
from unittest.mock import MagicMock


def test_import():
    from src.detective.module_c import GeopoliticalDetection, detect_geopolitical_presumptions
    assert GeopoliticalDetection is not None
    assert callable(detect_geopolitical_presumptions)


def test_returns_list():
    from src.detective.module_c import detect_geopolitical_presumptions
    provider = MagicMock()
    provider.complete.return_value = "score: 0.1"
    result = detect_geopolitical_presumptions("The regulator reviewed the filing.", provider)
    assert isinstance(result, list)


def test_detects_regulatory_assumption():
    """Assuming oversight worked as designed is the core geopolitical presumption."""
    from src.detective.module_c import detect_geopolitical_presumptions
    provider = MagicMock()
    provider.complete.return_value = "score: 0.85"
    results = detect_geopolitical_presumptions(
        "The SEC properly reviewed all disclosures as required by law.", provider
    )
    assert len(results) >= 1
    assert results[0].score >= 0.7


def test_detects_intelligence_agency_assumption():
    from src.detective.module_c import detect_geopolitical_presumptions
    provider = MagicMock()
    provider.complete.return_value = "score: 0.80"
    results = detect_geopolitical_presumptions(
        "The FBI conducted a thorough investigation per standard protocol.", provider
    )
    assert len(results) >= 1


def test_detects_prosecutorial_discretion_assumption():
    from src.detective.module_c import detect_geopolitical_presumptions
    provider = MagicMock()
    provider.complete.return_value = "score: 0.75"
    results = detect_geopolitical_presumptions(
        "The prosecutor declined to pursue charges based on insufficient evidence.", provider
    )
    assert len(results) >= 1


def test_low_score_filtered():
    from src.detective.module_c import detect_geopolitical_presumptions
    provider = MagicMock()
    provider.complete.return_value = "score: 0.2"
    results = detect_geopolitical_presumptions(
        "The organization submitted its annual report.", provider
    )
    assert results == []


def test_dataclass_frozen():
    from src.detective.module_c import GeopoliticalDetection
    from src.core.types import AssumptionType
    d = GeopoliticalDetection(
        assumption_type=AssumptionType.GEOPOLITICAL_PRESUMPTION,
        score=0.8,
        source_text="The regulator approved...",
        presumed_actor="SEC",
    )
    with pytest.raises(Exception):
        d.score = 0.5  # type: ignore[misc]


def test_assumption_type_is_geopolitical():
    from src.detective.module_c import detect_geopolitical_presumptions
    from src.core.types import AssumptionType
    provider = MagicMock()
    provider.complete.return_value = "score: 0.8"
    results = detect_geopolitical_presumptions(
        "Intelligence agencies monitored the situation per their mandate.", provider
    )
    assert all(r.assumption_type == AssumptionType.GEOPOLITICAL_PRESUMPTION for r in results)


def test_detects_unstated_actor_interest():
    """Assumes actor behavior reflects stated institutional role, not actual interests."""
    from src.detective.module_c import detect_geopolitical_presumptions
    provider = MagicMock()
    provider.complete.return_value = "score: 0.88"
    results = detect_geopolitical_presumptions(
        "The government cooperated fully with the international inquiry as expected.", provider
    )
    assert len(results) >= 1


def test_multiple_actors_in_text():
    from src.detective.module_c import detect_geopolitical_presumptions
    provider = MagicMock()
    provider.complete.return_value = "score: 0.78"
    results = detect_geopolitical_presumptions(
        "The SEC and DOJ jointly reviewed the matter per standard inter-agency protocol.", provider
    )
    assert len(results) >= 2
