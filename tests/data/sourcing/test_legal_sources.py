def test_legal_domain_configs_has_three_domains():
    from src.data.sourcing.legal_sources import LEGAL_DOMAIN_CONFIGS
    assert "criminal_justice" in LEGAL_DOMAIN_CONFIGS
    assert "territorial_rights" in LEGAL_DOMAIN_CONFIGS
    assert "foia_transparency" in LEGAL_DOMAIN_CONFIGS


def test_each_domain_has_required_keys():
    from src.data.sourcing.legal_sources import LEGAL_DOMAIN_CONFIGS
    for domain, config in LEGAL_DOMAIN_CONFIGS.items():
        assert "hf_dataset" in config, f"{domain} missing hf_dataset"
        assert "hf_config" in config, f"{domain} missing hf_config"
        assert "keyword_filters" in config, f"{domain} missing keyword_filters"
        assert "description" in config, f"{domain} missing description"


def test_load_legal_domain_batch_returns_list():
    from unittest.mock import patch, MagicMock
    from src.data.sourcing.legal_sources import load_legal_domain_batch

    mock_ds = MagicMock()
    mock_ds.__iter__ = MagicMock(return_value=iter([
        {"text": "The sentencing guidelines require..." * 10},
    ]))

    with patch("src.data.sourcing.legal_sources.load_dataset", return_value=mock_ds):
        results = load_legal_domain_batch("criminal_justice", max_documents=5)

    assert isinstance(results, list)
    assert len(results) >= 1
    assert results[0].source.startswith("huggingface:")
    assert results[0].metadata["legal_domain"] == "criminal_justice"


def test_load_legal_domain_batch_unknown_domain():
    from src.data.sourcing.legal_sources import load_legal_domain_batch
    import pytest
    with pytest.raises(ValueError, match="Unknown legal domain"):
        load_legal_domain_batch("unknown_domain")
