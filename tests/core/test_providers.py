import pytest
from unittest.mock import MagicMock, patch

from src.core.providers import (
    AzureFoundryProvider,
    MockProvider,
    ModelProvider,
    VLLMProvider,
    provider_from_env,
)


def test_mock_provider_satisfies_protocol() -> None:
    assert isinstance(MockProvider(), ModelProvider)


def test_mock_provider_complete_returns_configured_response() -> None:
    assert MockProvider(response="hello").complete("prompt") == "hello"


def test_mock_provider_embed_returns_configured_embedding() -> None:
    assert MockProvider(embedding=[1.0]).embed("text") == [1.0]


def test_vllm_provider_complete_calls_chat_completions() -> None:
    # Patch OpenAI at the usage site (module-level name) for proper test isolation.
    with patch("src.core.providers._OpenAI"):
        provider = VLLMProvider(base_url="http://localhost:8000/v1", model="mistral")

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices[0].message.content = "result"
    object.__setattr__(provider, "_client", mock_client)

    output = provider.complete("test prompt")

    mock_client.chat.completions.create.assert_called_once_with(
        model="mistral",
        messages=[{"role": "user", "content": "test prompt"}],
        temperature=0.0,
    )
    assert output == "result"


def test_vllm_provider_satisfies_protocol() -> None:
    with patch("src.core.providers._OpenAI"):
        provider = VLLMProvider(base_url="http://localhost:8000/v1", model="mistral")

    assert isinstance(provider, ModelProvider)


def test_vllm_provider_embed_calls_embeddings_create() -> None:
    with patch("src.core.providers._OpenAI"):
        provider = VLLMProvider(base_url="http://localhost:8000/v1", model="mistral")
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value.data = [MagicMock(embedding=[0.1, 0.2])]
    object.__setattr__(provider, "_client", mock_client)
    result = provider.embed("some text")
    mock_client.embeddings.create.assert_called_once_with(model="mistral", input="some text")
    assert result == [0.1, 0.2]


def test_vllm_provider_raises_on_missing_openai() -> None:
    with patch("src.core.providers._OpenAI", None):
        with pytest.raises(ImportError, match="openai package required"):
            VLLMProvider(base_url="http://localhost:8000/v1", model="mistral")


# --- AzureFoundryProvider tests ---


def test_azure_provider_satisfies_protocol() -> None:
    with patch("src.core.providers._ChatCompletionsClient"), \
         patch("src.core.providers._AzureKeyCredential"):
        provider = AzureFoundryProvider(
            endpoint="https://example.azure.com", api_key="key", model="claude"
        )
    assert isinstance(provider, ModelProvider)


def test_azure_provider_complete_calls_client() -> None:
    with patch("src.core.providers._ChatCompletionsClient"), \
         patch("src.core.providers._AzureKeyCredential"):
        provider = AzureFoundryProvider(
            endpoint="https://example.azure.com", api_key="key", model="claude"
        )
    mock_client = MagicMock()
    mock_client.complete.return_value.choices[0].message.content = "azure response"
    object.__setattr__(provider, "_client", mock_client)
    # Patch the lazy local import of UserMessage inside complete()
    mock_models = MagicMock()
    with patch.dict("sys.modules", {"azure.ai.inference.models": mock_models}):
        result = provider.complete("test prompt")
    assert result == "azure response"


def test_azure_provider_embed_raises() -> None:
    with patch("src.core.providers._ChatCompletionsClient"), \
         patch("src.core.providers._AzureKeyCredential"):
        provider = AzureFoundryProvider(
            endpoint="https://example.azure.com", api_key="key", model="claude"
        )
    with pytest.raises(NotImplementedError):
        provider.embed("text")


# --- provider_from_env tests ---


def test_provider_from_env_returns_vllm(monkeypatch) -> None:
    monkeypatch.setenv("DETECTIVE_PROVIDER", "vllm")
    monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("VLLM_MODEL", "mistral")
    with patch("src.core.providers._OpenAI"):
        p = provider_from_env()
    assert isinstance(p, VLLMProvider)
    assert p.base_url == "http://localhost:8000/v1"


def test_provider_from_env_returns_azure(monkeypatch) -> None:
    monkeypatch.setenv("DETECTIVE_PROVIDER", "azure")
    monkeypatch.setenv("AZURE_ENDPOINT", "https://example.azure.com")
    monkeypatch.setenv("AZURE_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_MODEL", "claude")
    with patch("src.core.providers._ChatCompletionsClient"), \
         patch("src.core.providers._AzureKeyCredential"):
        p = provider_from_env()
    assert isinstance(p, AzureFoundryProvider)


def test_provider_from_env_raises_on_unknown(monkeypatch) -> None:
    monkeypatch.setenv("DETECTIVE_PROVIDER", "unknown")
    with pytest.raises(ValueError, match="Unknown provider type"):
        provider_from_env()
