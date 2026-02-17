import pytest
from unittest.mock import MagicMock, patch

from src.core.providers import MockProvider, ModelProvider, VLLMProvider


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
