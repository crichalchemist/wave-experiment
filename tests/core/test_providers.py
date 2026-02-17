from unittest.mock import MagicMock, patch

from src.core.providers import MockProvider, ModelProvider, VLLMProvider


def test_mock_provider_satisfies_protocol() -> None:
    assert isinstance(MockProvider(), ModelProvider)


def test_mock_provider_complete_returns_configured_response() -> None:
    assert MockProvider(response="hello").complete("prompt") == "hello"


def test_mock_provider_embed_returns_configured_embedding() -> None:
    assert MockProvider(embedding=[1.0]).embed("text") == [1.0]


def test_vllm_provider_complete_calls_chat_completions() -> None:
    # Patch OpenAI at the import site inside __post_init__ so no real client is made.
    with patch("openai.OpenAI"):
        provider = VLLMProvider(base_url="http://localhost:8000/v1", model="mistral")

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices[0].message.content = "result"
    provider._client = mock_client

    output = provider.complete("test prompt")

    mock_client.chat.completions.create.assert_called_once_with(
        model="mistral",
        messages=[{"role": "user", "content": "test prompt"}],
        temperature=0.0,
    )
    assert output == "result"


def test_vllm_provider_satisfies_protocol() -> None:
    with patch("openai.OpenAI"):
        provider = VLLMProvider(base_url="http://localhost:8000/v1", model="mistral")

    assert isinstance(provider, ModelProvider)
