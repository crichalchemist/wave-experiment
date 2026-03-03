import json
import pytest
from unittest.mock import MagicMock, patch

from src.core.providers import (
    AzureFoundryProvider,
    HybridRoutingProvider,
    MockProvider,
    ModelProvider,
    OllamaProvider,
    VLLMProvider,
    classify_prompt,
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


# --- classify_prompt tests ---


class TestClassifyPrompt:
    """Test prompt classification against real patterns from the codebase."""

    def test_module_a_scoring_prompt(self) -> None:
        # From module_a.py _SCORING_PROMPT
        prompt = "Reply with: score: <float between 0 and 1>"
        assert classify_prompt(prompt) == "scoring"

    def test_module_b_scoring_prompt(self) -> None:
        # From module_b.py _SCORE_PROMPT
        prompt = "Reply with ONLY: score: <float>\n\nText span: some text"
        assert classify_prompt(prompt) == "scoring"

    def test_module_c_scoring_prompt(self) -> None:
        # From module_c.py _SCORE_PROMPT
        prompt = "Reply with ONLY: score: <float>\n\nSentence: test"
        assert classify_prompt(prompt) == "scoring"

    def test_evolution_scoring_prompt(self) -> None:
        # From evolution.py confidence update prompt
        prompt = "Reply with ONLY a float between 0.0 and 1.0 representing the updated confidence."
        assert classify_prompt(prompt) == "scoring"

    def test_graph_edge_scoring_prompt(self) -> None:
        # From graph.py EDGE_SCORE_PROMPT
        prompt = "Score the plausibility of this relationship on a scale from 0.0 to 1.0. Return only a single float.\n\nRelationship: A → B"
        assert classify_prompt(prompt) == "scoring"

    def test_parallel_evolution_scoring_prompt(self) -> None:
        # From parallel_evolution.py _BRANCH_PROMPT
        prompt = "Then state the updated confidence as: confidence: <float between 0 and 1>"
        assert classify_prompt(prompt) == "scoring"

    def test_reasoning_prompt_default(self) -> None:
        prompt = "Analyze the following evidence and provide step-by-step reasoning:\n\nEvidence: financial records missing for 2013-2015"
        assert classify_prompt(prompt) == "reasoning"

    def test_unknown_prompt_defaults_to_reasoning(self) -> None:
        prompt = "What is the meaning of life?"
        assert classify_prompt(prompt) == "reasoning"


# --- OllamaProvider tests ---


def test_ollama_provider_satisfies_protocol() -> None:
    with patch("src.core.providers._OpenAI"):
        provider = OllamaProvider()
    assert isinstance(provider, ModelProvider)


def test_ollama_provider_defaults() -> None:
    with patch("src.core.providers._OpenAI"):
        provider = OllamaProvider()
    assert provider.model == "qwen2.5:0.5b"
    assert provider.base_url == "http://localhost:11434/v1"


def test_ollama_provider_complete_calls_chat_completions() -> None:
    with patch("src.core.providers._OpenAI"):
        provider = OllamaProvider()

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices[0].message.content = "0.85"
    object.__setattr__(provider, "_client", mock_client)

    output = provider.complete("Reply with ONLY: score: <float>")

    mock_client.chat.completions.create.assert_called_once_with(
        model="qwen2.5:0.5b",
        messages=[{"role": "user", "content": "Reply with ONLY: score: <float>"}],
        temperature=0.0,
    )
    assert output == "0.85"


def test_ollama_provider_raises_on_missing_openai() -> None:
    with patch("src.core.providers._OpenAI", None):
        with pytest.raises(ImportError, match="openai package required"):
            OllamaProvider()


# --- AzureFoundryProvider tests ---


def _make_azure_urlopen_mock(text: str) -> MagicMock:
    """Build a mock for urllib.request.urlopen that returns an Anthropic-format body."""
    body = json.dumps({"content": [{"text": text}]}).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def test_azure_provider_satisfies_protocol() -> None:
    # AzureFoundryProvider no longer has __post_init__ — construct directly.
    provider = AzureFoundryProvider(
        endpoint="https://example.azure.com", api_key="key", model="claude"
    )
    assert isinstance(provider, ModelProvider)


def test_azure_provider_complete_calls_client() -> None:
    provider = AzureFoundryProvider(
        endpoint="https://example.azure.com/", api_key="key", model="claude"
    )
    mock_resp = _make_azure_urlopen_mock("azure response")
    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        result = provider.complete("test prompt")

    mock_urlopen.assert_called_once()
    req = mock_urlopen.call_args[0][0]
    assert req.full_url == "https://example.azure.com/anthropic/v1/messages"
    assert req.get_header("X-api-key") == "key"
    sent = json.loads(req.data)
    assert sent["model"] == "claude"
    assert sent["messages"][0]["content"] == "test prompt"
    assert result == "azure response"


def test_azure_provider_embed_raises() -> None:
    provider = AzureFoundryProvider(
        endpoint="https://example.azure.com", api_key="key", model="claude"
    )
    with pytest.raises(NotImplementedError):
        provider.embed("text")


# --- HybridRoutingProvider tests ---


class TestHybridRoutingProvider:
    def _make_hybrid(self, scoring_response="0.75", reasoning_response="detailed analysis") -> HybridRoutingProvider:
        scoring = MockProvider(response=scoring_response)
        reasoning = MockProvider(response=reasoning_response)
        # Use MockProvider as stand-ins — HybridRoutingProvider only calls .complete()
        return HybridRoutingProvider(
            scoring_provider=scoring,  # type: ignore[arg-type]
            reasoning_provider=reasoning,  # type: ignore[arg-type]
        )

    def test_hybrid_satisfies_protocol(self) -> None:
        hybrid = self._make_hybrid()
        assert isinstance(hybrid, ModelProvider)

    def test_scoring_prompt_routes_to_scoring_provider(self) -> None:
        hybrid = self._make_hybrid()
        result = hybrid.complete("Reply with ONLY: score: <float>")
        assert result == "0.75"

    def test_reasoning_prompt_routes_to_reasoning_provider(self) -> None:
        hybrid = self._make_hybrid()
        result = hybrid.complete("Analyze the following evidence and explain step-by-step")
        assert result == "detailed analysis"

    def test_scoring_fallback_on_exception(self) -> None:
        hybrid = self._make_hybrid()
        # Make scoring provider raise
        hybrid.scoring_provider.complete = MagicMock(side_effect=ConnectionError("Ollama down"))  # type: ignore[assignment]
        result = hybrid.complete("Reply with ONLY: score: <float>")
        assert result == "detailed analysis"  # fell back to reasoning

    def test_circuit_breaker_stays_open_after_failure(self) -> None:
        hybrid = self._make_hybrid()
        hybrid.scoring_provider.complete = MagicMock(side_effect=ConnectionError("Ollama down"))  # type: ignore[assignment]
        hybrid.complete("Reply with ONLY: score: <float>")  # triggers fallback

        # Now scoring should NOT be attempted — goes straight to reasoning
        hybrid.scoring_provider.complete = MagicMock(return_value="should not be called")  # type: ignore[assignment]
        result = hybrid.complete("Reply with ONLY: score: <float>")
        hybrid.scoring_provider.complete.assert_not_called()
        assert result == "detailed analysis"

    def test_reset_fallback_re_enables_scoring(self) -> None:
        hybrid = self._make_hybrid()
        hybrid._ollama_available = False
        hybrid.reset_fallback()
        result = hybrid.complete("Reply with ONLY: score: <float>")
        assert result == "0.75"

    def test_embed_raises(self) -> None:
        hybrid = self._make_hybrid()
        with pytest.raises(NotImplementedError):
            hybrid.embed("text")


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
    # AzureFoundryProvider no longer calls azure-ai-inference in __init__
    p = provider_from_env()
    assert isinstance(p, AzureFoundryProvider)


def test_provider_from_env_raises_on_unknown(monkeypatch) -> None:
    monkeypatch.setenv("DETECTIVE_PROVIDER", "unknown")
    with pytest.raises(ValueError, match="Unknown provider type"):
        provider_from_env()


def test_provider_from_env_raises_when_not_set(monkeypatch) -> None:
    monkeypatch.delenv("DETECTIVE_PROVIDER", raising=False)
    with pytest.raises(ValueError, match="DETECTIVE_PROVIDER is not set"):
        provider_from_env()
