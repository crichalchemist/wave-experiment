import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Any, Protocol, runtime_checkable
from dataclasses import dataclass, field

_logger = logging.getLogger(__name__)

try:
    from openai import OpenAI as _OpenAI
except ImportError:
    _OpenAI = None  # type: ignore[assignment,misc]

_VLLM_DUMMY_API_KEY: str = "not-needed"  # vLLM requires non-empty key; value is ignored
_AZURE_EMBED_NOT_SUPPORTED: str = "AzureFoundryProvider does not support embeddings"

_PROVIDER_VLLM: str = "vllm"
_PROVIDER_AZURE: str = "azure"
_PROVIDER_HYBRID: str = "hybrid"
_PROVIDER_OLLAMA: str = "ollama"
_ENV_PROVIDER: str = "DETECTIVE_PROVIDER"
_ENV_VLLM_URL: str = "VLLM_BASE_URL"
_ENV_VLLM_MODEL: str = "VLLM_MODEL"
_ENV_AZURE_ENDPOINT: str = "AZURE_ENDPOINT"
_ENV_AZURE_KEY: str = "AZURE_API_KEY"
_ENV_AZURE_MODEL: str = "AZURE_MODEL"
_ENV_OLLAMA_URL: str = "OLLAMA_BASE_URL"
_ENV_OLLAMA_MODEL: str = "OLLAMA_MODEL"

# Critic provider — separate from the main inference provider.
# Used in the CAI critique loop (constitutional warmup, preference pair generation).
# Reads AZURE_CRITIC_* vars so both Ollama (local) and Azure (critic) can be set at once.
_ENV_CRITIC_ENDPOINT: str = "AZURE_CRITIC_ENDPOINT"
_ENV_CRITIC_KEY: str = "AZURE_CRITIC_KEY"
_ENV_CRITIC_MODEL: str = "AZURE_CRITIC_MODEL"

# Patterns that identify scoring prompts (constrained output, float-only responses).
# Matched against prompt text to decide local-vs-cloud routing.
_SCORING_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"Reply with(?:\s+ONLY)?[:\s]+score:", re.IGNORECASE),
    re.compile(r"Reply with ONLY a float between", re.IGNORECASE),
    re.compile(r"Return only a single float", re.IGNORECASE),
    re.compile(r"confidence:\s*<float", re.IGNORECASE),
)


def classify_prompt(prompt: str) -> str:
    """Classify prompt as 'scoring' or 'reasoning' based on output directives."""
    for pattern in _SCORING_PATTERNS:
        if pattern.search(prompt):
            return "scoring"
    return "reasoning"


@runtime_checkable
class ModelProvider(Protocol):
    def complete(self, prompt: str, **kwargs) -> str: ...
    def embed(self, text: str) -> list[float]: ...


@dataclass
class MockProvider:
    """Test double for ModelProvider."""
    response: str = "mock response"
    embedding: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])

    def complete(self, prompt: str, **kwargs) -> str:
        return self.response

    def embed(self, text: str) -> list[float]:
        return self.embedding


@dataclass(frozen=True)
class VLLMProvider:
    """Connects to a local vLLM instance (OpenAI-compatible API)."""
    base_url: str
    model: str
    temperature: float = 0.0
    _client: Any = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if _OpenAI is None:
            raise ImportError("openai package required for VLLMProvider")
        object.__setattr__(self, "_client", _OpenAI(
            base_url=self.base_url,
            api_key=_VLLM_DUMMY_API_KEY,
            timeout=600.0,  # 10 min — CPU inference on large chunks is slow
        ))

    def complete(self, prompt: str, **kwargs) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.temperature),
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError(f"vLLM returned no content for model {self.model!r}")
        return content

    def embed(self, text: str) -> list[float]:
        response = self._client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding


@dataclass(frozen=True)
class OllamaProvider:
    """Connects to a local Ollama instance (OpenAI-compatible API).

    Designed for lightweight scoring prompts on small models (e.g. qwen2.5:0.5b).
    Uses a 30s timeout to fail fast — if Ollama is slow, callers should fall back
    to a cloud provider rather than block.
    """
    base_url: str = "http://localhost:11434/v1"
    model: str = "qwen2.5:0.5b"
    temperature: float = 0.0
    _client: Any = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if _OpenAI is None:
            raise ImportError("openai package required for OllamaProvider")
        object.__setattr__(self, "_client", _OpenAI(
            base_url=self.base_url,
            api_key=_VLLM_DUMMY_API_KEY,  # Ollama ignores this but OpenAI client requires it
            timeout=30.0,  # fail fast — scoring prompts are simple
        ))

    def complete(self, prompt: str, **kwargs) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.temperature),
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError(f"Ollama returned no content for model {self.model!r}")
        return content

    def embed(self, text: str) -> list[float]:
        response = self._client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding


@dataclass(frozen=True)
class AzureFoundryProvider:
    """Azure AI Foundry — Anthropic-native endpoint.

    Calls POST {endpoint}/anthropic/v1/messages with x-api-key auth.
    Deployment names discovered via `az cognitiveservices account deployment list`.

    Known deployments on scaffoldworker (resource group: tinkertown):
      claude-sonnet-4-5-2 | claude-opus-4-6 | claude-haiku-4-5
    """
    endpoint: str   # e.g. https://scaffoldworker.services.ai.azure.com/
    api_key: str
    model: str      # deployment name, e.g. claude-haiku-4-5

    def _messages_url(self) -> str:
        return self.endpoint.rstrip("/") + "/anthropic/v1/messages"

    def complete(self, prompt: str, **kwargs) -> str:
        payload = json.dumps({
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        req = urllib.request.Request(
            self._messages_url(),
            data=payload,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"Azure Foundry {exc.code} for model {self.model!r}: {exc.read().decode()[:200]}"
            ) from exc
        try:
            return body["content"][0]["text"]
        except (KeyError, IndexError) as exc:
            raise ValueError(f"Unexpected response shape from Azure Foundry: {body}") from exc

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError(_AZURE_EMBED_NOT_SUPPORTED)


@dataclass
class HybridRoutingProvider:
    """Routes scoring prompts to a local model and reasoning prompts to Azure.

    Uses classify_prompt() to inspect prompt text. Circuit-breaker: if the
    scoring provider fails once, all subsequent calls fall back to the
    reasoning provider until reset_fallback() is called.
    """
    scoring_provider: OllamaProvider
    reasoning_provider: AzureFoundryProvider
    _ollama_available: bool = field(default=True, init=False, repr=False)

    def complete(self, prompt: str, **kwargs) -> str:
        route = classify_prompt(prompt)
        if route == "scoring" and self._ollama_available:
            try:
                return self.scoring_provider.complete(prompt, **kwargs)
            except Exception as exc:
                _logger.warning("Ollama failed (%s), falling back to Azure", exc)
                self._ollama_available = False
                return self.reasoning_provider.complete(prompt, **kwargs)
        return self.reasoning_provider.complete(prompt, **kwargs)

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError("HybridRoutingProvider does not support embeddings")

    def reset_fallback(self) -> None:
        """Re-enable scoring provider after Ollama recovery."""
        self._ollama_available = True


def critic_provider_from_env() -> "AzureFoundryProvider":
    """Load Azure Foundry critic from AZURE_CRITIC_* env vars.

    Separate from provider_from_env() so local Ollama inference and
    Azure Foundry critique can be active simultaneously during CAI warmup.

    Required env vars:
      AZURE_CRITIC_ENDPOINT  — e.g. https://scaffoldworker.services.ai.azure.com/
      AZURE_CRITIC_KEY       — Azure AI Foundry API key
    Optional:
      AZURE_CRITIC_MODEL     — defaults to claude-sonnet-4-5
    """
    endpoint = os.environ.get(_ENV_CRITIC_ENDPOINT)
    api_key = os.environ.get(_ENV_CRITIC_KEY)
    if not endpoint or not api_key:
        raise ValueError(
            f"Set {_ENV_CRITIC_ENDPOINT} and {_ENV_CRITIC_KEY} to use Azure Foundry as critic. "
            f"Endpoint format: https://<resource>.services.ai.azure.com/"
        )
    model = os.environ.get(_ENV_CRITIC_MODEL, "claude-sonnet-4-5-2")
    return AzureFoundryProvider(endpoint=endpoint, api_key=api_key, model=model)


def provider_from_env() -> ModelProvider:
    """Select provider from environment. DETECTIVE_PROVIDER=vllm|azure|ollama|hybrid."""
    provider_type = os.environ.get(_ENV_PROVIDER)
    if provider_type is None:
        raise ValueError(
            f"{_ENV_PROVIDER} is not set. Must be one of: "
            f"{_PROVIDER_VLLM!r}, {_PROVIDER_AZURE!r}, {_PROVIDER_OLLAMA!r}, {_PROVIDER_HYBRID!r}"
        )
    if provider_type == _PROVIDER_VLLM:
        base_url = os.environ[_ENV_VLLM_URL]
        model = os.environ[_ENV_VLLM_MODEL]
        return VLLMProvider(base_url=base_url, model=model)
    elif provider_type == _PROVIDER_AZURE:
        endpoint = os.environ[_ENV_AZURE_ENDPOINT]
        api_key = os.environ[_ENV_AZURE_KEY]
        model = os.environ[_ENV_AZURE_MODEL]
        return AzureFoundryProvider(endpoint=endpoint, api_key=api_key, model=model)
    elif provider_type == _PROVIDER_OLLAMA:
        base_url = os.environ.get(_ENV_OLLAMA_URL, "http://localhost:11434/v1")
        model = os.environ.get(_ENV_OLLAMA_MODEL, "qwen2.5:0.5b")
        return OllamaProvider(base_url=base_url, model=model)
    elif provider_type == _PROVIDER_HYBRID:
        # Scoring → local Ollama, reasoning → Azure Foundry
        ollama_url = os.environ.get(_ENV_OLLAMA_URL, "http://localhost:11434/v1")
        ollama_model = os.environ.get(_ENV_OLLAMA_MODEL, "qwen2.5:0.5b")
        scoring = OllamaProvider(base_url=ollama_url, model=ollama_model)
        azure_endpoint = os.environ[_ENV_AZURE_ENDPOINT]
        azure_key = os.environ[_ENV_AZURE_KEY]
        azure_model = os.environ[_ENV_AZURE_MODEL]
        reasoning = AzureFoundryProvider(
            endpoint=azure_endpoint, api_key=azure_key, model=azure_model
        )
        return HybridRoutingProvider(scoring_provider=scoring, reasoning_provider=reasoning)
    raise ValueError(
        f"Unknown provider type {provider_type!r}. "
        f"Set {_ENV_PROVIDER}=vllm|azure|ollama|hybrid"
    )
