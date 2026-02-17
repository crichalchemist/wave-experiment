import os
from typing import Any, Protocol, runtime_checkable
from dataclasses import dataclass, field

try:
    from openai import OpenAI as _OpenAI
except ImportError:
    _OpenAI = None  # type: ignore[assignment,misc]

try:
    from azure.ai.inference import ChatCompletionsClient as _ChatCompletionsClient
    from azure.ai.inference.models import UserMessage as _UserMessage
    from azure.core.credentials import AzureKeyCredential as _AzureKeyCredential
except ImportError:
    _ChatCompletionsClient = None  # type: ignore[assignment,misc]
    _AzureKeyCredential = None     # type: ignore[assignment,misc]
    _UserMessage = None            # type: ignore[assignment,misc]

_VLLM_DUMMY_API_KEY: str = "not-needed"  # vLLM requires non-empty key; value is ignored
_AZURE_EMBED_NOT_SUPPORTED: str = "AzureFoundryProvider does not support embeddings"

_PROVIDER_VLLM: str = "vllm"
_PROVIDER_AZURE: str = "azure"
_ENV_PROVIDER: str = "DETECTIVE_PROVIDER"
_ENV_VLLM_URL: str = "VLLM_BASE_URL"
_ENV_VLLM_MODEL: str = "VLLM_MODEL"
_ENV_AZURE_ENDPOINT: str = "AZURE_ENDPOINT"
_ENV_AZURE_KEY: str = "AZURE_API_KEY"
_ENV_AZURE_MODEL: str = "AZURE_MODEL"


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
        object.__setattr__(self, "_client", _OpenAI(base_url=self.base_url, api_key=_VLLM_DUMMY_API_KEY))

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
class AzureFoundryProvider:
    """Azure AI Foundry (Claude). Used as critic model in CAI loops."""
    endpoint: str
    api_key: str
    model: str
    _client: Any = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if _ChatCompletionsClient is None:
            raise ImportError("azure-ai-inference package required for AzureFoundryProvider")
        client = _ChatCompletionsClient(
            endpoint=self.endpoint,
            credential=_AzureKeyCredential(self.api_key),
        )
        object.__setattr__(self, "_client", client)

    def complete(self, prompt: str, **kwargs) -> str:
        response = self._client.complete(
            messages=[_UserMessage(content=prompt)],
            model=self.model,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError(f"Azure Foundry returned no content for model {self.model!r}")
        return content

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError(_AZURE_EMBED_NOT_SUPPORTED)


def provider_from_env() -> ModelProvider:
    """Select provider from environment. DETECTIVE_PROVIDER=vllm|azure."""
    provider_type = os.environ.get(_ENV_PROVIDER)
    if provider_type is None:
        raise ValueError(
            f"{_ENV_PROVIDER} is not set. Must be one of: {_PROVIDER_VLLM!r}, {_PROVIDER_AZURE!r}"
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
    raise ValueError(f"Unknown provider type {provider_type!r}. Set {_ENV_PROVIDER}=vllm|azure")
