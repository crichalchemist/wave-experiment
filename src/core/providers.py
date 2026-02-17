from typing import Any, Protocol, runtime_checkable
from dataclasses import dataclass, field

try:
    from openai import OpenAI as _OpenAI
except ImportError:
    _OpenAI = None  # type: ignore[assignment,misc]

_VLLM_DUMMY_API_KEY: str = "not-needed"  # vLLM requires non-empty key; value is ignored


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
