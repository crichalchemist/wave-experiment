"""
OCR provider abstraction: DeepSeek-OCR (GPU) or pytesseract (CPU).

The OcrBackend Protocol allows transparent switching between backends.
At import time, the module detects which backend is available and exports
`ocr_image` as the correct implementation.

GPU path (Azure NC-series):
  - deepseek-ai/DeepSeek-OCR via transformers
  - Requires: CUDA + torch==2.6.0 + flash-attn==2.7.3
  - ~2500 tokens/second on A100-40G
  - Outputs markdown with layout preservation

CPU path (L-series / local dev):
  - pytesseract + Tesseract system binary
  - Requires: tesseract-ocr (apt package)
  - Slower but no GPU required
  - Plain text output

Selection: set OCR_BACKEND=deepseek to force GPU path.
Default: auto-detect based on torch.cuda.is_available().
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ALPHA_WEIGHT: float = 0.4
_WORD_DENSITY_WEIGHT: float = 0.3
_LENGTH_WEIGHT: float = 0.3
_LENGTH_NORMALIZER: int = 50  # text >= 50 chars gets full length score
_WORD_RE: re.Pattern[str] = re.compile(r"[a-zA-Z]{2,}")

try:
    from PIL import Image as _PILImage
except ImportError:
    _PILImage = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# OcrResult + confidence estimation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OcrResult:
    """Immutable result of an OCR extraction with confidence score."""

    text: str
    confidence: float  # [0.0, 1.0]
    backend_name: str

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence}"
            )


def estimate_ocr_confidence(text: str) -> float:
    """Heuristic OCR quality score.

    Combines three signals:
    - 40% alpha ratio: fraction of alphabetic characters
    - 30% word density: fraction of text covered by 2+ letter words
    - 30% length score: penalizes very short output (< 50 chars)
    """
    if not text or not text.strip():
        return 0.0

    alpha_ratio = sum(1 for c in text if c.isalpha()) / len(text)
    words = _WORD_RE.findall(text)
    word_density = sum(len(w) for w in words) / len(text) if text else 0.0
    length_score = min(1.0, len(text.strip()) / _LENGTH_NORMALIZER)

    raw = (
        _ALPHA_WEIGHT * alpha_ratio
        + _WORD_DENSITY_WEIGHT * word_density
        + _LENGTH_WEIGHT * length_score
    )
    return min(1.0, max(0.0, raw))


# ---------------------------------------------------------------------------
# OcrBackend Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class OcrBackend(Protocol):
    """Structural contract for OCR backends."""
    def extract_text(self, image: "_PILImage.Image") -> str: ...  # type: ignore[type-arg]
    @property
    def name(self) -> str: ...


class _TesseractBackend:
    """CPU fallback OCR using system Tesseract binary."""

    @property
    def name(self) -> str:
        return "tesseract"

    def extract_text(self, image: "_PILImage.Image") -> str:  # type: ignore[type-arg]
        try:
            import pytesseract
        except ImportError as e:
            raise ImportError("pip install pytesseract && apt install tesseract-ocr") from e
        return pytesseract.image_to_string(image)


class _DeepSeekOcrBackend:
    """
    GPU OCR using deepseek-ai/DeepSeek-OCR.

    Loads model lazily on first call to avoid startup cost when not needed.
    Requires CUDA + torch==2.6.0 + flash-attn==2.7.3.
    """

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None

    @property
    def name(self) -> str:
        return "deepseek-ocr"

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise ImportError("pip install transformers>=4.51.1") from e

        self._tokenizer = AutoTokenizer.from_pretrained(
            "deepseek-ai/DeepSeek-OCR",
            trust_remote_code=True,
        )
        self._model = AutoModel.from_pretrained(
            "deepseek-ai/DeepSeek-OCR",
            _attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )

    def extract_text(self, image: "_PILImage.Image") -> str:  # type: ignore[type-arg]
        self._load()
        import os
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image.save(tmp.name)
            tmp_path = tmp.name
        try:
            result = self._model.infer(
                self._tokenizer,
                prompt="<image>\nFree OCR.",
                image_file=tmp_path,
                base_size=1024,
                image_size=640,
            )
            return result if isinstance(result, str) else str(result)
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Fallback chain
# ---------------------------------------------------------------------------

_DEFAULT_CONFIDENCE_THRESHOLD: float = 0.6


@dataclass
class OcrFallbackChain:
    """Try OCR backends in order, return the highest-confidence result.

    Satisfies the ``OcrBackend`` Protocol so it can be used as a drop-in
    replacement at any call site that accepts an ``OcrBackend``.
    """

    backends: list[OcrBackend]
    confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD

    @property
    def name(self) -> str:
        return "fallback_chain"

    def extract_text(self, image: "_PILImage.Image") -> str:  # type: ignore[type-arg]
        """OcrBackend Protocol method — returns best text as plain string."""
        return self.extract_text_with_confidence(image).text

    def extract_text_with_confidence(self, image: "_PILImage.Image") -> OcrResult:  # type: ignore[type-arg]
        """Try each backend, keeping the highest-confidence result.

        Stops early if a result meets ``confidence_threshold``.
        """
        best = OcrResult(text="", confidence=0.0, backend_name="none")

        for backend in self.backends:
            try:
                text = backend.extract_text(image)
                conf = estimate_ocr_confidence(text)
                if conf > best.confidence:
                    best = OcrResult(
                        text=text, confidence=conf, backend_name=backend.name
                    )
                if conf >= self.confidence_threshold:
                    break
            except Exception:
                _logger.warning(
                    "OCR backend %s failed, trying next", backend.name
                )

        return best


def _select_backend() -> OcrBackend:
    """Auto-select backend: DeepSeek if CUDA available and not overridden."""
    forced = os.environ.get("OCR_BACKEND", "").lower()
    if forced == "tesseract":
        return _TesseractBackend()
    if forced == "deepseek":
        return _DeepSeekOcrBackend()

    # Auto-detect
    try:
        import torch
        if torch.cuda.is_available():
            return _DeepSeekOcrBackend()
    except ImportError:
        pass

    return _TesseractBackend()


# Module-level singleton — selected at import time
default_backend: OcrBackend = _select_backend()


def ocr_image(image: "_PILImage.Image", backend: OcrBackend | None = None) -> str:  # type: ignore[type-arg]
    """Extract text from a PIL Image using the selected OCR backend."""
    b = backend or default_backend
    return b.extract_text(image)
