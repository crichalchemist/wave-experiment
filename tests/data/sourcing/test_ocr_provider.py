"""Tests for OCR provider abstraction."""
import pytest
from unittest.mock import MagicMock

from src.data.sourcing.ocr_provider import (
    OcrBackend,
    OcrFallbackChain,
    OcrResult,
    _DeepSeekOcrBackend,
    _TesseractBackend,
    _select_backend,
    estimate_ocr_confidence,
    ocr_image,
)


# --------------------------------------------------------------------------- #
# Protocol compliance
# --------------------------------------------------------------------------- #

def test_ocr_backend_protocol_compliance():
    """Verify _TesseractBackend satisfies OcrBackend Protocol."""
    backend = _TesseractBackend()
    assert isinstance(backend, OcrBackend)


def test_deepseek_backend_protocol_compliance():
    """Verify _DeepSeekOcrBackend satisfies OcrBackend Protocol."""
    backend = _DeepSeekOcrBackend()
    assert isinstance(backend, OcrBackend)


# --------------------------------------------------------------------------- #
# Backend name properties
# --------------------------------------------------------------------------- #

def test_tesseract_backend_name():
    """Verify name property returns 'tesseract'."""
    backend = _TesseractBackend()
    assert backend.name == "tesseract"


def test_deepseek_backend_name():
    """Verify name property returns 'deepseek-ocr'."""
    backend = _DeepSeekOcrBackend()
    assert backend.name == "deepseek-ocr"


# --------------------------------------------------------------------------- #
# Lazy loading
# --------------------------------------------------------------------------- #

def test_deepseek_backend_lazy_loading():
    """Create instance, verify _model is None before extract_text called."""
    backend = _DeepSeekOcrBackend()
    assert backend._model is None
    assert backend._tokenizer is None


# --------------------------------------------------------------------------- #
# Backend selection via environment
# --------------------------------------------------------------------------- #

def test_select_backend_tesseract_env_override(monkeypatch):
    """Set OCR_BACKEND=tesseract, call _select_backend, verify returns _TesseractBackend."""
    monkeypatch.setenv("OCR_BACKEND", "tesseract")
    backend = _select_backend()
    assert isinstance(backend, _TesseractBackend)


def test_select_backend_deepseek_env_override(monkeypatch):
    """Set OCR_BACKEND=deepseek, call _select_backend, verify returns _DeepSeekOcrBackend."""
    monkeypatch.setenv("OCR_BACKEND", "deepseek")
    backend = _select_backend()
    assert isinstance(backend, _DeepSeekOcrBackend)


# --------------------------------------------------------------------------- #
# Delegation
# --------------------------------------------------------------------------- #

def test_ocr_image_delegates_to_backend():
    """Create mock backend with extract_text method, call ocr_image, verify delegation."""
    mock_backend = MagicMock(spec=OcrBackend)
    mock_backend.extract_text.return_value = "extracted text from image"

    fake_image = MagicMock()  # Stand-in for PIL Image
    result = ocr_image(fake_image, backend=mock_backend)

    assert result == "extracted text from image"
    mock_backend.extract_text.assert_called_once_with(fake_image)


# --------------------------------------------------------------------------- #
# OcrResult frozen dataclass
# --------------------------------------------------------------------------- #


class TestOcrResult:
    def test_frozen(self):
        r = OcrResult(text="hello world", confidence=0.9, backend_name="tesseract")
        with pytest.raises(AttributeError):
            r.text = "mutated"  # type: ignore[misc]

    def test_fields(self):
        r = OcrResult(text="some text", confidence=0.75, backend_name="deepseek-ocr")
        assert r.text == "some text"
        assert r.confidence == 0.75
        assert r.backend_name == "deepseek-ocr"

    def test_confidence_lower_bound(self):
        with pytest.raises(ValueError, match="confidence must be in"):
            OcrResult(text="hello", confidence=-0.1, backend_name="test")

    def test_confidence_upper_bound(self):
        with pytest.raises(ValueError, match="confidence must be in"):
            OcrResult(text="hello", confidence=1.5, backend_name="test")

    def test_confidence_zero(self):
        r = OcrResult(text="", confidence=0.0, backend_name="none")
        assert r.confidence == 0.0

    def test_confidence_one(self):
        r = OcrResult(text="perfect", confidence=1.0, backend_name="oracle")
        assert r.confidence == 1.0


# --------------------------------------------------------------------------- #
# estimate_ocr_confidence
# --------------------------------------------------------------------------- #


class TestEstimateOcrConfidence:
    def test_good_text(self):
        conf = estimate_ocr_confidence(
            "Jeffrey Epstein was arrested in New York on federal charges."
        )
        assert 0.5 < conf <= 1.0

    def test_empty_text(self):
        assert estimate_ocr_confidence("") == 0.0

    def test_whitespace_only(self):
        assert estimate_ocr_confidence("   \n\t  ") == 0.0

    def test_garbage_text(self):
        garbage = "||||///\\\\~~~~@@@###$$$" * 10
        assert estimate_ocr_confidence(garbage) <= 0.3

    def test_returns_float(self):
        result = estimate_ocr_confidence("Some normal text here.")
        assert isinstance(result, float)

    def test_bounded_zero_to_one(self):
        for text in ["", "x", "Hello world" * 100, "###" * 50]:
            conf = estimate_ocr_confidence(text)
            assert 0.0 <= conf <= 1.0

    def test_short_text_lower_than_long(self):
        short_conf = estimate_ocr_confidence("Hi")
        long_conf = estimate_ocr_confidence(
            "This is a much longer sentence with many real English words."
        )
        assert long_conf > short_conf


# --------------------------------------------------------------------------- #
# OcrFallbackChain
# --------------------------------------------------------------------------- #


def _make_mock_backend(name: str, text: str, raises: bool = False) -> MagicMock:
    """Create a mock OcrBackend with given name and extract_text behavior."""
    backend = MagicMock(spec=OcrBackend)
    backend.name = name
    if raises:
        backend.extract_text.side_effect = RuntimeError("backend crashed")
    else:
        backend.extract_text.return_value = text
    return backend


class TestOcrFallbackChain:
    def test_protocol_compliance(self):
        """OcrFallbackChain satisfies the OcrBackend Protocol."""
        chain = OcrFallbackChain(backends=[])
        assert isinstance(chain, OcrBackend)

    def test_name_property(self):
        chain = OcrFallbackChain(backends=[])
        assert chain.name == "fallback_chain"

    def test_best_confidence_wins(self):
        """Chain picks the backend producing the highest confidence text."""
        low = _make_mock_backend("low", "x")  # very short → low confidence
        high = _make_mock_backend(
            "high",
            "Jeffrey Epstein was arrested in New York on federal charges last Tuesday",
        )
        chain = OcrFallbackChain(backends=[low, high], confidence_threshold=0.95)
        result = chain.extract_text_with_confidence(MagicMock())
        assert result.backend_name == "high"
        assert result.confidence > 0.5

    def test_early_stop_on_threshold(self):
        """Chain stops trying backends once confidence exceeds threshold."""
        good = _make_mock_backend(
            "good",
            "Jeffrey Epstein was arrested in New York on federal charges last Tuesday",
        )
        never_called = _make_mock_backend("never", "should not be called")
        chain = OcrFallbackChain(
            backends=[good, never_called], confidence_threshold=0.5
        )
        result = chain.extract_text_with_confidence(MagicMock())
        assert result.backend_name == "good"
        never_called.extract_text.assert_not_called()

    def test_fallback_on_exception(self):
        """If first backend raises, chain falls back to next."""
        failing = _make_mock_backend("failing", "", raises=True)
        working = _make_mock_backend(
            "working",
            "This document was obtained through a Freedom of Information Act request.",
        )
        chain = OcrFallbackChain(backends=[failing, working])
        result = chain.extract_text_with_confidence(MagicMock())
        assert result.backend_name == "working"
        assert result.confidence > 0.0

    def test_empty_chain_returns_empty(self):
        """Chain with no backends returns empty OcrResult."""
        chain = OcrFallbackChain(backends=[])
        result = chain.extract_text_with_confidence(MagicMock())
        assert result.text == ""
        assert result.confidence == 0.0
        assert result.backend_name == "none"

    def test_all_backends_fail(self):
        """If all backends fail, returns empty OcrResult."""
        b1 = _make_mock_backend("b1", "", raises=True)
        b2 = _make_mock_backend("b2", "", raises=True)
        chain = OcrFallbackChain(backends=[b1, b2])
        result = chain.extract_text_with_confidence(MagicMock())
        assert result.text == ""
        assert result.confidence == 0.0

    def test_extract_text_returns_string(self):
        """The Protocol-compatible extract_text() returns plain text."""
        backend = _make_mock_backend(
            "test",
            "This is a normal OCR result with recognizable English words in it.",
        )
        chain = OcrFallbackChain(backends=[backend])
        text = chain.extract_text(MagicMock())
        assert isinstance(text, str)
        assert "normal OCR result" in text

    def test_extract_text_with_confidence_returns_ocr_result(self):
        backend = _make_mock_backend(
            "test",
            "A reasonably long sentence with enough words for confidence scoring.",
        )
        chain = OcrFallbackChain(backends=[backend])
        result = chain.extract_text_with_confidence(MagicMock())
        assert isinstance(result, OcrResult)
