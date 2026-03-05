"""Tests for OCR provider abstraction."""
from unittest.mock import MagicMock

from src.data.sourcing.ocr_provider import (
    OcrBackend,
    _DeepSeekOcrBackend,
    _TesseractBackend,
    _select_backend,
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
