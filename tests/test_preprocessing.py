"""
test_preprocessing.py
=====================
G19 Facial Emotion Recognition — CSE276

Purpose:
    Unit tests for src/preprocessing.py.
    Run with: pytest tests/test_preprocessing.py -v

Tests verify:
    - preprocess_image handles all valid input shapes.
    - Output has the correct shape (1, 48, 48, 1) and dtype float32.
    - Pixel values are in the [0, 1] range after normalisation.
    - Invalid inputs raise the expected exceptions.
    - apply_clahe_batch returns correct shape and dtype.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src.preprocessing import preprocess_image, apply_clahe_batch


# ---------------------------------------------------------------------------
# Fixtures: common dummy images
# ---------------------------------------------------------------------------

@pytest.fixture
def colour_image():
    """A random 100×100 BGR colour image (uint8)."""
    return np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def greyscale_image():
    """A random 80×80 greyscale image (uint8)."""
    return np.random.randint(0, 256, (80, 80), dtype=np.uint8)


@pytest.fixture
def greyscale_with_channel():
    """A random 60×60 greyscale image with channel dim (uint8)."""
    return np.random.randint(0, 256, (60, 60, 1), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Tests for preprocess_image
# ---------------------------------------------------------------------------

class TestPreprocessImage:

    def test_output_shape_from_colour(self, colour_image):
        """Output shape must be (1, 48, 48, 1) for a colour input."""
        result = preprocess_image(colour_image)
        assert result.shape == (1, 48, 48, 1), f"Unexpected shape: {result.shape}"

    def test_output_shape_from_greyscale(self, greyscale_image):
        """Output shape must be (1, 48, 48, 1) for a greyscale input."""
        result = preprocess_image(greyscale_image)
        assert result.shape == (1, 48, 48, 1), f"Unexpected shape: {result.shape}"

    def test_output_shape_from_greyscale_with_channel(self, greyscale_with_channel):
        """Output shape must be (1, 48, 48, 1) for a (H, W, 1) input."""
        result = preprocess_image(greyscale_with_channel)
        assert result.shape == (1, 48, 48, 1), f"Unexpected shape: {result.shape}"

    def test_output_dtype(self, colour_image):
        """Output must be float32."""
        result = preprocess_image(colour_image)
        assert result.dtype == np.float32, f"Unexpected dtype: {result.dtype}"

    def test_output_range(self, colour_image):
        """All pixel values must lie in [0.0, 1.0]."""
        result = preprocess_image(colour_image)
        assert result.min() >= 0.0, "Min value below 0"
        assert result.max() <= 1.0, "Max value above 1"

    def test_raises_on_none(self):
        """Should raise ValueError when given None."""
        with pytest.raises((ValueError, AttributeError)):
            preprocess_image(None)

    def test_raises_on_invalid_channels(self):
        """Should raise ValueError for a 4-channel image."""
        bad_image = np.zeros((48, 48, 4), dtype=np.uint8)
        with pytest.raises(ValueError):
            preprocess_image(bad_image)

    def test_handles_already_48x48(self):
        """Should work correctly when input is already 48×48."""
        img = np.random.randint(0, 256, (48, 48), dtype=np.uint8)
        result = preprocess_image(img)
        assert result.shape == (1, 48, 48, 1)

    def test_baseline_skips_clahe_by_default(self):
        """Baseline preprocess_image(img) should match img/255 without CLAHE."""
        img = (np.arange(48 * 48) % 256).astype(np.uint8).reshape(48, 48)
        result = preprocess_image(img)
        expected = (img.astype(np.float32) / 255.0)[np.newaxis, :, :, np.newaxis]
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_optional_clahe_flag(self):
        """preprocess_image(img, use_clahe=True) should return valid normalised output."""
        img = np.random.randint(0, 256, (48, 48), dtype=np.uint8)
        result = preprocess_image(img, use_clahe=True)
        assert result.shape == (1, 48, 48, 1)
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0


# ---------------------------------------------------------------------------
# Tests for apply_clahe_batch
# ---------------------------------------------------------------------------

class TestApplyCLAHEBatch:

    def test_output_shape_no_channel_dim(self):
        """Input (N, 48, 48) → output (N, 48, 48)."""
        batch = np.random.randint(0, 256, (10, 48, 48), dtype=np.uint8)
        result = apply_clahe_batch(batch)
        assert result.shape == (10, 48, 48)

    def test_output_shape_with_channel_dim(self):
        """Input (N, 48, 48, 1) → output (N, 48, 48, 1)."""
        batch = np.random.randint(0, 256, (5, 48, 48, 1), dtype=np.uint8)
        result = apply_clahe_batch(batch)
        assert result.shape == (5, 48, 48, 1)

    def test_output_dtype_float32(self):
        """Output must be float32."""
        batch = np.random.randint(0, 256, (4, 48, 48), dtype=np.uint8)
        result = apply_clahe_batch(batch)
        assert result.dtype == np.float32

    def test_output_range(self):
        """Output values must be in [0, 1]."""
        batch = np.random.randint(0, 256, (4, 48, 48), dtype=np.uint8)
        result = apply_clahe_batch(batch)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_accepts_float_input(self):
        """Should handle float32 input (values in [0, 1]) correctly."""
        batch = np.random.rand(3, 48, 48).astype(np.float32)
        result = apply_clahe_batch(batch)
        assert result.shape == (3, 48, 48)
        assert result.dtype == np.float32
