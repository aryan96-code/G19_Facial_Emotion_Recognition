"""
test_face_detector.py
=====================
Unit tests for src/face_detector.py.

Verifies:
- compute_padded_bbox calculation and edge boundary clipping.
- Input validation (None, empty, invalid dimensions).
- Error handling when Haar Cascade or CascadeClassifier is unavailable.
- Detection, primary face selection (largest area), and annotation with mock cascades.
- Output compatibility with the existing preprocessing pipeline (preprocess_image).
"""

from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src.face_detector import (
    FaceDetectorError,
    FaceDetectionResult,
    compute_padded_bbox,
    detect_and_crop_primary_face,
    get_haar_cascade_path,
    load_face_cascade,
)
from src.preprocessing import preprocess_image


# ===========================================================================
# 1. Padded Bounding Box Calculation & Clipping
# ===========================================================================

class TestComputePaddedBbox:
    """Tests for compute_padded_bbox()."""

    def test_standard_padding(self):
        """10% padding on a 100x100 box well inside a 500x500 image."""
        # x=100, y=100, w=100, h=100 -> pad_w=10, pad_h=10
        # Expected: x1=90, y1=90, x2=210, y2=210
        x1, y1, x2, y2 = compute_padded_bbox(
            x=100, y=100, w=100, h=100,
            img_width=500, img_height=500,
            padding_ratio=0.10,
        )
        assert (x1, y1, x2, y2) == (90, 90, 210, 210)

    def test_clipping_at_top_left_boundaries(self):
        """Bounding box near top-left should clip coordinates at 0."""
        # x=5, y=5, w=50, h=50 -> pad=5 -> x1=0, y1=0
        x1, y1, x2, y2 = compute_padded_bbox(
            x=5, y=5, w=50, h=50,
            img_width=200, img_height=200,
            padding_ratio=0.20,
        )
        assert x1 == 0
        assert y1 == 0
        assert x2 == 65
        assert y2 == 65

    def test_clipping_at_bottom_right_boundaries(self):
        """Bounding box near bottom-right should clip at img_width/img_height."""
        x1, y1, x2, y2 = compute_padded_bbox(
            x=180, y=180, w=40, h=40,
            img_width=200, img_height=200,
            padding_ratio=0.25,  # pad=10 -> x2=230 -> clips to 200
        )
        assert x1 == 170
        assert y1 == 170
        assert x2 == 200
        assert y2 == 200

    def test_zero_padding_ratio(self):
        """0.0 padding ratio should return identical box."""
        x1, y1, x2, y2 = compute_padded_bbox(
            x=30, y=40, w=60, h=70,
            img_width=200, img_height=200,
            padding_ratio=0.0,
        )
        assert (x1, y1, x2, y2) == (30, 40, 90, 110)


# ===========================================================================
# 2. Input Validation
# ===========================================================================

class TestFaceDetectorInputValidation:
    """Tests for input argument validation in detect_and_crop_primary_face()."""

    def test_raises_on_none_image(self):
        with pytest.raises(ValueError, match="empty or invalid image"):
            detect_and_crop_primary_face(None)

    def test_raises_on_empty_array(self):
        empty_img = np.array([], dtype=np.uint8)
        with pytest.raises(ValueError, match="empty or invalid image"):
            detect_and_crop_primary_face(empty_img)

    def test_raises_on_invalid_dimensions(self):
        # 1D array
        with pytest.raises(ValueError, match="Image must have 2 or 3 dimensions"):
            detect_and_crop_primary_face(np.zeros((100,), dtype=np.uint8))

        # 4D array
        with pytest.raises(ValueError, match="Image must have 2 or 3 dimensions"):
            detect_and_crop_primary_face(np.zeros((1, 48, 48, 1), dtype=np.uint8))


# ===========================================================================
# 3. Cascade Loading & Error Handling
# ===========================================================================

class TestCascadeLoadingAndErrors:
    """Tests for error handling when cascade classifier is missing or broken."""

    def test_missing_cv2_data_haarcascades(self):
        with patch("src.face_detector.cv2") as mock_cv2:
            mock_cv2.data = None
            with pytest.raises(FaceDetectorError, match="cv2.data.haarcascades is not available"):
                get_haar_cascade_path()

    def test_missing_xml_file_raises_error(self, tmp_path):
        with patch("src.face_detector.cv2") as mock_cv2:
            mock_cv2.data.haarcascades = str(tmp_path)
            with pytest.raises(FaceDetectorError, match="Haar cascade XML not found"):
                get_haar_cascade_path()

    def test_load_face_cascade_missing_classifier_attr(self):
        with patch("src.face_detector.cv2", spec=[]):
            with pytest.raises(FaceDetectorError, match="cv2.CascadeClassifier is not available"):
                load_face_cascade()

    def test_detect_and_crop_gracefully_returns_error_on_missing_cascade(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        with patch("src.face_detector.load_face_cascade", side_effect=FaceDetectorError("Cascade not found")):
            result = detect_and_crop_primary_face(img)
            assert result.detected is False
            assert result.cropped_face is None
            assert result.num_faces == 0
            assert "Cascade not found" in result.error_message


# ===========================================================================
# 4. Face Detection & Primary Selection Logic
# ===========================================================================

class TestDetectionLogicWithMockClassifier:
    """Tests for face detection, crop geometry, and primary face selection."""

    def test_no_face_detected(self):
        mock_cascade = MagicMock()
        mock_cascade.detectMultiScale.return_value = ()

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = detect_and_crop_primary_face(img, cascade=mock_cascade)

        assert isinstance(result, FaceDetectionResult)
        assert result.detected is False
        assert result.cropped_face is None
        assert result.bbox is None
        assert result.raw_bbox is None
        assert result.num_faces == 0
        assert result.error_message is None
        assert result.annotated_image.shape == img.shape

    def test_single_face_detected_and_cropped(self):
        mock_cascade = MagicMock()
        # Single face at (20, 20) with width=40, height=40
        mock_cascade.detectMultiScale.return_value = np.array([[20, 20, 40, 40]])

        # 100x100 BGR image with distinct pixel values
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = detect_and_crop_primary_face(img, cascade=mock_cascade, padding_ratio=0.10)

        assert result.detected is True
        assert result.num_faces == 1
        assert result.raw_bbox == (20, 20, 40, 40)

        # pad_w = round(40 * 0.10) = 4 -> x1=16, y1=16, x2=64, y2=64 -> 48x48
        assert result.bbox == (16, 16, 64, 64)
        assert result.cropped_face is not None
        assert result.cropped_face.shape == (48, 48, 3)

    def test_multiple_faces_selects_largest_by_area(self):
        mock_cascade = MagicMock()
        # Three faces:
        # Face 1: area = 30 * 30 = 900
        # Face 2: area = 50 * 50 = 2500 (largest primary)
        # Face 3: area = 20 * 20 = 400
        mock_cascade.detectMultiScale.return_value = np.array([
            [10, 10, 30, 30],
            [60, 60, 50, 50],
            [150, 150, 20, 20],
        ])

        img = np.zeros((250, 250, 3), dtype=np.uint8)
        result = detect_and_crop_primary_face(img, cascade=mock_cascade)

        assert result.detected is True
        assert result.num_faces == 3
        # Primary face must be the second face (area 2500)
        assert result.raw_bbox == (60, 60, 50, 50)
        assert result.cropped_face is not None
        assert result.cropped_face.shape[0] > 0
        assert result.cropped_face.shape[1] > 0

    def test_greyscale_input_image_handling(self):
        mock_cascade = MagicMock()
        mock_cascade.detectMultiScale.return_value = np.array([[10, 10, 30, 30]])

        # 2D greyscale image
        img_gray = np.full((100, 100), 100, dtype=np.uint8)
        result = detect_and_crop_primary_face(img_gray, cascade=mock_cascade)

        assert result.detected is True
        assert result.cropped_face.ndim == 2


# ===========================================================================
# 5. End-to-End Pipeline Compatibility
# ===========================================================================

class TestPipelineCompatibility:
    """Tests verifying that cropped faces seamlessly feed into preprocess_image()."""

    def test_cropped_face_passes_to_preprocess_image(self):
        mock_cascade = MagicMock()
        mock_cascade.detectMultiScale.return_value = np.array([[30, 30, 80, 80]])

        # 200x200 BGR image
        img = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        result = detect_and_crop_primary_face(img, cascade=mock_cascade)

        assert result.detected is True
        assert result.cropped_face is not None

        # Pass cropped face to the standard inference preprocessor
        tensor = preprocess_image(result.cropped_face, use_clahe=False)

        # Model input requirements: (1, 48, 48, 1), float32, [0.0, 1.0]
        assert tensor.shape == (1, 48, 48, 1)
        assert tensor.dtype == np.float32
        assert 0.0 <= tensor.min() <= tensor.max() <= 1.0
