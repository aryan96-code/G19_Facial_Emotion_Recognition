"""
face_detector.py
================
G19 Facial Emotion Recognition — CSE276

Purpose:
    Reusable face detection and cropping module for real-world images
    (uploads and webcam frames) before feeding into the FER model.

Key Features:
    - Uses OpenCV's Haar Cascade classifier (`haarcascade_frontalface_default.xml`).
    - Robust error handling for unavailable classifiers or missing cascade files.
    - Automatic selection of the primary face (largest bounding box by area).
    - Configurable margin padding (~10%) with boundary clipping.
    - Annotates all detected faces on a copy of the input image.
    - Never silently falls back to classifying the full uncropped image.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import os
from pathlib import Path
import cv2
import numpy as np


class FaceDetectorError(RuntimeError):
    """Raised when face detection infrastructure (classifier, model) is unavailable."""
    pass


@dataclass
class FaceDetectionResult:
    """Structured result of face detection on an input image."""
    cropped_face: np.ndarray | None
    detected: bool
    bbox: tuple[int, int, int, int] | None       # Padded bbox: (x1, y1, x2, y2)
    raw_bbox: tuple[int, int, int, int] | None   # Unpadded bbox: (x, y, w, h)
    num_faces: int
    annotated_image: np.ndarray
    error_message: str | None = None


def get_haar_cascade_path() -> str:
    """
    Resolve the absolute path to OpenCV's default frontal face Haar Cascade.

    Returns
    -------
    str
        Path to 'haarcascade_frontalface_default.xml'.

    Raises
    ------
    FaceDetectorError
        If cv2.data is missing or the XML file does not exist.
    """
    data_dir = getattr(getattr(cv2, "data", None), "haarcascades", None)
    if not data_dir:
        raise FaceDetectorError(
            "cv2.data.haarcascades is not available. "
            "Please ensure 'opencv-python<5.0.0' is installed."
        )

    xml_path = os.path.join(data_dir, "haarcascade_frontalface_default.xml")
    if not os.path.isfile(xml_path):
        raise FaceDetectorError(
            f"Haar cascade XML not found at '{xml_path}'. "
            "Please verify OpenCV installation (opencv-python<5.0.0)."
        )
    return xml_path


def load_face_cascade(cascade_path: str | None = None) -> cv2.CascadeClassifier:
    """
    Load and validate an OpenCV CascadeClassifier for frontal face detection.

    Parameters
    ----------
    cascade_path : str, optional
        Custom path to cascade XML. If None, resolves via get_haar_cascade_path().

    Returns
    -------
    cv2.CascadeClassifier

    Raises
    ------
    FaceDetectorError
        If CascadeClassifier class is missing or fails to load the cascade XML.
    """
    if not hasattr(cv2, "CascadeClassifier"):
        raise FaceDetectorError(
            "cv2.CascadeClassifier is not available in the installed OpenCV package. "
            "Please install 'opencv-python>=4.8.0,<5.0.0'."
        )

    resolved_path = cascade_path or get_haar_cascade_path()
    cascade = cv2.CascadeClassifier(resolved_path)

    if cascade.empty():
        raise FaceDetectorError(
            f"Failed to load Haar Cascade from '{resolved_path}'. The classifier is empty."
        )
    return cascade


def compute_padded_bbox(
    x: int,
    y: int,
    w: int,
    h: int,
    img_width: int,
    img_height: int,
    padding_ratio: float = 0.10,
) -> tuple[int, int, int, int]:
    """
    Add proportional padding around a bounding box, clipped to image boundaries.

    Parameters
    ----------
    x, y, w, h : int
        Original bounding box coordinates and dimensions.
    img_width, img_height : int
        Image width and height for boundary clipping.
    padding_ratio : float
        Fraction of width and height to add on each side (default 0.10 = 10%).

    Returns
    -------
    tuple[int, int, int, int]
        Clipped padded coordinates (x1, y1, x2, y2).
    """
    pad_w = int(round(w * padding_ratio))
    pad_h = int(round(h * padding_ratio))

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(img_width, x + w + pad_w)
    y2 = min(img_height, y + h + pad_h)

    return x1, y1, x2, y2


def detect_and_crop_primary_face(
    image: np.ndarray,
    cascade: cv2.CascadeClassifier | None = None,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_size: tuple[int, int] = (30, 30),
    padding_ratio: float = 0.10,
) -> FaceDetectionResult:
    """
    Detect frontal faces, select the largest face, and return a padded crop.

    Parameters
    ----------
    image : np.ndarray
        Input image in BGR or greyscale format (uint8).
    cascade : cv2.CascadeClassifier, optional
        Pre-loaded classifier. If None, loaded via load_face_cascade().
    scale_factor : float
        Parameter specifying how much image size is reduced at each image scale.
    min_neighbors : int
        Parameter specifying how many neighbors each candidate rectangle should have.
    min_size : tuple[int, int]
        Minimum possible face size. Smaller faces are ignored.
    padding_ratio : float
        Proportional padding around the detected face (default 0.10 = ~10%).

    Returns
    -------
    FaceDetectionResult
        Named tuple/dataclass containing:
        - cropped_face: cropped uint8 BGR/greyscale sub-image, or None if no face found.
        - detected: bool indicating if at least one face was detected.
        - bbox: padded bounding box (x1, y1, x2, y2) or None.
        - raw_bbox: unpadded bounding box (x, y, w, h) or None.
        - num_faces: total count of detected faces.
        - annotated_image: copy of input image with drawn bounding boxes.
        - error_message: str description if detection could not proceed.

    Raises
    ------
    ValueError
        If the image is None, empty, or has an invalid shape.
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("detect_and_crop_primary_face received an empty or invalid image array.")

    if image.ndim not in (2, 3):
        raise ValueError(f"Image must have 2 or 3 dimensions, got {image.ndim}.")

    annotated = image.copy()
    img_h, img_w = image.shape[:2]

    # Convert to greyscale for Haar Cascade detection if needed
    if image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 1:
        gray = image[:, :, 0]
    else:
        gray = image

    # Load cascade if not supplied
    try:
        classifier = cascade if cascade is not None else load_face_cascade()
    except FaceDetectorError as err:
        return FaceDetectionResult(
            cropped_face=None,
            detected=False,
            bbox=None,
            raw_bbox=None,
            num_faces=0,
            annotated_image=annotated,
            error_message=str(err),
        )

    # Detect faces
    faces = classifier.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
        flags=getattr(cv2, "CASCADE_SCALE_IMAGE", 0),
    )

    if faces is None or len(faces) == 0:
        return FaceDetectionResult(
            cropped_face=None,
            detected=False,
            bbox=None,
            raw_bbox=None,
            num_faces=0,
            annotated_image=annotated,
            error_message=None,
        )

    # Convert faces to list of (x, y, w, h) tuples
    face_list = [tuple(map(int, f)) for f in faces]

    # Select primary face by maximum bounding-box area (w * h)
    primary_face = max(face_list, key=lambda b: b[2] * b[3])
    px, py, pw, ph = primary_face

    # Draw bounding boxes on annotated image
    for (fx, fy, fw, fh) in face_list:
        if (fx, fy, fw, fh) == primary_face:
            # Primary face: Green bounding box (BGR: 0, 255, 0)
            cv2.rectangle(annotated, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                "Primary Face",
                (fx, max(0, fy - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        else:
            # Secondary faces: Blue/Grey bounding box (BGR: 200, 150, 50)
            cv2.rectangle(annotated, (fx, fy), (fx + fw, fy + fh), (200, 150, 50), 1)
            cv2.putText(
                annotated,
                "Secondary",
                (fx, max(0, fy - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 150, 50),
                1,
                cv2.LINE_AA,
            )

    # Compute padded crop for the primary face
    x1, y1, x2, y2 = compute_padded_bbox(
        px, py, pw, ph,
        img_width=img_w,
        img_height=img_h,
        padding_ratio=padding_ratio,
    )

    cropped_face = image[y1:y2, x1:x2].copy()

    return FaceDetectionResult(
        cropped_face=cropped_face,
        detected=True,
        bbox=(x1, y1, x2, y2),
        raw_bbox=(px, py, pw, ph),
        num_faces=len(face_list),
        annotated_image=annotated,
        error_message=None,
    )
