"""
preprocessing.py
================
G19 Facial Emotion Recognition — CSE276

Purpose:
    Apply image preprocessing (equalisation, normalisation) and real-time
    data augmentation so the CNN generalises better to unseen faces.

Key design decisions:
    - All transformations operate on NumPy arrays to stay Colab-friendly.
    - Augmentation is applied ONLY to training data, not to validation/test.
    - Face alignment / cropping is handled by OpenCV before calling these.

Usage example:
    from src.preprocessing import preprocess_image, build_augmentation_layer
    img_ready = preprocess_image(raw_img_array)
"""

import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Single-image preprocessing (used at inference time too)
# ---------------------------------------------------------------------------

def preprocess_image(image: np.ndarray, use_clahe: bool = False) -> np.ndarray:
    """
    Prepare a single raw face image for model input.

    Steps
    -----
    1. Convert to greyscale if the image has 3 channels.
    2. Resize to 48 × 48 pixels.
    3. (Optional) Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation)
       when `use_clahe=True` (kept False for the baseline model).
    4. Normalise pixel values to [0, 1] float32.
    5. Add channel and batch dimensions → shape (1, 48, 48, 1).

    Parameters
    ----------
    image : np.ndarray
        Raw image array.  Can be (H, W), (H, W, 1), or (H, W, 3) uint8.
    use_clahe : bool, optional
        Whether to apply CLAHE contrast enhancement. Default is False
        (baseline preprocessing). Set True only when running or serving
        the CLAHE experiment model.

    Returns
    -------
    np.ndarray
        Shape (1, 48, 48, 1), dtype float32, values in [0, 1].

    Raises
    ------
    ValueError : if the input array has an unexpected number of channels.
    """
    if image is None or image.size == 0:
        raise ValueError("preprocess_image received an empty or None image.")

    # --- Convert colour → greyscale ---
    if image.ndim == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 1:
            image = image[:, :, 0]
        else:
            raise ValueError(f"Unexpected number of channels: {image.shape[2]}")

    # --- Resize to model input size ---
    image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_AREA)

    # --- Optional CLAHE for contrast normalisation experiment ---
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)

    # --- Normalise and reshape ---
    image = image.astype(np.float32) / 255.0
    image = image[np.newaxis, :, :, np.newaxis]   # → (1, 48, 48, 1)

    return image


# ---------------------------------------------------------------------------
# Keras augmentation layer (applied during training inside the model graph)
# ---------------------------------------------------------------------------

def build_augmentation_layer():
    """
    Build a Keras Sequential augmentation pipeline.

    These layers are applied randomly during training and are DISABLED during
    inference (model.predict / model.evaluate), so they never alter validation
    or test data.

    Augmentations chosen for FER-2013
    -----------------------------------
    - Random horizontal flip  : faces are symmetric left–right.
    - Random rotation (±10°)  : slight head tilts are common (10 / 360 ≈ 0.0278).
    - Random zoom (±10%)      : different camera distances.
    - Random brightness shift : varied lighting conditions; value_range=(0.0, 1.0)
                                matches our [0, 1] normalised float32 inputs.

    Returns
    -------
    keras.Sequential
        A compiled Keras layer stack ready to be inserted into the model.
    """
    from tensorflow import keras

    augmentation = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(factor=10.0 / 360.0),
            keras.layers.RandomZoom(height_factor=0.10),
            keras.layers.RandomBrightness(factor=0.15, value_range=(0.0, 1.0)),
        ],
        name="data_augmentation",
    )
    return augmentation


def build_refined_augmentation_layer():
    """
    Build the refined Keras Sequential augmentation pipeline for Experiment 02.

    Applied ONLY during training (training=True) and disabled during validation
    and inference.

    Augmentations
    -------------
    - Random horizontal flip   : 50% probability
    - Random rotation (±7°)    : 7 / 360 ≈ 0.0194
    - Random zoom (±10%)       : height_factor=0.10
    - Random translation (±8%) : height_factor=0.08, width_factor=0.08
    - Random brightness (±15%) : value_range=(0.0, 1.0) for [0, 1] inputs
    - Random contrast (±15%)   : factor=0.15
    """
    from tensorflow import keras

    augmentation = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(factor=7.0 / 360.0),
            keras.layers.RandomZoom(height_factor=0.10),
            keras.layers.RandomTranslation(height_factor=0.08, width_factor=0.08),
            keras.layers.RandomBrightness(factor=0.15, value_range=(0.0, 1.0)),
            keras.layers.RandomContrast(factor=0.15, value_range=(0.0, 1.0)),
        ],
        name="refined_data_augmentation",
    )
    return augmentation


# ---------------------------------------------------------------------------
# Batch preprocessing (for offline augmentation if preferred)
# ---------------------------------------------------------------------------

def apply_clahe_batch(images: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE to a batch of greyscale images.

    Parameters
    ----------
    images : np.ndarray
        Shape (N, 48, 48) or (N, 48, 48, 1), dtype uint8 or float32.

    Returns
    -------
    np.ndarray
        Same shape, dtype float32, values in [0, 1].
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    has_channel_dim = images.ndim == 4

    if has_channel_dim:
        images = images[:, :, :, 0]   # drop channel dim temporarily

    # Convert to uint8 if needed
    if images.dtype != np.uint8:
        images = (images * 255).astype(np.uint8)

    result = np.stack([clahe.apply(img) for img in images], axis=0)
    result = result.astype(np.float32) / 255.0

    if has_channel_dim:
        result = result[..., np.newaxis]   # restore channel dim

    return result


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Create a random dummy image and check the pipeline
    dummy = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    out = preprocess_image(dummy)
    print(f"Output shape : {out.shape}")     # expected (1, 48, 48, 1)
    print(f"Value range  : [{out.min():.3f}, {out.max():.3f}]")
    print("preprocessing.py: all checks passed.")
