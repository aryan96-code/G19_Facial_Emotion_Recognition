"""
data_loader.py
==============
G19 Facial Emotion Recognition — CSE276

Purpose:
    Load the FER-2013 dataset from a CSV file, decode pixel strings into
    NumPy arrays, and return train / validation / test splits ready for
    model training.

FER-2013 CSV schema:
    emotion  : integer label (0–6)
    pixels   : space-separated string of 2304 pixel values (48 × 48 grayscale)
    Usage    : "Training" | "PublicTest" | "PrivateTest"

Usage example:
    from src.data_loader import load_fer2013
    X_train, y_train, X_val, y_val, X_test, y_test = load_fer2013("dataset/fer2013.csv")
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMOTION_LABELS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}

IMAGE_SIZE = 48          # FER-2013 images are 48 × 48 pixels
NUM_CLASSES = 7


# ---------------------------------------------------------------------------
# Helper: parse a pixel string → 48 × 48 NumPy array
# ---------------------------------------------------------------------------

def _parse_pixels(pixel_string: str) -> np.ndarray:
    """Convert a space-separated pixel string to a (48, 48) uint8 array."""
    pixels = np.array(pixel_string.split(), dtype=np.uint8)
    return pixels.reshape(IMAGE_SIZE, IMAGE_SIZE)


# ---------------------------------------------------------------------------
# Shared CSV helpers
# ---------------------------------------------------------------------------

def _read_fer2013_csv(csv_path: str) -> pd.DataFrame:
    """Validate and read a FER-2013 CSV file into a DataFrame."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Download fer2013.csv from Kaggle and place it in the dataset/ folder."
        )

    print(f"[data_loader] Loading dataset from: {path}")
    df = pd.read_csv(path)

    required_columns = {"emotion", "pixels", "Usage"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    return df


def _process_split(split_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Parse pixels and normalise to float32 [0, 1] with shape (N, 48, 48, 1)."""
    images = np.stack(split_df["pixels"].apply(_parse_pixels).values)
    images = images.astype(np.float32) / 255.0
    images = images[..., np.newaxis]           # add channel dim → (N, 48, 48, 1)
    labels = split_df["emotion"].values.astype(np.int32)
    if not np.all((labels >= 0) & (labels < NUM_CLASSES)):
        raise ValueError(f"Labels must be in [0, {NUM_CLASSES - 1}], got {np.unique(labels)}")
    return images, labels


# ---------------------------------------------------------------------------
# Main loaders
# ---------------------------------------------------------------------------

def load_fer2013(csv_path: str) -> tuple:
    """
    Load FER-2013 from a CSV file and return six NumPy arrays.

    Parameters
    ----------
    csv_path : str
        Path to 'fer2013.csv'.

    Returns
    -------
    X_train, y_train : training images and labels
    X_val,   y_val   : validation images and labels  (PublicTest split)
    X_test,  y_test  : test images and labels        (PrivateTest split)

    Each X array has shape  (N, 48, 48, 1)  — float32 in [0, 1].
    Each y array has shape  (N,)            — integer class index.

    Raises
    ------
    FileNotFoundError : if the CSV does not exist at csv_path.
    ValueError        : if the CSV is missing expected columns.
    """
    df = _read_fer2013_csv(csv_path)

    # --- Split by the 'Usage' column (preserve original FER-2013 splits) ---
    # Training    -> model training
    # PublicTest  -> validation / hyperparameter tuning / early stopping
    # PrivateTest -> final held-out testing ONLY
    train_df = df[df["Usage"] == "Training"].reset_index(drop=True)
    val_df   = df[df["Usage"] == "PublicTest"].reset_index(drop=True)
    test_df  = df[df["Usage"] == "PrivateTest"].reset_index(drop=True)

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError(
            f"Expected non-empty FER-2013 splits, got "
            f"Training={len(train_df)}, PublicTest={len(val_df)}, PrivateTest={len(test_df)}."
        )

    print(f"[data_loader] Training samples   (Training)   : {len(train_df)}")
    print(f"[data_loader] Validation samples (PublicTest) : {len(val_df)}")
    print(f"[data_loader] Test samples       (PrivateTest): {len(test_df)}")

    X_train, y_train = _process_split(train_df)
    X_val,   y_val   = _process_split(val_df)
    X_test,  y_test  = _process_split(test_df)

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_fer2013_train_val(csv_path: str) -> tuple:
    """
    Load ONLY the Training and PublicTest splits from FER-2013.

    PrivateTest rows are never decoded, materialized, or returned, guaranteeing
    strict isolation of the held-out test set during model development and tuning.

    Returns
    -------
    X_train, y_train, X_val, y_val
    """
    df = _read_fer2013_csv(csv_path)

    train_df = df[df["Usage"] == "Training"].reset_index(drop=True)
    val_df   = df[df["Usage"] == "PublicTest"].reset_index(drop=True)

    if len(train_df) == 0 or len(val_df) == 0:
        raise ValueError(
            f"Expected non-empty Training and PublicTest splits, got "
            f"Training={len(train_df)}, PublicTest={len(val_df)}."
        )

    print(f"[data_loader] Training samples   (Training)   : {len(train_df)}")
    print(f"[data_loader] Validation samples (PublicTest) : {len(val_df)}")

    X_train, y_train = _process_split(train_df)
    X_val,   y_val   = _process_split(val_df)

    return X_train, y_train, X_val, y_val


# ---------------------------------------------------------------------------
# Utility: label index → human-readable string
# ---------------------------------------------------------------------------

def index_to_label(index: int) -> str:
    """Return the emotion name for a given class index."""
    return EMOTION_LABELS.get(index, f"Unknown({index})")


def compute_training_class_weights(y_train: np.ndarray) -> dict[int, float]:
    """
    Compute balanced class weights using ONLY the training labels.

    Parameters
    ----------
    y_train : np.ndarray
        Integer training labels in [0, NUM_CLASSES - 1].

    Returns
    -------
    dict[int, float]
        Mapping from class index (0..6) to balanced weight.
    """
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    return {int(cls): float(w) for cls, w in zip(classes, weights)}


def compute_smoothed_class_weights(
    y_train: np.ndarray,
    power: float = 0.5,
) -> dict[int, float]:
    """
    Compute smoothed balanced class weights from ONLY the training labels.

    Raises each balanced weight to `power` (default 0.5 = square-root smoothing)
    and normalises so the mean weight across classes is 1.0, reducing extreme
    gradients on rare classes while still addressing imbalance.

    Parameters
    ----------
    y_train : np.ndarray
        Integer training labels in [0, NUM_CLASSES - 1].
    power : float, optional
        Exponent applied to balanced weights. Default 0.5 (square root).

    Returns
    -------
    dict[int, float]
        Mapping from class index (0..6) to smoothed weight.
    """
    raw_weights = compute_training_class_weights(y_train)
    smoothed = {cls: w ** power for cls, w in raw_weights.items()}
    mean_w = sum(smoothed.values()) / len(smoothed)
    return {cls: float(w / mean_w) for cls, w in smoothed.items()}


# ---------------------------------------------------------------------------
# Quick sanity check (run as script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else "dataset/fer2013.csv"
    X_train, y_train, X_val, y_val, X_test, y_test = load_fer2013(csv)
    print(f"\nX_train shape: {X_train.shape}, dtype: {X_train.dtype}")
    print(f"y_train shape: {y_train.shape}, unique labels: {np.unique(y_train)}")
    print(f"Pixel value range: [{X_train.min():.3f}, {X_train.max():.3f}]")
