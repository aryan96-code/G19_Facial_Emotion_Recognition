"""
utils.py
========
G19 Facial Emotion Recognition — CSE276

Purpose:
    Shared helper utilities used across notebooks, training scripts, and the
    Streamlit app.  Keeps duplicated helper code in one maintainable place.

Contents:
    - set_random_seeds()          : reproducibility across numpy/tensorflow
    - display_sample_images()     : visualise a grid of dataset samples
    - plot_class_distribution()   : bar chart of label frequencies
    - load_model_safe()           : load a .keras / .h5 model with error handling
    - get_project_root()          : resolve the project root directory
"""

import os
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_loader import EMOTION_LABELS, NUM_CLASSES


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and TensorFlow to ensure reproducibility.

    Parameters
    ----------
    seed : int
        The seed value.  Default is 42.

    Note
    ----
    GPU operations in TensorFlow may still introduce minor non-determinism.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        print(f"[utils] Random seeds set to {seed} (Python, NumPy, TensorFlow).")
    except ImportError:
        print(f"[utils] Random seeds set to {seed} (Python, NumPy). TensorFlow not found.")


# ---------------------------------------------------------------------------
# Dataset visualisation helpers
# ---------------------------------------------------------------------------

def display_sample_images(
    images: np.ndarray,
    labels: np.ndarray,
    n_per_class: int = 3,
    save_path: str = "results/sample_images.png",
) -> None:
    """
    Display a grid of sample images — n_per_class samples per emotion class.

    Parameters
    ----------
    images      : np.ndarray — shape (N, 48, 48) or (N, 48, 48, 1)
    labels      : np.ndarray — shape (N,), integer class indices
    n_per_class : int        — how many samples to show per emotion (default 3)
    save_path   : str        — where to save the resulting PNG
    """
    # Squeeze channel dimension if present
    if images.ndim == 4:
        images = images[:, :, :, 0]

    n_classes = NUM_CLASSES
    fig, axes = plt.subplots(
        n_classes, n_per_class, figsize=(n_per_class * 2, n_classes * 2)
    )

    for class_idx in range(n_classes):
        # Find indices where label == class_idx
        class_indices = np.where(labels == class_idx)[0]

        if len(class_indices) == 0:
            continue

        # Randomly sample n_per_class images for this class
        chosen = np.random.choice(class_indices, size=min(n_per_class, len(class_indices)), replace=False)

        for col, img_idx in enumerate(chosen):
            ax = axes[class_idx][col]
            ax.imshow(images[img_idx], cmap="gray")
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(EMOTION_LABELS[class_idx], fontsize=8, rotation=0, labelpad=40)

    plt.suptitle("Sample Images by Emotion Class", fontsize=12, fontweight="bold")
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"[utils] Sample images saved → {save_path}")


def plot_class_distribution(
    labels: np.ndarray,
    split_name: str = "Training",
    save_path: str = "results/class_distribution.png",
) -> None:
    """
    Plot and save a bar chart of class label frequencies.

    Parameters
    ----------
    labels     : np.ndarray — shape (N,) integer class indices
    split_name : str        — label for the chart title (e.g. "Training")
    save_path  : str        — output PNG path
    """
    class_names = [EMOTION_LABELS[i] for i in range(NUM_CLASSES)]
    counts = [np.sum(labels == i) for i in range(NUM_CLASSES)]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(class_names, counts, color="steelblue", edgecolor="black")
    ax.set_title(f"Class Distribution — {split_name} Set", fontsize=13, fontweight="bold")
    ax.set_xlabel("Emotion", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 10,
            str(count),
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"[utils] Class distribution chart saved → {save_path}")


# ---------------------------------------------------------------------------
# Safe model loader
# ---------------------------------------------------------------------------

def load_model_safe(model_path: str):
    """
    Load a Keras model from a .keras or .h5 file with friendly error messages.

    Parameters
    ----------
    model_path : str
        Path to the saved model file.

    Returns
    -------
    keras.Model

    Raises
    ------
    FileNotFoundError : if the model file does not exist.
    RuntimeError      : if the model fails to load for any other reason.
    """
    from pathlib import Path

    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found at '{path}'.\n"
            "Train the model using notebook 04_cnn_training.ipynb and download "
            "the saved file to the models/ directory."
        )

    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(str(path))
        print(f"[utils] Model loaded successfully from: {path}")
        return model
    except Exception as exc:
        raise RuntimeError(f"Failed to load model from '{path}'.\nDetails: {exc}") from exc


# ---------------------------------------------------------------------------
# Project root resolver
# ---------------------------------------------------------------------------

def get_project_root() -> Path:
    """
    Return the absolute path of the project root directory.

    Assumes this file is located at: <project_root>/src/utils.py
    """
    return Path(__file__).resolve().parent.parent
