"""
evaluation.py
=============
G19 Facial Emotion Recognition — CSE276

Purpose:
    Evaluate a trained model and produce all assessment artefacts:
    - Classification report (precision, recall, F1 per class)
    - Confusion matrix heatmap
    - Training / validation loss and accuracy curves
    - Per-class accuracy bar chart

All plots are saved to the results/ directory; nothing is shown inline to
keep this module compatible with headless Colab cells.

Usage example:
    from src.evaluation import evaluate_model, plot_training_history
    evaluate_model(model, X_test, y_test)
    plot_training_history(history)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend for Colab / scripts
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

from src.data_loader import EMOTION_LABELS, NUM_CLASSES


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("results")
CLASS_NAMES = [EMOTION_LABELS[i] for i in range(NUM_CLASSES)]


def _ensure_results_dir():
    """Create results/ if it does not exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Run the complete evaluation pipeline on the test split.

    Steps
    -----
    1. Compute model predictions.
    2. Print a per-class classification report.
    3. Save a confusion matrix heatmap to results/.
    4. Save a per-class accuracy bar chart to results/.

    Parameters
    ----------
    model   : keras.Model — Trained model (loaded from models/).
    X_test  : np.ndarray  — Test images, shape (N, 48, 48, 1).
    y_test  : np.ndarray  — True class labels, shape (N,).

    Returns
    -------
    dict with keys:
        'report'     : sklearn classification_report as a dict
        'cm'         : confusion matrix as np.ndarray
        'test_loss'  : float
        'test_acc'   : float
    """
    _ensure_results_dir()

    print("[evaluation] Running model.evaluate() ...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[evaluation] Test Loss     : {test_loss:.4f}")
    print(f"[evaluation] Test Accuracy : {test_acc * 100:.2f}%")

    # --- Predictions ---
    y_probs = model.predict(X_test, verbose=0)
    y_pred  = np.argmax(y_probs, axis=1)

    # --- Classification report (all 7 classes) ---
    all_labels = list(range(NUM_CLASSES))
    print("\n[evaluation] Per-class Classification Report:")
    report_str = classification_report(
        y_test,
        y_pred,
        labels=all_labels,
        target_names=CLASS_NAMES,
        zero_division=0,
    )
    print(report_str)

    report_dict = classification_report(
        y_test,
        y_pred,
        labels=all_labels,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    # --- Confusion matrix ---
    cm = confusion_matrix(y_test, y_pred)
    _plot_confusion_matrix(cm)

    # --- Per-class accuracy bar chart ---
    _plot_per_class_accuracy(cm)

    return {
        "report"    : report_dict,
        "cm"        : cm,
        "test_loss" : test_loss,
        "test_acc"  : test_acc,
    }


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def _plot_confusion_matrix(cm: np.ndarray):
    """Save a normalised confusion matrix heatmap."""
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_title("Confusion Matrix (Normalised)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    save_path = RESULTS_DIR / "confusion_matrix.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[evaluation] Confusion matrix saved → {save_path}")


def _plot_per_class_accuracy(cm: np.ndarray):
    """Save a bar chart of per-class accuracy derived from the confusion matrix."""
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(CLASS_NAMES, per_class_acc, color="steelblue", edgecolor="black")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-class Accuracy", fontsize=14, fontweight="bold")
    ax.set_xlabel("Emotion", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.axhline(y=per_class_acc.mean(), color="red", linestyle="--", label="Mean")
    ax.legend()

    # Annotate bars
    for bar, acc in zip(bars, per_class_acc):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.01,
            f"{acc:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    save_path = RESULTS_DIR / "per_class_accuracy.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[evaluation] Per-class accuracy chart saved → {save_path}")


def plot_training_history(history) -> None:
    """
    Plot and save training vs validation loss and accuracy curves.

    Parameters
    ----------
    history : keras.callbacks.History
        The object returned by model.fit().
    """
    _ensure_results_dir()

    epochs = range(1, len(history.history["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(epochs, history.history["loss"],     label="Train Loss",      color="blue")
    ax1.plot(epochs, history.history["val_loss"], label="Val Loss",        color="orange")
    ax1.set_title("Loss over Epochs", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curve
    ax2.plot(epochs, history.history["accuracy"],     label="Train Accuracy", color="blue")
    ax2.plot(epochs, history.history["val_accuracy"], label="Val Accuracy",   color="orange")
    ax2.set_title("Accuracy over Epochs", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Training History — FER-2013 CNN", fontsize=14, fontweight="bold")
    plt.tight_layout()

    save_path = RESULTS_DIR / "training_history.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[evaluation] Training history chart saved → {save_path}")
