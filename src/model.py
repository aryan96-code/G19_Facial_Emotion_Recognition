"""
model.py
========
G19 Facial Emotion Recognition — CSE276

Purpose:
    Define the CNN architecture for 7-class facial emotion classification.

Architecture overview:
    Input → [Conv → BN → ReLU → MaxPool → Dropout] × 3 blocks
          → GlobalAveragePooling
          → Dense(256) → Dropout
          → Dense(7, softmax)

Design rationale:
    - Three convolutional blocks capture low-level edges, mid-level textures,
      and high-level facial feature patterns progressively.
    - Batch Normalisation stabilises training and allows higher learning rates.
    - Dropout (0.25–0.5) regularises to reduce overfitting on the small FER dataset.
    - GlobalAveragePooling reduces spatial dimensions without a large Flatten layer,
      lowering parameter count and improving generalisation.
    - Softmax output gives calibrated class probabilities.

Usage example:
    from src.model import build_cnn_model
    model = build_cnn_model()
    model.summary()
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.preprocessing import build_augmentation_layer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_SHAPE = (48, 48, 1)   # Greyscale 48 × 48
NUM_CLASSES = 7


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_cnn_model(
    input_shape: tuple = INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
    use_augmentation: bool = True,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """
    Build and compile the FER-2013 CNN model.

    Parameters
    ----------
    input_shape    : tuple  — (height, width, channels). Default (48, 48, 1).
    num_classes    : int    — Number of output classes. Default 7.
    use_augmentation : bool — Prepend random augmentation layers. Default True.
    learning_rate  : float  — Initial Adam learning rate. Default 1e-3.

    Returns
    -------
    keras.Model
        Compiled model ready for model.fit().
    """

    inputs = keras.Input(shape=input_shape, name="input_image")
    x = inputs

    # -----------------------------------------------------------------------
    # Optional data augmentation (active only during training)
    # -----------------------------------------------------------------------
    if use_augmentation:
        augmentation = build_augmentation_layer()
        x = augmentation(x)

    # -----------------------------------------------------------------------
    # Block 1 — detect low-level features (edges, gradients)
    # -----------------------------------------------------------------------
    x = layers.Conv2D(32, kernel_size=3, padding="same", name="conv1_1")(x)
    x = layers.BatchNormalization(name="bn1_1")(x)
    x = layers.Activation("relu", name="relu1_1")(x)

    x = layers.Conv2D(32, kernel_size=3, padding="same", name="conv1_2")(x)
    x = layers.BatchNormalization(name="bn1_2")(x)
    x = layers.Activation("relu", name="relu1_2")(x)

    x = layers.MaxPooling2D(pool_size=2, strides=2, name="pool1")(x)
    x = layers.Dropout(0.25, name="drop1")(x)

    # -----------------------------------------------------------------------
    # Block 2 — detect mid-level features (facial landmarks, shapes)
    # -----------------------------------------------------------------------
    x = layers.Conv2D(64, kernel_size=3, padding="same", name="conv2_1")(x)
    x = layers.BatchNormalization(name="bn2_1")(x)
    x = layers.Activation("relu", name="relu2_1")(x)

    x = layers.Conv2D(64, kernel_size=3, padding="same", name="conv2_2")(x)
    x = layers.BatchNormalization(name="bn2_2")(x)
    x = layers.Activation("relu", name="relu2_2")(x)

    x = layers.MaxPooling2D(pool_size=2, strides=2, name="pool2")(x)
    x = layers.Dropout(0.25, name="drop2")(x)

    # -----------------------------------------------------------------------
    # Block 3 — detect high-level features (expressions, emotions)
    # -----------------------------------------------------------------------
    x = layers.Conv2D(128, kernel_size=3, padding="same", name="conv3_1")(x)
    x = layers.BatchNormalization(name="bn3_1")(x)
    x = layers.Activation("relu", name="relu3_1")(x)

    x = layers.Conv2D(128, kernel_size=3, padding="same", name="conv3_2")(x)
    x = layers.BatchNormalization(name="bn3_2")(x)
    x = layers.Activation("relu", name="relu3_2")(x)

    x = layers.MaxPooling2D(pool_size=2, strides=2, name="pool3")(x)
    x = layers.Dropout(0.25, name="drop3")(x)

    # -----------------------------------------------------------------------
    # Classification head
    # -----------------------------------------------------------------------
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(256, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.5, name="drop_fc")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    # -----------------------------------------------------------------------
    # Build and compile
    # -----------------------------------------------------------------------
    model = keras.Model(inputs=inputs, outputs=outputs, name="FER_CNN")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ---------------------------------------------------------------------------
# Callbacks used during training
# ---------------------------------------------------------------------------

def get_callbacks(model_save_path: str = "models/best_model.keras") -> list:
    """
    Return a list of standard Keras callbacks for training.

    Callbacks
    ---------
    ModelCheckpoint : Saves the model only when validation loss improves.
    EarlyStopping   : Stops training when val_loss stops improving (patience=10).
    ReduceLROnPlateau : Halves the learning rate if val_loss plateaus (patience=5).

    Parameters
    ----------
    model_save_path : str
        Where to save the best model weights.

    Returns
    -------
    list of keras.callbacks.Callback
    """
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=model_save_path,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1,
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1,
    )

    return [checkpoint, early_stop, reduce_lr]


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = build_cnn_model()
    model.summary()
    print("\nmodel.py: build_cnn_model() ran successfully.")
