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

from src.preprocessing import build_augmentation_layer, build_refined_augmentation_layer


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
# Experiment 02 — Improved 4-Block CNN & Callbacks (Baseline preserved above)
# ---------------------------------------------------------------------------

def _conv_bn_relu_block(x, filters: int, dropout_rate: float, block_idx: int):
    """Apply two 3x3 Conv-BN-ReLU layers, MaxPooling2D(2), and Dropout."""
    for sub in (1, 2):
        x = layers.Conv2D(
            filters, kernel_size=3, padding="same",
            name=f"exp02_conv{block_idx}_{sub}",
        )(x)
        x = layers.BatchNormalization(name=f"exp02_bn{block_idx}_{sub}")(x)
        x = layers.Activation("relu", name=f"exp02_relu{block_idx}_{sub}")(x)

    x = layers.MaxPooling2D(pool_size=2, strides=2, name=f"exp02_pool{block_idx}")(x)
    x = layers.Dropout(dropout_rate, name=f"exp02_drop{block_idx}")(x)
    return x


def build_improved_cnn_model(
    input_shape: tuple = INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
    use_augmentation: bool = True,
    augmentation_type: str = "baseline",
    head_type: str = "A",
    learning_rate: float = 1e-3,
) -> keras.Model:
    """
    Build and compile the Experiment 02 4-block CNN for FER-2013.

    Parameters
    ----------
    input_shape : tuple
        Input tensor shape. Default (48, 48, 1).
    num_classes : int
        Number of emotion classes. Default 7.
    use_augmentation : bool
        Prepend training-only augmentation layers. Default True.
    augmentation_type : {"baseline", "refined"}
        Which augmentation pipeline to prepend when `use_augmentation=True`.
    head_type : {"A", "B"}
        "A" = Flatten -> Dense(256, use_bias=False) -> BN -> ReLU -> Dropout(0.45) -> Softmax
        "B" = [GlobalAvgPool2D, GlobalMaxPool2D] -> Concatenate -> Dense(256) -> BN -> ReLU -> Dropout(0.45) -> Softmax
    learning_rate : float
        Initial Adam learning rate. Default 1e-3.

    Returns
    -------
    keras.Model
        Compiled Experiment 02 model.
    """
    head_key = head_type.upper()
    if head_key not in {"A", "B"}:
        raise ValueError(f"Unsupported head_type '{head_type}'. Choose 'A' or 'B'.")

    aug_key = augmentation_type.lower()
    if aug_key not in {"baseline", "refined"}:
        raise ValueError(
            f"Unsupported augmentation_type '{augmentation_type}'. Choose 'baseline' or 'refined'."
        )

    inputs = keras.Input(shape=input_shape, name="input_image")
    x = inputs

    if use_augmentation:
        aug_layer = (
            build_augmentation_layer()
            if aug_key == "baseline"
            else build_refined_augmentation_layer()
        )
        x = aug_layer(x)

    # Four convolutional blocks: 32 -> 64 -> 128 -> 256 (padding="same")
    for idx, (filters, drop) in enumerate(
        [(32, 0.15), (64, 0.20), (128, 0.25), (256, 0.25)], start=1
    ):
        x = _conv_bn_relu_block(x, filters=filters, dropout_rate=drop, block_idx=idx)

    # Configurable classification head
    if head_key == "A":
        x = layers.Flatten(name="exp02_flatten")(x)
        x = layers.Dense(256, use_bias=False, name="exp02_fc1")(x)
    else:
        avg_pool = layers.GlobalAveragePooling2D(name="exp02_gap")(x)
        max_pool = layers.GlobalMaxPooling2D(name="exp02_gmp")(x)
        x = layers.Concatenate(name="exp02_pool_concat")([avg_pool, max_pool])
        x = layers.Dense(256, name="exp02_fc1")(x)

    x = layers.BatchNormalization(name="exp02_bn_fc")(x)
    x = layers.Activation("relu", name="exp02_relu_fc")(x)
    x = layers.Dropout(0.45, name="exp02_drop_fc")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=f"FER_Improved_CNN_Head{head_key}",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_exp02_callbacks(
    local_save_path: str = "models/exp02_best_model.keras",
    drive_save_path: str | None = None,
) -> list:
    """
    Return Experiment 02 training callbacks (EarlyStopping patience=12).

    Saves the best full model by validation loss to `local_save_path` and
    optionally to a separate `drive_save_path`.
    """
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=local_save_path,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        )
    ]

    if drive_save_path and drive_save_path != local_save_path:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=drive_save_path,
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                verbose=1,
            )
        )

    callbacks.extend([
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ])
    return callbacks


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = build_cnn_model()
    model.summary()
    print("\nmodel.py: build_cnn_model() ran successfully.")
