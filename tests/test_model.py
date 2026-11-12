"""
test_model.py
=============
G19 Facial Emotion Recognition — CSE276

Purpose:
    Unit tests for src/model.py.
    Run with: pytest tests/test_model.py -v

Tests verify:
    - build_cnn_model() returns a compiled keras.Model.
    - The model has the correct input and output shapes.
    - The model produces valid probability distributions (sum ≈ 1).
    - The model produces 7 output classes.
    - get_callbacks() returns a list of Keras callbacks.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Lazy import of TensorFlow — skip tests gracefully if TF is not installed
# ---------------------------------------------------------------------------

tf = pytest.importorskip("tensorflow", reason="TensorFlow is not installed.")


from src.model import build_cnn_model, get_callbacks, INPUT_SHAPE, NUM_CLASSES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model():
    """Build and return the model once for all tests in this module."""
    return build_cnn_model(use_augmentation=False)   # No augmentation for speed


@pytest.fixture
def dummy_batch():
    """Small batch of random normalised images for forward pass testing."""
    return np.random.rand(4, 48, 48, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Tests for build_cnn_model
# ---------------------------------------------------------------------------

class TestBuildCNNModel:

    def test_returns_keras_model(self, model):
        """build_cnn_model() should return a tf.keras.Model instance."""
        assert isinstance(model, tf.keras.Model)

    def test_input_shape(self, model):
        """Model input should accept (batch, 48, 48, 1)."""
        expected = (None,) + INPUT_SHAPE
        assert model.input_shape == expected, f"Input shape mismatch: {model.input_shape}"

    def test_output_shape(self, model):
        """Model output should have NUM_CLASSES=7 units."""
        expected = (None, NUM_CLASSES)
        assert model.output_shape == expected, f"Output shape mismatch: {model.output_shape}"

    def test_model_has_name(self, model):
        """Model should have the name 'FER_CNN'."""
        assert model.name == "FER_CNN"

    def test_model_is_compiled(self, model):
        """Model should be compiled (optimizer should not be None)."""
        assert model.optimizer is not None

    def test_trainable_params_positive(self, model):
        """Model must have trainable parameters."""
        trainable = sum(np.prod(w.shape) for w in model.trainable_weights)
        assert trainable > 0, "Model has no trainable parameters."


# ---------------------------------------------------------------------------
# Tests for forward pass (inference)
# ---------------------------------------------------------------------------

class TestModelForwardPass:

    def test_output_shape_matches_batch(self, model, dummy_batch):
        """Output shape should be (batch_size, 7)."""
        output = model.predict(dummy_batch, verbose=0)
        assert output.shape == (4, NUM_CLASSES)

    def test_output_is_probability_distribution(self, model, dummy_batch):
        """Each row of the output should sum to approximately 1 (softmax)."""
        output = model.predict(dummy_batch, verbose=0)
        row_sums = output.sum(axis=1)
        np.testing.assert_allclose(
            row_sums, np.ones(4), atol=1e-5,
            err_msg="Softmax probabilities do not sum to 1."
        )

    def test_output_values_in_range(self, model, dummy_batch):
        """All output values should be in [0, 1]."""
        output = model.predict(dummy_batch, verbose=0)
        assert output.min() >= 0.0, "Output contains values below 0."
        assert output.max() <= 1.0, "Output contains values above 1."

    def test_argmax_in_valid_range(self, model, dummy_batch):
        """Predicted class indices should be in [0, NUM_CLASSES-1]."""
        output = model.predict(dummy_batch, verbose=0)
        preds  = np.argmax(output, axis=1)
        assert all(0 <= p < NUM_CLASSES for p in preds), "Predicted class out of range."


# ---------------------------------------------------------------------------
# Tests for get_callbacks
# ---------------------------------------------------------------------------

class TestGetCallbacks:

    def test_returns_list(self):
        """get_callbacks() should return a list."""
        callbacks = get_callbacks()
        assert isinstance(callbacks, list)

    def test_returns_three_callbacks(self):
        """Should return exactly 3 callbacks."""
        callbacks = get_callbacks()
        assert len(callbacks) == 3

    def test_callback_types(self):
        """Should include ModelCheckpoint, EarlyStopping, ReduceLROnPlateau."""
        callbacks = get_callbacks()
        types = {type(cb).__name__ for cb in callbacks}
        assert "ModelCheckpoint"    in types
        assert "EarlyStopping"      in types
        assert "ReduceLROnPlateau"  in types


# ---------------------------------------------------------------------------
# Tests for augmentation behaviour & model save/load round-trip
# ---------------------------------------------------------------------------

class TestAugmentationAndSerialization:

    def test_brightness_value_range_is_unit_interval(self):
        """RandomBrightness must use value_range=(0.0, 1.0) for [0, 1] inputs."""
        from src.preprocessing import build_augmentation_layer

        aug = build_augmentation_layer()
        brightness_layers = [
            layer for layer in aug.layers
            if isinstance(layer, tf.keras.layers.RandomBrightness)
        ]
        assert len(brightness_layers) == 1
        assert tuple(brightness_layers[0].value_range) == (0.0, 1.0)

    def test_augmentation_disabled_during_inference(self, dummy_batch):
        """Augmentation must not alter validation/test inputs (training=False)."""
        from src.preprocessing import build_augmentation_layer

        aug = build_augmentation_layer()
        out = aug(dummy_batch, training=False).numpy()
        np.testing.assert_allclose(out, dummy_batch, atol=1e-6)

    def test_model_save_and_load_roundtrip(self, tmp_path, dummy_batch):
        """Saved .keras model should reload and produce identical predictions."""
        model_with_aug = build_cnn_model(use_augmentation=True)
        preds_before = model_with_aug.predict(dummy_batch, verbose=0)

        save_path = tmp_path / "test_model.keras"
        model_with_aug.save(save_path)
        reloaded = tf.keras.models.load_model(save_path)
        preds_after = reloaded.predict(dummy_batch, verbose=0)

        np.testing.assert_allclose(preds_before, preds_after, atol=1e-5)
