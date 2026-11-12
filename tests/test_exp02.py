"""
test_exp02.py
=============
G19 Facial Emotion Recognition — CSE276

Unit tests for Experiment 02 (Improved 4-block CNN, refined augmentation,
smoothed class weights, and dual checkpoint callbacks).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="TensorFlow is not installed.")

from src.data_loader import (
    compute_smoothed_class_weights,
    compute_training_class_weights,
    load_fer2013,
    load_fer2013_train_val,
)
from src.model import (
    INPUT_SHAPE,
    NUM_CLASSES,
    build_improved_cnn_model,
    get_exp02_callbacks,
)
from src.preprocessing import build_augmentation_layer, build_refined_augmentation_layer


@pytest.fixture
def dummy_batch():
    """Random normalised 48x48x1 batch in [0, 1]."""
    rng = np.random.default_rng(42)
    return rng.random((4, 48, 48, 1), dtype=np.float32)


class TestImprovedCNNModel:

    @pytest.mark.parametrize("head_type", ["A", "B"])
    def test_model_construction_and_shapes(self, head_type):
        model = build_improved_cnn_model(head_type=head_type, use_augmentation=False)
        assert isinstance(model, tf.keras.Model)
        assert model.input_shape == (None,) + INPUT_SHAPE
        assert model.output_shape == (None, NUM_CLASSES)
        assert model.optimizer is not None

    def test_invalid_head_and_augmentation_raise(self):
        with pytest.raises(ValueError):
            build_improved_cnn_model(head_type="C")
        with pytest.raises(ValueError):
            build_improved_cnn_model(augmentation_type="invalid")

    def test_head_a_and_head_b_have_distinct_param_counts(self):
        model_a = build_improved_cnn_model(head_type="A", use_augmentation=False)
        model_b = build_improved_cnn_model(head_type="B", use_augmentation=False)
        params_a = model_a.count_params()
        params_b = model_b.count_params()
        assert params_a > 0 and params_b > 0
        # Head A (Flatten 3x3x256 -> 2304 -> 256) has more params than Head B (512 -> 256)
        assert params_a > params_b

    @pytest.mark.parametrize("head_type", ["A", "B"])
    def test_forward_pass_and_softmax_output(self, head_type, dummy_batch):
        model = build_improved_cnn_model(head_type=head_type, augmentation_type="refined")
        probs = model.predict(dummy_batch, verbose=0)
        assert probs.shape == (4, NUM_CLASSES)
        assert probs.min() >= 0.0 and probs.max() <= 1.0
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(4), atol=1e-5)

        # Training mode forward pass also succeeds
        train_out = model(dummy_batch, training=True).numpy()
        assert train_out.shape == (4, NUM_CLASSES)

    def test_improved_model_save_and_load_roundtrip(self, tmp_path, dummy_batch):
        model = build_improved_cnn_model(head_type="B", augmentation_type="refined")
        preds_before = model.predict(dummy_batch, verbose=0)
        save_path = tmp_path / "exp02_test.keras"
        model.save(save_path)
        reloaded = tf.keras.models.load_model(save_path)
        preds_after = reloaded.predict(dummy_batch, verbose=0)
        np.testing.assert_allclose(preds_before, preds_after, atol=1e-5)


class TestExp02Augmentation:

    def test_refined_augmentation_layers_and_brightness_range(self, dummy_batch):
        aug = build_refined_augmentation_layer()
        layer_types = [type(layer).__name__ for layer in aug.layers]
        assert "RandomFlip" in layer_types
        assert "RandomRotation" in layer_types
        assert "RandomZoom" in layer_types
        assert "RandomTranslation" in layer_types
        assert "RandomBrightness" in layer_types
        assert "RandomContrast" in layer_types

        brightness = [l for l in aug.layers if isinstance(l, tf.keras.layers.RandomBrightness)][0]
        assert tuple(brightness.value_range) == (0.0, 1.0)

        contrast = [l for l in aug.layers if isinstance(l, tf.keras.layers.RandomContrast)][0]
        assert tuple(contrast.value_range) == (0.0, 1.0)

        # Validation/test inference must not be altered
        out = aug(dummy_batch, training=False).numpy()
        np.testing.assert_allclose(out, dummy_batch, atol=1e-6)

    def test_baseline_augmentation_preserved(self):
        aug = build_augmentation_layer()
        layer_types = [type(layer).__name__ for layer in aug.layers]
        assert layer_types == [
            "RandomFlip",
            "RandomRotation",
            "RandomZoom",
            "RandomBrightness",
        ]


class TestExp02ClassWeightsAndCallbacks:

    def test_smoothed_class_weights_from_training_labels(self):
        # Imbalanced synthetic training labels (classes 0..6)
        y_train = np.array([0] * 20 + [1] * 2 + [2] * 10 + [3] * 40 + [4] * 15 + [5] * 8 + [6] * 18)
        full_w = compute_training_class_weights(y_train)
        smooth_w = compute_smoothed_class_weights(y_train, power=0.5)

        assert set(full_w.keys()) == set(range(7))
        assert set(smooth_w.keys()) == set(range(7))
        # Mean of smoothed weights is normalised to 1.0
        assert np.isclose(np.mean(list(smooth_w.values())), 1.0, atol=1e-6)
        # Rare class (1) still has higher weight than majority class (3), but less extreme than full_w
        assert smooth_w[1] > smooth_w[3]
        assert (smooth_w[1] / smooth_w[3]) < (full_w[1] / full_w[3])

    def test_exp02_callbacks_with_local_and_drive_paths(self, tmp_path):
        local_p = str(tmp_path / "exp02_best_model.keras")
        drive_p = str(tmp_path / "exp02_best_model_drive.keras")
        callbacks = get_exp02_callbacks(local_save_path=local_p, drive_save_path=drive_p)

        ckpts = [cb for cb in callbacks if isinstance(cb, tf.keras.callbacks.ModelCheckpoint)]
        assert len(ckpts) == 2
        assert {c.filepath for c in ckpts} == {local_p, drive_p}
        for c in ckpts:
            assert c.monitor == "val_loss"
            assert c.save_best_only is True

        early = [cb for cb in callbacks if isinstance(cb, tf.keras.callbacks.EarlyStopping)][0]
        assert early.patience == 12
        assert early.restore_best_weights is True

        rlr = [cb for cb in callbacks if isinstance(cb, tf.keras.callbacks.ReduceLROnPlateau)][0]
        assert rlr.patience == 5
        assert np.isclose(rlr.factor, 0.5)
        assert np.isclose(rlr.min_lr, 1e-6)


class TestTrainValIsolationLoader:

    def test_train_val_loader_never_decodes_private_test(self, tmp_path):
        """PrivateTest rows with invalid pixels must never be decoded by load_fer2013_train_val."""
        valid_pixels = " ".join(["128"] * (48 * 48))
        csv_file = tmp_path / "synthetic_fer.csv"
        csv_file.write_text(
            "emotion,pixels,Usage\n"
            f"0,{valid_pixels},Training\n"
            f"3,{valid_pixels},Training\n"
            f"6,{valid_pixels},PublicTest\n"
            "2,NOT_A_VALID_PIXEL_STRING,PrivateTest\n"
        )

        result = load_fer2013_train_val(str(csv_file))
        assert len(result) == 4
        X_train, y_train, X_val, y_val = result
        assert X_train.shape == (2, 48, 48, 1)
        assert X_val.shape == (1, 48, 48, 1)
        assert X_train.dtype == np.float32 and X_val.dtype == np.float32
        assert y_train.dtype == np.int32 and y_val.dtype == np.int32
        assert y_train.tolist() == [0, 3]
        assert y_val.tolist() == [6]

    def test_real_fer2013_train_val_counts_and_baseline_loader(self):
        """Verify exact FER-2013 Training (28,709) and PublicTest (3,589) counts if CSV is present."""
        csv_path = Path(__file__).resolve().parent.parent / "dataset" / "fer2013.csv"
        if not csv_path.exists():
            pytest.skip("dataset/fer2013.csv not present locally.")

        X_train, y_train, X_val, y_val = load_fer2013_train_val(str(csv_path))
        assert X_train.shape == (28709, 48, 48, 1)
        assert y_train.shape == (28709,)
        assert X_val.shape == (3589, 48, 48, 1)
        assert y_val.shape == (3589,)

        # Confirm original baseline load_fer2013() remains unchanged (returns 6 arrays)
        res6 = load_fer2013(str(csv_path))
        assert len(res6) == 6
        assert res6[4].shape == (3589, 48, 48, 1)
