"""
test_predictor.py
Tests unitaires pour le module de prédiction d'émotion.
Vérifie le prétraitement des images et la cohérence des sorties
du modèle (forme, plage de valeurs, somme à 1 pour la softmax).
"""

import numpy as np
import pytest

from predict.predictor import (
    CLASS_NAMES,
    load_model,
    predict_emotion,
    preprocess_image,
)


@pytest.fixture(scope="module")
def model():
    return load_model()


def test_class_names_count():
    assert len(CLASS_NAMES) == 7


def test_class_names_unique():
    assert len(set(CLASS_NAMES)) == 7


def test_preprocess_grayscale_returns_correct_shape():
    img = np.random.randint(0, 256, size=(100, 80), dtype=np.uint8)
    out = preprocess_image(img)
    assert out.shape == (1, 48, 48, 1)


def test_preprocess_color_returns_correct_shape():
    img = np.random.randint(0, 256, size=(100, 80, 3), dtype=np.uint8)
    out = preprocess_image(img)
    assert out.shape == (1, 48, 48, 1)


def test_preprocess_pixel_range_normalized():
    img = np.full((100, 100), 255, dtype=np.uint8)
    out = preprocess_image(img)
    assert out.max() <= 1.0
    assert out.min() >= 0.0


def test_preprocess_dtype_is_float32():
    img = np.random.randint(0, 256, size=(50, 50), dtype=np.uint8)
    out = preprocess_image(img)
    assert out.dtype == np.float32


def test_predict_returns_valid_label(model):
    img = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
    label, _ = predict_emotion(model, img)
    assert label in CLASS_NAMES


def test_predict_probas_have_correct_shape(model):
    img = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
    _, probas = predict_emotion(model, img)
    assert probas.shape == (7,)


def test_predict_probas_sum_to_one(model):
    img = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
    _, probas = predict_emotion(model, img)
    assert np.isclose(probas.sum(), 1.0, atol=1e-5)


def test_predict_probas_in_valid_range(model):
    img = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
    _, probas = predict_emotion(model, img)
    assert (probas >= 0).all()
    assert (probas <= 1).all()


def test_predict_label_matches_argmax(model):
    img = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
    label, probas = predict_emotion(model, img)
    assert label == CLASS_NAMES[np.argmax(probas)]