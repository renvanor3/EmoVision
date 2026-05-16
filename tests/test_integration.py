import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from predict.face_detector import detect_faces, extract_face
from predict.predictor import CLASS_NAMES, load_model, predict_emotion


PROJECT_DIR = Path(__file__).parent.parent
TEST_DIR = PROJECT_DIR / "data" / "test"


@pytest.fixture(scope="module")
def model():
    return load_model()


def _get_sample_image(class_name):
    cls_dir = TEST_DIR / class_name
    filename = sorted(os.listdir(cls_dir))[0]
    return cv2.imread(str(cls_dir / filename), cv2.IMREAD_GRAYSCALE)


def test_pipeline_on_real_image(model):
    img = _get_sample_image("happy")
    assert img is not None
    label, probas = predict_emotion(model, img)
    assert label in CLASS_NAMES
    assert probas.shape == (7,)


def test_pipeline_runs_on_all_classes(model):
    for cls in CLASS_NAMES:
        img = _get_sample_image(cls)
        label, probas = predict_emotion(model, img)
        assert label in CLASS_NAMES
        assert np.isclose(probas.sum(), 1.0, atol=1e-5)


def test_pipeline_accepts_color_image(model):
    img_gray = _get_sample_image("happy")
    img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    label, _ = predict_emotion(model, img_color)
    assert label in CLASS_NAMES


def test_pipeline_handles_different_resolutions(model):
    img = _get_sample_image("happy")
    for size in [(24, 24), (100, 100), (200, 150)]:
        resized = cv2.resize(img, size)
        label, _ = predict_emotion(model, resized)
        assert label in CLASS_NAMES


def test_face_detection_then_prediction(model):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    faces = detect_faces(frame)
    if len(faces) > 0:
        face_img = extract_face(frame, faces[0])
        label, _ = predict_emotion(model, face_img)
        assert label in CLASS_NAMES
    assert isinstance(faces, np.ndarray) or isinstance(faces, tuple) or len(faces) == 0


def test_extract_face_then_predict_consistency(model):
    img = _get_sample_image("happy")
    img_large = cv2.resize(img, (200, 200))
    bbox = (50, 50, 100, 100)
    face = extract_face(img_large, bbox)
    label, _ = predict_emotion(model, face)
    assert label in CLASS_NAMES