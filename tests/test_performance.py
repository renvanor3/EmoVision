import time

import numpy as np
import pytest

from predict.face_detector import detect_faces
from predict.predictor import load_model, predict_emotion


N_WARMUP = 5
N_RUNS = 50
MIN_REQUIRED_FPS = 10


@pytest.fixture(scope="module")
def model():
    return load_model()


def test_model_load_time():
    start = time.perf_counter()
    load_model()
    elapsed = time.perf_counter() - start
    print(f"\nChargement du modèle : {elapsed:.2f} s")
    assert elapsed < 30


def test_inference_speed_alone(model):
    img = np.random.randint(0, 256, size=(48, 48), dtype=np.uint8)

    for _ in range(N_WARMUP):
        predict_emotion(model, img)

    start = time.perf_counter()
    for _ in range(N_RUNS):
        predict_emotion(model, img)
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / N_RUNS) * 1000
    fps = N_RUNS / elapsed
    print(f"\nInference seule : {avg_ms:.2f} ms par appel ({fps:.1f} FPS théoriques)")

    assert fps >= MIN_REQUIRED_FPS


def test_detection_speed():
    img = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)

    for _ in range(N_WARMUP):
        detect_faces(img)

    start = time.perf_counter()
    for _ in range(N_RUNS):
        detect_faces(img)
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / N_RUNS) * 1000
    fps = N_RUNS / elapsed
    print(f"\nDetection seule : {avg_ms:.2f} ms par appel ({fps:.1f} FPS théoriques)")

    assert fps >= MIN_REQUIRED_FPS


def test_full_pipeline_speed(model):
    frame = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)
    face_crop = frame[100:200, 100:200]

    for _ in range(N_WARMUP):
        detect_faces(frame)
        predict_emotion(model, face_crop)

    start = time.perf_counter()
    for _ in range(N_RUNS):
        detect_faces(frame)
        predict_emotion(model, face_crop)
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / N_RUNS) * 1000
    fps = N_RUNS / elapsed
    print(f"\nPipeline complet (detection + prediction) : {avg_ms:.2f} ms par frame ({fps:.1f} FPS)")

    assert fps >= MIN_REQUIRED_FPS