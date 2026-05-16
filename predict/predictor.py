import os
import random
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


PROJECT_DIR = Path(__file__).parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "final.keras"
TEST_DIR = PROJECT_DIR / "data" / "test"

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
IMG_SIZE = (48, 48)


def load_model(model_path=MODEL_PATH):
    return tf.keras.models.load_model(model_path)


def preprocess_image(image):
    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, IMG_SIZE)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=(0, -1))
    return image


def predict_emotion(model, image):
    x = preprocess_image(image)
    probabilities = model.predict(x, verbose=0)[0]
    predicted_idx = int(np.argmax(probabilities))
    predicted_label = CLASS_NAMES[predicted_idx]
    return predicted_label, probabilities


if __name__ == "__main__":
    model = load_model()
    print(f"Modèle chargé depuis : {MODEL_PATH}\n")
    print(f"{'Vraie classe':<12} {'Prédiction':<12} {'Confiance':<10} {'Statut'}")
    print("-" * 50)

    correct = 0
    for cls in CLASS_NAMES:
        cls_dir = TEST_DIR / cls
        filename = random.choice(os.listdir(cls_dir))
        img_path = cls_dir / filename
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        label, probas = predict_emotion(model, img)
        status = "OK" if label == cls else "X"
        if label == cls:
            correct += 1
        print(f"{cls:<12} {label:<12} {probas.max()*100:>6.1f}%    {status}")

    print(f"\nScore sur cet échantillon : {correct}/7")