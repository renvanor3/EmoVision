"""
train_baseline.py
Entraînement d'un CNN baseline simple sur FER-2013.
Sert de smoke test pour valider le pipeline complet et de point de
départ utilisable pour le développement de la GUI en parallèle.
"""

import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt

from data_loader import get_datasets


PROJECT_DIR = Path(__file__).parent.parent
MODELS_DIR = PROJECT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

NUM_CLASSES = 7
EPOCHS = 5


def build_baseline_model(input_shape=(48, 48, 1), num_classes=NUM_CLASSES):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ], name="baseline_cnn")
    return model


def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"], label="Train")
    axes[0].plot(history.history["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="Train")
    axes[1].plot(history.history["val_accuracy"], label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_ds, val_ds, test_ds, class_names = get_datasets()

    model = build_baseline_model()
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
    )

    test_loss, test_acc = model.evaluate(test_ds)
    print(f"\nTest accuracy : {test_acc:.4f}")
    print(f"Test loss     : {test_loss:.4f}")

    save_path = MODELS_DIR / "baseline.keras"
    model.save(save_path)
    print(f"Modèle sauvegardé : {save_path}")

    plot_history(history)