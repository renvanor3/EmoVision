import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from data_loader import TRAIN_DIR, get_datasets


PROJECT_DIR = Path(__file__).parent.parent
MODELS_DIR = PROJECT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

NUM_CLASSES = 7
EPOCHS = 60
INITIAL_LR = 1e-3


def build_model(input_shape=(48, 48, 1), num_classes=NUM_CLASSES):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),

        tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ], name="emovision_cnn")
    return model


def compute_class_weights(class_names):
    counts = np.array([len(os.listdir(TRAIN_DIR / cls)) for cls in class_names])
    total = counts.sum()
    n = len(class_names)
    weights = total / (n * counts)
    return {i: float(w) for i, w in enumerate(weights)}


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
    plt.savefig(MODELS_DIR / "training_history.png")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    ax.set_title("Matrice de confusion (normalisée par ligne)")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(MODELS_DIR / "confusion_matrix.png")
    plt.show()


def evaluate_model(model, test_ds, class_names):
    y_true_all, y_pred_all = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true_all.append(np.argmax(labels.numpy(), axis=1))
        y_pred_all.append(np.argmax(preds, axis=1))

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)

    print("\n=== Classification report ===")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=3))

    plot_confusion_matrix(y_true, y_pred, class_names)


if __name__ == "__main__":
    train_ds, val_ds, test_ds, class_names = get_datasets()

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    class_weights = compute_class_weights(class_names)
    print(f"\nClass weights : {class_weights}\n")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODELS_DIR / "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=12,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    print("\n=== Évaluation finale sur le test set ===")
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test accuracy : {test_acc:.4f}")
    print(f"Test loss     : {test_loss:.4f}")

    final_path = MODELS_DIR / "final.keras"
    model.save(final_path)
    print(f"\nModèle final sauvegardé : {final_path}")

    plot_history(history)
    evaluate_model(model, test_ds, class_names)