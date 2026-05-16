from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from data_loader import get_datasets


PROJECT_DIR = Path(__file__).parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "final.keras"


if __name__ == "__main__":
    print(f"Chargement du modèle : {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Chargement des datasets...")
    _, _, test_ds, class_names = get_datasets()

    print("\n=== Évaluation sur le test set ===")
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"\nTest accuracy : {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test loss     : {test_loss:.4f}")

    print("\n=== Classification report par classe ===")
    y_true_all, y_pred_all = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true_all.append(np.argmax(labels.numpy(), axis=1))
        y_pred_all.append(np.argmax(preds, axis=1))

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)

    print(classification_report(y_true, y_pred, target_names=class_names, digits=3))

    print("=== Matrice de confusion (valeurs brutes) ===")
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n{'':>10}", end="")
    for name in class_names:
        print(f"{name[:7]:>8}", end="")
    print()
    for i, name in enumerate(class_names):
        print(f"{name[:9]:>10}", end="")
        for j in range(len(class_names)):
            print(f"{cm[i, j]:>8}", end="")
        print()