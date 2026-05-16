import os
from pathlib import Path
import cv2
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).parent.parent
TRAIN_DIR = PROJECT_DIR / "data" / "train"
TEST_DIR = PROJECT_DIR / "data" / "test"


def list_classes():
    classes = sorted(os.listdir(TRAIN_DIR))
    print(f"Classes trouvées ({len(classes)}) : {classes}\n")
    return classes


def count_images(classes):
    print(f"{'Classe':<12} {'Train':>8} {'Test':>8}")
    print("-" * 32)

    train_counts = []
    test_counts = []
    for cls in classes:
        n_train = len(os.listdir(TRAIN_DIR / cls))
        n_test = len(os.listdir(TEST_DIR / cls))
        train_counts.append(n_train)
        test_counts.append(n_test)
        print(f"{cls:<12} {n_train:>8} {n_test:>8}")

    print("-" * 32)
    print(f"{'TOTAL':<12} {sum(train_counts):>8} {sum(test_counts):>8}\n")
    return train_counts, test_counts


def plot_distribution(classes, train_counts, test_counts):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(classes))
    ax.bar([i - 0.2 for i in x], train_counts, width=0.4, label="Train")
    ax.bar([i + 0.2 for i in x], test_counts, width=0.4, label="Test")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_ylabel("Nombre d'images")
    ax.set_title("Distribution des classes dans FER-2013")
    ax.legend()
    plt.tight_layout()
    plt.show()


def show_samples(classes, n_samples=4):
    fig, axes = plt.subplots(len(classes), n_samples, figsize=(10, 14))

    for i, cls in enumerate(classes):
        cls_dir = TRAIN_DIR / cls
        filenames = sorted(os.listdir(cls_dir))[:n_samples]

        for j, fname in enumerate(filenames):
            img = cv2.imread(str(cls_dir / fname), cv2.IMREAD_GRAYSCALE)
            axes[i, j].imshow(img, cmap="gray")
            if j == 0:
                axes[i, j].set_ylabel(cls, rotation=0, ha="right", va="center", fontsize=11)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    plt.suptitle(f"Échantillons de FER-2013 ({n_samples} images par classe)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    classes = list_classes()
    train_counts, test_counts = count_images(classes)
    plot_distribution(classes, train_counts, test_counts)
    show_samples(classes, n_samples=4)