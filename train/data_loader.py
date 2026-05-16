import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).parent.parent
TRAIN_DIR = PROJECT_DIR / "data" / "train"
TEST_DIR = PROJECT_DIR / "data" / "test"

IMG_SIZE = (48, 48)
BATCH_SIZE = 64
VAL_SPLIT = 0.15
SEED = 42


def load_raw_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=VAL_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        label_mode="categorical",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=VAL_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        label_mode="categorical",
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        label_mode="categorical",
        shuffle=False,
    )

    return train_ds, val_ds, test_ds


def get_augmentation_layer():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ], name="data_augmentation")


def prepare_datasets(train_ds, val_ds, test_ds):
    normalization = tf.keras.layers.Rescaling(1.0 / 255)
    augmentation = get_augmentation_layer()
    autotune = tf.data.AUTOTUNE

    train_ds = train_ds.map(lambda x, y: (normalization(x), y), num_parallel_calls=autotune)
    val_ds = val_ds.map(lambda x, y: (normalization(x), y), num_parallel_calls=autotune)
    test_ds = test_ds.map(lambda x, y: (normalization(x), y), num_parallel_calls=autotune)

    train_ds = train_ds.cache()
    val_ds = val_ds.cache()
    test_ds = test_ds.cache()

    train_ds = train_ds.map(
        lambda x, y: (augmentation(x, training=True), y),
        num_parallel_calls=autotune,
    )

    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)

    return train_ds, val_ds, test_ds


def get_datasets():
    train_ds, val_ds, test_ds = load_raw_datasets()
    class_names = train_ds.class_names
    train_ds, val_ds, test_ds = prepare_datasets(train_ds, val_ds, test_ds)
    return train_ds, val_ds, test_ds, class_names


def show_batch(dataset, class_names, title="Batch"):
    images, labels = next(iter(dataset))
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        if i >= len(images):
            break
        ax.imshow(images[i].numpy().squeeze(), cmap="gray")
        label_idx = labels[i].numpy().argmax()
        ax.set_title(class_names[label_idx])
        ax.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_ds, val_ds, test_ds, class_names = get_datasets()
    print(f"\nClasses : {class_names}")
    print(f"Batches train : {tf.data.experimental.cardinality(train_ds).numpy()}")
    print(f"Batches val   : {tf.data.experimental.cardinality(val_ds).numpy()}")
    print(f"Batches test  : {tf.data.experimental.cardinality(test_ds).numpy()}")
    show_batch(train_ds, class_names, title="Train (normalisé + augmenté)")