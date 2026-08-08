"""CNN for detecting cirrus clouds in orbital photography.

The module trains and applies a small convolutional network on a directory of labeled
images. Class labels are inferred from the names of the subdirectories of the training
directory, so that a layout of

    training_data/
        cirrus/
        not_cirrus/

produces the classes `cirrus` and `not_cirrus`.

Three commands may be used:

    python cirrus_classifier.py kfold      # stratified k-fold cross validation, yields scores for individual folds
    python cirrus_classifier.py train      # trains on a single split and saves the model
    python cirrus_classifier.py classify   # sorts the contents of to_classify/ into classified/<class>/

The dataset was severely imbalanced initially, which led the model to place every photo in the
majority class. Further photos were selected to correct that imbalance. All images were taken
from https://www.flickr.com/photos/raspberrypi/albums.

This is the original training script with a number of modifications, marked by a FIX
comment if not purely cosmetic. Only issues identifiable by reading the code have been
treated. All changes requiring a training run to test will be added, if any, once the dataset is
recovered.

Conventions:
    - For later use, the inferred labels are saved in saved_model/class_names.json, in sorted order.


Assumptions:
    - Every subdirectory of the training directory names exactly one class.
    - Photographs are large enough that scaling to 684x513 preserves the cloud structure.
"""

import argparse
import json
import multiprocessing
import os
import shutil

import matplotlib

# FIX: selects a non-interactive backend, so that figures are written to disk instead of
# displayed. Must precede the pyplot import.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

DATA_DIR = "training_data"
SAMPLES_DIR = "to_classify"
CLASSIFIED_DIR = "classified"
PLOTS_DIR = "plots"

SAVED_MODEL = "saved_model/my_model"
# FIX: the mapping from labels to class names is saved alongside the model. In the original it
# was re-derived from the training directory at classification time, which caused mislabeled
# predictions if the directory structure differed from the one used in training.
CLASS_NAMES_FILE = "saved_model/class_names.json"

BATCH_SIZE = 8
IMG_WIDTH = 684
IMG_HEIGHT = 513

NO_SPLITS = 10
KFOLD_EPOCHS = 50
TRAIN_EPOCHS = 65

# FIX: fixes the k-fold splits, the train/validation split and the shuffle order, so that a run
# can be reproduced.
SEED = 42

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")

AUTOTUNE = tf.data.AUTOTUNE

# FIX: subprocess isolation is used only where the fork start method is available. Under spawn
# (Windows, and macOS since Python 3.8) each child re-imports the module, so the dataset would
# be redundantly reloaded each fold; clear_session() is used instead.
# os.name also returns posix on macOS, so macOS users must override this.
USE_SUBPROCESSES = os.name == "posix"

# Populated by load_dataset(), which main() calls for the kfold and train commands.
# Kept at module level because a forked child inherits the parent's memory.
data = None
labels = None
no_classes = 0
class_names = []


def is_image(filename: str) -> bool:
    """Checks if filename ends with a recognized image extension.

    Args:
        filename (str): the name to test (with/without directory components).

    Returns:
        bool: True if the extension is an element of IMAGE_SUFFIXES.
    """
    return filename.lower().endswith(IMAGE_SUFFIXES)


def load_image(path: str) -> np.ndarray:
    """Reads an image and resizes it to the network's input dimensions.

    Used for both training and inference, to ensure the model is never used on pixels produced by a
    different resampling filter than the one it was fitted on.

    Args:
        path (str): path of the image to read.

    Returns:
        np.ndarray: float32 array of shape (IMG_HEIGHT, IMG_WIDTH, 3), values in [0, 255].
    """
    # FIX: convert("RGB") is applied to the opened image to fix channel count.
    img = Image.open(path).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.LANCZOS)

    return np.asarray(img).astype("float32")


def load_data(data_dir: str) -> tuple[list, list, int, list]:
    """Loads every image in a directory, labeled by the name of subdirectory containing it.

    Args:
        data_dir (str): directory with one subdirectory per class.

    Returns:
        tuple[list, list, int, list]: the image arrays, their integer labels, the number of
            classes, and the class names ordered so that index i names label i.

    Raises:
        FileNotFoundError: if data_dir does not exist, has no subdirectories, or no
            images exist in any of them.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {os.path.abspath(data_dir)}")

    label_names = []
    data = []
    labels = []

    # FIX: sorted(). os.listdir() does not guarantee an order. An unsorted listing may assign
    # labels to classes differently on a different machine, leading to erroneous predictions.
    for dir_name in sorted(os.listdir(data_dir)):

        curr_path = os.path.join(data_dir, dir_name)

        # FIX: skips files so that they are not mistakenly treated as a class.
        if not os.path.isdir(curr_path):
            continue

        label = len(label_names)
        label_names.append(dir_name)

        for image in sorted(os.listdir(curr_path)):

            # FIX: ensures only images are included in the dataset.
            if not is_image(image):
                continue

            labels.append(label)
            data.append(load_image(os.path.join(curr_path, image)))

    if not label_names:
        raise FileNotFoundError(f"No class subdirectories inside {os.path.abspath(data_dir)}")

    if not data:
        raise FileNotFoundError(f"No images found under {os.path.abspath(data_dir)}")

    label_no = len(label_names)

    print(f"Finished loading {len(data)} images across {label_no} classes")
    for index, name in enumerate(label_names):
        print(f"  {name}: {labels.count(index)}")

    return data, labels, label_no, label_names


def load_dataset() -> None:
    """Populates the data global variables to be accessed across processes.

    Called from the entry point, not at import time. Loading at import would require that
    the training directory be present even when classifying.
    """
    global data, labels, no_classes, class_names

    data, labels, no_classes, class_names = load_data(DATA_DIR)

    data = np.asarray(data).astype("float32")
    labels = np.asarray(labels).astype("int")


def configure_for_performance(ds: tf.data.Dataset, shuffle: bool = False) -> tf.data.Dataset:
    """Applies caching, shuffling, batching, and prefetching to a dataset.

    Args:
        ds (tf.data.Dataset): the dataset to configure.
        shuffle (bool): whether to shuffle. Ideally off for validation data, since order has no
            effect on the results in that case.

    Returns:
        tf.data.Dataset: the configured dataset.
    """
    ds = ds.cache()

    # FIX: shuffling is now optional and seeded.
    if shuffle:
        ds = ds.shuffle(buffer_size=1000, seed=SEED)

    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


def create_model(no_classes: int) -> keras.Model:
    """Builds and compiles the CNN.

    The model follows a typical CNN structure and has been limited to a small number of layers due to
    overfitting. Among the solutions to that problem are data augmentation and dropout layers,
    the latter of which is implemented below. Adding further layers, in a manner inspired by
    VGG16, was tried, however most of the improvement came from changing optimizers and their
    parameters.

    Args:
        no_classes (int): number of output units, one per class.

    Returns:
        keras.Model: the compiled model. Output logits, so loss must be constructed with from_logits=True.

    .. note::
        The flattening required before Dense leads to Dense creating approximately 44.5 million parameters.
        This number may be drastically reduced by replacing Flatten with GlobalAveragePooling2D.
        Whether this preserves accuracy will be decided by further testing.
    """
    model = Sequential([

        layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(no_classes, name="outputs")

    ])

    optimizer = keras.optimizers.SGD(learning_rate=0.0001, weight_decay=0.1)

    model.compile(optimizer=optimizer,
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])

    return model


def visualization(history: keras.callbacks.History, filename: str) -> None:
    """Saves the accuracy and loss curves of a training run to a file.

    Args:
        history (keras.callbacks.History): the object returned by model.fit.
        filename (str): name of the image to save to (in PLOTS_DIR).
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    # FIX: epoch count is derived from history to ensure the plot remains correct
    # when the epoch count is not the expected one (e.g. training stopped early).
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")

    out_path = os.path.join(PLOTS_DIR, filename)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

    print(f"Saved plot to {out_path}")


def run_fold(index: int, no_classes: int, training_indices: np.ndarray,
             validation_indices: np.ndarray, accuracies, losses
             ) -> None:
    """Trains the model on one fold and records the final validation scores.

    Reads the dataset from the global variables, since, under the fork start
    method, a child process inherits them at no cost.

    Args:
        index (int): fold number, indexed from 0.
        no_classes (int): number of classes to be passed to create_model.
        training_indices, validation_indices (np.ndarray): what data and labels to be used
        accuracies, losses (multiprocessing.Array): shared arrays of length NO_SPLITS, into which
            the final validation accuracy and loss of this fold are written.
    """
    print(f"Fold {index + 1} out of {NO_SPLITS}")

    training_dataset = tf.data.Dataset.from_tensor_slices((data[training_indices], labels[training_indices]))
    validation_dataset = tf.data.Dataset.from_tensor_slices((data[validation_indices], labels[validation_indices]))

    training_dataset = configure_for_performance(training_dataset, shuffle=True)
    validation_dataset = configure_for_performance(validation_dataset)

    print("Loaded training and validation datasets")

    model = create_model(no_classes)

    print("Training start")

    # FIX: batch_size is no longer passed to fit() since the datasets are already batched.
    history = model.fit(

        training_dataset,
        validation_data=validation_dataset,
        epochs=KFOLD_EPOCHS
    )

    accuracies[index] = history.history["val_accuracy"][-1]
    losses[index] = history.history["val_loss"][-1]

    visualization(history, f"fold_{index + 1}.png")


def run_fold_in_process(index: int, no_classes: int, training_indices: np.ndarray,
                        validation_indices: np.ndarray, accuracies, losses
                        ) -> None:
    """Runs one fold in the current process, clearing Keras state beforehand.

    Used if subprocess isolation is unavailable. clear_session() discards the
    previous fold's graph and weights, to ensure memory efficiency.

    Args:
        See run_fold, whose arguments are forwarded unchanged.
    """
    keras.backend.clear_session()

    run_fold(index, no_classes, training_indices, validation_indices, accuracies, losses)


def run_kfold() -> None:
    """Runs k-fold cross validation and prints fold and averaged scores.

    Returns nothing, as this is used solely for validation. Training is done in multiple folds,
    each of which uses a different portion of the data. Indicates how well the
    given architecture is able to generalize, informing the final model choice.

    Each fold is run in a separate process if the OS allows for it, because a process
    returns all of its memory, including GPU allocations, on exit. For the optimization to be effective,
    the processes must be run sequentially (via join()), not in parallel.
    """
    # Stratifying ensures that, for each fold, the class distribution is similar to that of the entire dataset.
    cross_validator = StratifiedKFold(n_splits=NO_SPLITS, shuffle=True, random_state=SEED)

    accuracies = multiprocessing.Array("d", NO_SPLITS)
    losses = multiprocessing.Array("d", NO_SPLITS)

    for index, (training_indices, validation_indices) in enumerate(cross_validator.split(data, labels)):

        fold_args = (index, no_classes, training_indices, validation_indices, accuracies, losses)

        if USE_SUBPROCESSES:
            process = multiprocessing.Process(target=run_fold, args=fold_args)
            process.start()
            process.join()
        else:
            run_fold_in_process(*fold_args)

    print("------------------------------------------------------------------------")
    print("Score per fold")
    for i in range(0, len(accuracies)):
        print("------------------------------------------------------------------------")
        print(f"> Fold {i + 1} - Loss: {losses[i]} - Accuracy: {accuracies[i]}")
    print("------------------------------------------------------------------------")
    print("Average scores for all folds:")
    print(f"> Accuracy: {np.mean(accuracies)} (+- {np.std(accuracies)})")
    print(f"> Loss: {np.mean(losses)}")
    print("------------------------------------------------------------------------")


def train() -> None:
    """Trains on a single stratified split and saves the model and its class names.

    Intended to be run once the scores reported by run_kfold are satisfactory.
    """
    training_data, validation_data, training_labels, validation_labels = train_test_split(
        data, labels, test_size=0.2, stratify=labels, random_state=SEED
    )

    training_dataset = tf.data.Dataset.from_tensor_slices((training_data, training_labels))
    validation_dataset = tf.data.Dataset.from_tensor_slices((validation_data, validation_labels))

    training_dataset = configure_for_performance(training_dataset, shuffle=True)
    validation_dataset = configure_for_performance(validation_dataset)

    print("Loaded training and validation datasets")

    model = create_model(no_classes)

    print("Training start")

    history = model.fit(

        training_dataset,
        validation_data=validation_dataset,
        epochs=TRAIN_EPOCHS
    )

    visualization(history, "training.png")

    os.makedirs(os.path.dirname(CLASS_NAMES_FILE), exist_ok=True)

    # FIX: include_optimizer=False. The momentum slots are intended for resuming training,
    # and therefore not needed.
    model.save(SAVED_MODEL, include_optimizer=False)

    with open(CLASS_NAMES_FILE, "w") as f:
        json.dump(class_names, f, indent=2)

    print(f"Saved model to {SAVED_MODEL} and class names to {CLASS_NAMES_FILE}")


def classify(model_path: str) -> None:
    """Sorts every image in SAMPLES_DIR into a subdirectory of CLASSIFIED_DIR indicating its class.

    Args:
        model_path (str): directory of the SavedModel as produced by train().

    Raises:
        FileNotFoundError: if the model, the recorded class names, or the sample directory are
            absent.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No saved model at {model_path}. Run the train command first.")

    if not os.path.exists(CLASS_NAMES_FILE):
        raise FileNotFoundError(f"No class names at {CLASS_NAMES_FILE}. Run the train command first.")

    if not os.path.isdir(SAMPLES_DIR):
        raise FileNotFoundError(f"Directory not found: {os.path.abspath(SAMPLES_DIR)}")

    model = keras.models.load_model(model_path)

    # FIX: read from the file written at training time. Deriving these from the training
    # directory made classification depend on that directory being present and unchanged.
    with open(CLASS_NAMES_FILE) as f:
        saved_class_names = json.load(f)

    # FIX: the filenames are listed once and reused for both the prediction and the move.
    # The original listed the directory twice and indexed the predictions by the second listing,
    # causing any disagreement between the two to pair a prediction with the wrong file.
    image_files = sorted(f for f in os.listdir(SAMPLES_DIR) if is_image(f))

    if not image_files:
        print(f"No images to classify in {SAMPLES_DIR}")
        return

    samples_to_predict = [load_image(os.path.join(SAMPLES_DIR, image)) for image in image_files]
    samples_to_predict = np.asarray(samples_to_predict).astype("float32")

    predictions = model.predict(samples_to_predict)

    # FIX: the destination directories are created here, since shutil.move requires the
    # destination to exist.
    for name in saved_class_names:
        os.makedirs(os.path.join(CLASSIFIED_DIR, name), exist_ok=True)

    for index, image in enumerate(image_files):

        score = tf.nn.softmax(predictions[index])
        predicted = saved_class_names[int(np.argmax(score))]

        shutil.move(os.path.join(SAMPLES_DIR, image), os.path.join(CLASSIFIED_DIR, predicted, image))

        print(f"{image}: {predicted} ({float(np.max(score)):.1%})")


def main() -> None:
    """Parses and runs the commands, loading the dataset only where needed."""
    parser = argparse.ArgumentParser(description="Cirrus cloud classifier")
    parser.add_argument("command", choices=["kfold", "train", "classify"])
    args = parser.parse_args()

    if args.command == "classify":
        classify(SAVED_MODEL)
        return

    load_dataset()

    if args.command == "kfold":
        run_kfold()
    else:
        train()


# FIX: check if the module is imported as __main__. Otherwise, importing this module
# runs the whole script, which causes problems in the spawn start regime.
if __name__ == "__main__":
    main()