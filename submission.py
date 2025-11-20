import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# set seeds
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_and_preprocess_data():
    # 60k train, 10k test
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    # [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = np.expand_dims(x_train, -1)  # (60000, 28, 28, 1)
    x_test = np.expand_dims(x_test, -1)    # (10000, 28, 28, 1)

    # 12000 
    x_train_sub, x_val = x_train[:-12000], x_train[-12000:]
    y_train_sub, y_val = y_train[:-12000], y_train[-12000:]

    print("Train:", x_train_sub.shape, y_train_sub.shape)
    print("Val:  ", x_val.shape, y_val.shape)
    print("Test: ", x_test.shape, y_test.shape)

    return (x_train_sub, y_train_sub), (x_val, y_val), (x_test, y_test)


def build_model():
    model = keras.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(28, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(56, (3, 3), activation="relu", padding="same"),
            layers.Flatten(),
            layers.Dense(56, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )
    return model


def plot_accuracy(history, out_path):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    epochs = range(1, len(acc) + 1)

    plt.figure()
    plt.plot(epochs, acc, marker="o", label="Train accuracy")
    plt.plot(epochs, val_acc, marker="s", label="Val accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_misclassified_examples(x_test, y_test, y_pred, out_path):
    num_classes = 10
    # sample one misclassified example per class
    chosen_indices = []
    for c in range(num_classes):
        idxs = np.where((y_test == c) & (y_pred != c))[0]
        if len(idxs) > 0:
            chosen_indices.append(idxs[0])
        else:
            # no misclassification for this class, pick any example
            idxs = np.where(y_test == c)[0]
            chosen_indices.append(idxs[0])

    plt.figure(figsize=(10, 4))
    for i, idx in enumerate(chosen_indices):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_test[idx].squeeze(), cmap="gray")
        plt.axis("off")
        plt.title(f"T:{y_test[idx]}, P:{y_pred[idx]}")
    plt.suptitle("Misclassified examples (T=true, P=pred)")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(out_path)
    plt.close()


def main():
    os.makedirs("imgs", exist_ok=True)

    # 1. data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_data()

    # 2. architecture
    model = build_model()
    model.summary()

    # print number of parameters
    print("Trainable parameters:", model.count_params())

    # 3.(Adam + sparse categorical crossentropy)
    optimizer = keras.optimizers.Adam(learning_rate=3e-4)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # 4. training
    history = model.fit(
        x_train,
        y_train,
        batch_size=32,
        epochs=10,
        validation_data=(x_val, y_val),
        verbose=2,
    )

    # 5. accuracy 
    plot_accuracy(history, "imgs/accuracy.png")
    print("Saved training curve to imgs/accuracy.png")

    # 6. test evaluation
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    # 7. misclassified examples
    y_probs = model.predict(x_test, batch_size=256, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)
    plot_misclassified_examples(
        x_test, y_test, y_pred, "imgs/misclassified_examples.png"
    )
    print("Saved misclassified examples to imgs/misclassified_examples.png")


if __name__ == "__main__":
    main()
