import os
import numpy as np
import tensorflow as tf
import yaml
from tensorflow.keras import layers, models

params = yaml.safe_load(open("params.yaml"))["train"]
data = np.load("data/processed/cifar10.npz")
x_train, y_train = data["x_train"], data["y_train"]

model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Rescaling(1.0 / 255),
    layers.Conv2D(32, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(10, activation="softmax"),
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(params["learning_rate"]),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(x_train, y_train, epochs=params["epochs"],
          batch_size=params["batch_size"], validation_split=0.1)

os.makedirs("models", exist_ok=True)
model.save("models/model.keras")
print("Model saved")
