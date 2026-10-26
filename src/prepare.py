import os
import numpy as np
import yaml
from tensorflow.keras.datasets import cifar10

params = yaml.safe_load(open("params.yaml"))["prepare"]
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train, y_train = x_train[: params["train_samples"]], y_train[: params["train_samples"]]
x_test, y_test = x_test[: params["test_samples"]], y_test[: params["test_samples"]]

os.makedirs("data/processed", exist_ok=True)
np.savez_compressed(
    "data/processed/cifar10.npz",
    x_train=x_train, y_train=y_train.flatten(),
    x_test=x_test, y_test=y_test.flatten(),
)
print("Saved", x_train.shape, x_test.shape)
