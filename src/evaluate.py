import json
import os
import numpy as np
import tensorflow as tf

data = np.load("data/processed/cifar10.npz")
model = tf.keras.models.load_model("models/model.keras")
loss, acc = model.evaluate(data["x_test"], data["y_test"], verbose=0)

os.makedirs("metrics", exist_ok=True)
with open("metrics/metrics.json", "w") as f:
    json.dump({"test_loss": float(loss), "test_accuracy": float(acc)}, f, indent=2)
print({"test_loss": loss, "test_accuracy": acc})
