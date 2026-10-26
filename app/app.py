import io
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template_string, request
from PIL import Image

app = Flask(__name__)
MODEL = tf.keras.models.load_model("models/model.keras")
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

PAGE = """
<!doctype html>
<title>Image Classifier</title>
<body style="font-family:sans-serif;max-width:520px;margin:40px auto;text-align:center">
  <h2>CIFAR-10 Image Classifier</h2>
  <p>Upload an image of: {{ classes }}</p>
  <form method="post" enctype="multipart/form-data">
    <input type="file" name="file" accept="image/*" required>
    <button type="submit">Predict</button>
  </form>
  {% if label %}
    <h3>Prediction: {{ label }} ({{ conf }}%)</h3>
  {% endif %}
  {% if error %}<p style="color:red">{{ error }}</p>{% endif %}
</body>
"""

def predict(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB").resize((32, 32))
    arr = np.expand_dims(np.array(img, dtype="float32"), 0)
    probs = MODEL.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    return CLASSES[idx], round(float(probs[idx]) * 100, 2)

@app.route("/", methods=["GET", "POST"])
def index():
    label = conf = error = None
    if request.method == "POST":
        try:
            label, conf = predict(request.files["file"].read())
        except Exception as e:
            error = f"Could not process image: {e}"
    return render_template_string(PAGE, label=label, conf=conf,
                                  error=error, classes=", ".join(CLASSES))

@app.route("/health")
def health():
    return jsonify(status="ok")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
