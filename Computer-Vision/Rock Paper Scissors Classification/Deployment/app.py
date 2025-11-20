from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import io
# Load Model
MODEL_PATH = "model/rockpaperscissors.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Initialize Flask app
app = Flask(__name__)

# Endpoint untuk prediksi
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ambil data input berupa gambar
        file = request.files["file"]
        img = tf.keras.utils.load_img(io.BytesIO(file.read()), target_size=(100, 150))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalisasi gambar

        # Prediksi menggunakan model
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]

        # Mapping hasil prediksi
        class_names = ["Paper", "Rock", "Scissors"]
        result = {"class": class_names[predicted_class], "confidence": float(np.max(predictions))}

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

# Endpoint test
@app.route("/", methods=["GET"])
def home():
    return "RPS Classification API is running!"

# Jalankan aplikasi
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
