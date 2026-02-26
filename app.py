from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model('plant_model.h5')
with open('labels.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']

    # Save temporarily
    upload_dir = 'uploads'
    os.makedirs(upload_dir, exist_ok=True)
    filepath = os.path.join(upload_dir, file.filename)
    file.save(filepath)

    try:
        # Preprocess image
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)
        index = np.argmax(predictions)
        disease_name = labels[index]
        confidence = float(np.max(predictions))

        return jsonify({
            'disease': disease_name,
            'confidence': f"{confidence*100:.1f}%",
            'solution': get_solution(disease_name)
        })
    except Exception as e:
        return jsonify({'error': str(e)})

def get_solution(name):
    solutions = {
        "Tomato_Blight": "Apply copper-based fungicide and remove lower leaves.",
        "Corn_Rust": "Use sulfur-based spray. Ensure better air circulation.",
        "Healthy": "No disease detected. Continue current irrigation."
    }
    return solutions.get(name, "Consult an agronomist for specialized treatment.")

if __name__ == '__main__':
    app.run(debug=True)