from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import io
from PIL import Image

app = Flask(__name__)
CORS(app)  # Enable CORS for the entire app

# Load the saved model
model = load_model('MobileNetV2.keras')

# Define the target image size
IMAGE_SIZE = (224, 224)

# Class labels
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

def load_and_preprocess_image(file):
    try:
        img = Image.open(io.BytesIO(file.read()))
        img = img.resize(IMAGE_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

@tf.function
def predict_class(img_array):
    predictions = model(img_array)
    predicted_class_index = tf.argmax(predictions, axis=1)[0]
    return predicted_class_index

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    try:
        img_array = load_and_preprocess_image(file)

        if img_array is not None:
            predicted_class_index = predict_class(img_array).numpy()
            predicted_class = CLASS_NAMES[predicted_class_index]
            return jsonify({'class': predicted_class}), 200
        else:
            return jsonify({'error': 'Image processing failed'}), 500
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': 'Prediction failed'}), 500

if __name__ == '__main__':
    app.run(debug=True)
