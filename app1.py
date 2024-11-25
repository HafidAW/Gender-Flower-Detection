from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Flask setup
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Debugging log
print("Starting application...")

# Verify model directory
MODEL_FOLDER = 'model'
if not os.path.exists(MODEL_FOLDER):
    raise FileNotFoundError(f"Model directory '{MODEL_FOLDER}' not found.")

# Load models with error handling
try:
    print("Loading gender model...")
    gender_model = tf.keras.models.load_model(os.path.join(MODEL_FOLDER, 'bunga_cnn_model.h5'))
    print("Gender model loaded successfully.")
except Exception as e:
    print(f"Failed to load gender model: {e}")
    raise FileNotFoundError("Ensure 'bunga_cnn_model.h5' exists in the 'model' directory.")

try:
    print("Loading flower type model...")
    flower_type_model = tf.keras.models.load_model(os.path.join(MODEL_FOLDER, 'type_model.h5'))
    print("Flower type model loaded successfully.")
except Exception as e:
    print(f"Failed to load flower type model: {e}")
    raise FileNotFoundError("Ensure 'type_model.h5' exists in the 'model' directory.")

# Load class labels
try:
    print("Loading class labels...")
    class_labels = np.load(os.path.join(MODEL_FOLDER, 'class_labels.npy'), allow_pickle=True).item()
    print("Class labels loaded successfully.")
except Exception as e:
    print(f"Failed to load class labels: {e}")
    raise FileNotFoundError("Ensure 'class_labels.npy' exists in the 'model' directory.")

# Set known flower classes
known_classes = set(class_labels.values())

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Load and preprocess the image
        img = image.load_img(file_path, target_size=(150, 150))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Flower type prediction
        flower_type_prediction = flower_type_model.predict(img_array)
        predicted_class_index = np.argmax(flower_type_prediction) + 1
        flower_type = class_labels.get(predicted_class_index, 'unknown')

        # Check if the flower type is recognized
        if flower_type in known_classes:
            type_confidence = round(np.max(flower_type_prediction) * 100, 2)

            # Gender prediction
            gender_prediction = gender_model.predict(img_array)
            if gender_prediction[0][0] < 0.5:
                class_label = 'Male'
                confidence = round((1 - gender_prediction[0][0]) * 100, 2)
            else:
                class_label = 'Female'
                confidence = round(gender_prediction[0][0] * 100, 2)

            # Render template with predictions
            return render_template(
                'index.html',
                file_path=filename,
                flower_type=flower_type,
                class_label=class_label,
                confidence=confidence,
                type_confidence=type_confidence,
            )
        else:
            # Handle unknown flower type
            return render_template(
                'index.html',
                file_path=filename,
                flower_type='unknown',
                class_label='N/A',
                confidence='N/A',
                type_confidence='N/A',
            )

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Ensure upload folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
