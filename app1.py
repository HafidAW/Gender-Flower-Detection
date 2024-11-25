from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Validate required files
if not os.path.exists('model/bunga_cnn_model.h5') or not os.path.exists('model/type_model.h5'):
    raise FileNotFoundError("Model files are missing. Ensure 'bunga_cnn_model.h5' and 'type_model.h5' exist in the 'model/' directory.")
if not os.path.exists('model/class_labels.npy'):
    raise FileNotFoundError("Class labels file is missing. Ensure 'class_labels.npy' exists in the 'model/' directory.")

# Load both models
gender_model = tf.keras.models.load_model('model/bunga_cnn_model.h5')
flower_type_model = tf.keras.models.load_model('model/type_model.h5')

# Load class labels for flower types
class_labels = np.load('model/class_labels.npy', allow_pickle=True).item()

# Set known flower classes
known_classes = set(class_labels.values())

# Helper function to check file type
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', error="No file part in the request.")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No file selected.")

    if not allowed_file(file.filename):
        return render_template('index.html', error="Unsupported file type. Please upload an image.")

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Load and preprocess the image
    try:
        img = image.load_img(file_path, target_size=(150, 150))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
    except Exception as e:
        return render_template('index.html', error=f"Image processing failed: {str(e)}")

    # Flower type prediction
    try:
        flower_type_prediction = flower_type_model.predict(img_array)
        predicted_class_index = np.argmax(flower_type_prediction)
        flower_type = class_labels.get(predicted_class_index, 'unknown')

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
            return render_template('index.html', 
                                   file_path=filename, 
                                   flower_type=flower_type, 
                                   class_label=class_label, 
                                   confidence=confidence, 
                                   type_confidence=type_confidence)
        else:
            # Handle unknown flower type
            return render_template('index.html', 
                                   file_path=filename, 
                                   flower_type='unknown', 
                                   class_label='', 
                                   confidence='', 
                                   type_confidence='')

    except Exception as e:
        return render_template('index.html', error=f"Prediction failed: {str(e)}")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        return "File not found", 404
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Error handling for unexpected errors
@app.errorhandler(500)
def internal_server_error(e):
    return render_template('error.html', error="An internal error occurred. Please try again later."), 500

@app.errorhandler(404)
def not_found_error(e):
    return render_template('error.html', error="Page not found."), 404

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
