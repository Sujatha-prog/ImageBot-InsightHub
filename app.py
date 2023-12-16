from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import decode_predictions, preprocess_input
from PIL import Image
import os
import numpy as np
import tensorflow as tf

# Set the environment variable to turn off oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)

# Use the v1 API for sparse softmax cross-entropy loss
sparse_softmax_cross_entropy = tf.compat.v1.losses.sparse_softmax_cross_entropy

model = InceptionV3(weights='imagenet')
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index_image.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        prediction = predict_image(file_path)
        return render_template('result_image.html', filename=filename, prediction=prediction)
    else:
        return "Invalid file format. Allowed formats: jpg, jpeg, png."

def predict_image(file_path):
    img = Image.open(file_path)
    img = img.resize((299, 299))
    img_array = preprocess_input(np.array(img))
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions)
    return decoded_predictions[0][0][1]

if __name__ == '__main__':
    app.run(debug=True)
