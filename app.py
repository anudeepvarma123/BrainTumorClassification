from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
import cv2
from keras.models import load_model
import shutil
import logging
import warnings

# Suppress unnecessary logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=UserWarning, module='keras')

# Configure logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.INFO)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['PREDICT_FOLDER'] = 'static/predict/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PREDICT_FOLDER'], exist_ok=True)

# Load the CNN model from the .h5 file
cnn_model = load_model('cnn_model.h5')

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (32, 32))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/app/')
def app_page():
    return render_template('app.html')

@app.route('/app/braintumor/', methods=['GET', 'POST'])
def braintumor():
    if request.method == 'POST':
        image_file = request.files['image_name']
        if image_file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
            image_file.save(image_path)

            # Preprocess image and predict
            img = preprocess_image(image_path)
            prediction = cnn_model.predict(img)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            tumor_classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
            result = [(tumor_classes[predicted_class], confidence)]

            # Save the predicted image (overwrite if exists)
            predict_path = os.path.join(app.config['PREDICT_FOLDER'], 'predicted_image.jpg')
            if os.path.exists(predict_path):
                os.remove(predict_path)
            shutil.move(image_path, predict_path)

            return render_template('braintumor.html', fileupload=True, report=result, image_size='150px')

    return render_template('braintumor.html', fileupload=False, image_size='150px')

if __name__ == '__main__':
    app.run(debug=True)