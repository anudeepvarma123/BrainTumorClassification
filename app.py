from flask import Flask, render_template, request, redirect, url_for, flash
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
app.secret_key = 'your_secret_key'  # Needed for flashing messages
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['PREDICT_FOLDER'] = 'static/predict/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PREDICT_FOLDER'], exist_ok=True)

# Load model
cnn_model = load_model('cnn_model.h5')

# Labels
tumor_classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Preprocess function
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (32, 32))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Check if likely MRI
def is_likely_mri(image, grayscale_threshold=0.9):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(image)
    diff_bg = np.mean(np.abs(b.astype("float") - g.astype("float")))
    diff_br = np.mean(np.abs(b.astype("float") - r.astype("float")))
    diff_gr = np.mean(np.abs(g.astype("float") - r.astype("float")))
    color_diff = (diff_bg + diff_br + diff_gr) / 3.0
    return color_diff < grayscale_threshold

# Route
@app.route('/app/braintumor/', methods=['GET', 'POST'])
def braintumor():
    if request.method == 'POST':
        image_file = request.files['image_name']
        if image_file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
            image_file.save(image_path)

            image = cv2.imread(image_path)
            if not is_likely_mri(image):
                flash('Uploaded image does not look like an MRI. Please upload a valid MRI image.', 'warning')
                os.remove(image_path)
                return redirect(request.url)

            # Predict
            img = preprocess_image(image_path)
            prediction = cnn_model.predict(img, verbose=0)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)

            if confidence < 0.80:
                flash(f'Prediction confidence is low ({confidence*100:.2f}%). Please upload a clearer MRI.', 'warning')
                os.remove(image_path)
                return redirect(request.url)

            result = [(tumor_classes[predicted_class], confidence * 100)]

            # Save predicted image
            predict_path = os.path.join(app.config['PREDICT_FOLDER'], 'predicted_image.jpg')
            if os.path.exists(predict_path):
                os.remove(predict_path)
            shutil.move(image_path, predict_path)

            return render_template('braintumor.html', fileupload=True, report=result, image_size='150px')

    return render_template('braintumor.html', fileupload=False, image_size='150px')

if __name__ == '__main__':
    app.run(debug=True)
