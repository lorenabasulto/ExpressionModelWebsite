from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000

expressions = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

# Load your trained model
model = load_model('/Users/lore/Desktop/newWebsite/model.h5', compile=False)  # Relative path from server.py to the model file

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join('/Users/lore/Desktop/IndStudy2/', filename)  # Corrected path joining
        file.save(file_path)

        # Open and resize the image
        img = Image.open(file_path).convert('L')  # Open image in grayscale
        img = img.resize((28, 28))  # Resize image to (28, 28)
        img_arr = np.array(img)

        # Scale images to the [0, 1] range
        img_arr = img_arr.astype("float32") / 255

        # Make sure images have shape (28, 28, 1)
        img_arr = np.expand_dims(img_arr, -1)

        # Expand dimension to match model's input shape (1, 28, 28, 1)
        img_arr = np.expand_dims(img_arr, 0)

        # Predict the class of the image
        prediction = model.predict(img_arr)
        predicted_class = np.argmax(prediction)

        print("Prediction is: {} ", prediction)
        print("Predicted class is: {} ", predicted_class)

        return render_template('result.html', prediction=expressions[predicted_class])  # pass predicted_class to the template

if __name__ == '__main__':
    app.run()
