from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os


# Load your trained model
model = load_model('/Users/lore/Desktop/newWebsite/model.h5', compile=False)  # Relative path from server.py to the model file

expressions = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']  
# for i, expression in enumerate(expressions):
pic_dir = f'CK+48/'

for image in os.listdir(pic_dir):
    if image:

        image_path = os.path.join(pic_dir, image)

        # Open and resize the image
        img = Image.open(image_path).convert('L')  # Open image in grayscale
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

        # print("Prediction is: {} ", prediction)
        print("Predicted class is: {} ", predicted_class)

            # print(prediction)

        