from PIL import Image
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras import layers

# **** Prepare the Data ******
num_classes = 7  
input_shape = (28, 28, 1)
expressions = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']  

# Initialize the arrays
x_train = []
y_train = []
x_test = []
y_test = []


for i, expression in enumerate(expressions):
    train_path = f'CK+48/{expression}'
    #test_path = f'CK+48/anger'    
    test_path = f'CK+48/{expression}_test'  

   
    for image_name in os.listdir(train_path):
        if image_name in ['.DS_Store', 'Thumbs.db', '11702.jpg']:
            continue
        image_path = os.path.join(train_path, image_name)

        image = Image.open(image_path).convert('L')
        image = image.resize((28, 28))
        x_train.append(np.array(image))
        y_train.append(i)  # Label for the current class

    # Load test images
    for image_name in os.listdir(test_path):
        if image_name in ['.DS_Store', 'Thumbs.db', '11702.jpg']:
            continue
        image_path = os.path.join(test_path, image_name)

        image = Image.open(image_path).convert('L')
        image = image.resize((28, 28))
        x_test.append(np.array(image))
        y_test.append(i)  # Label for the current class

# Convert to numpy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# **** Build the Model ****
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()


# **** Train the Model ****
batch_size = 128
epochs = 15

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
score = model.evaluate(x_test, y_test, verbose=0)

# After model is trained
model.save("model.h5")  # Save the model
print("Test loss:", score[0])
print("Test accuracy:", score[1])
def classify_image(image_path):
    # Load the image
    image = Image.open(image_path).convert('L')  # Open image in grayscale
    image = image.resize((28, 28))  # Resize image to (28, 28)
    image_arr = np.array(image)

    # Scale image to the [0, 1] range
    image_arr = image_arr.astype("float32") / 255

    # Make sure image has shape (28, 28, 1)
    image_arr = np.expand_dims(image_arr, -1)

    # Expand dimension to match model's input shape (1, 28, 28, 1)
    image_arr = np.expand_dims(image_arr, 0)

    # Predict the class of the image
    prediction = model.predict(image_arr)

    # Get the predicted class label
    predicted_class = np.argmax(prediction)

    return predicted_class

print(classify_image('/Users/lore/Desktop/newWebsite/CK+48/disgust/S005_001_00000010.png'))
