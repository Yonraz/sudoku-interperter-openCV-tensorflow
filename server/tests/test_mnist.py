import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import os


# Load mnist data set
# mnist = tf.keras.datasets.mnist
# # split mnist into training and testing data
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# # normalize the data (transform 0-255 values to 0-1, to make it easier for the model to learn)
# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.load_model('mnist_model.keras')

# loss, accuracy = model.evaluate(x_test, y_test)
# print('Test accuracy:', accuracy)
# print('Test loss:', loss)
image_num = 1
# iterate through digits folder

def get_prediction(img, model):
    inv_img = np.invert(img)
    resized = cv2.resize(inv_img, (28, 28))
    flattened = resized / 255.0

    # Reshape the image to match the model's input shape
    input_data = np.expand_dims(flattened, axis=0)
    confidence_threshold = 0.9


    probabilities = model.predict(input_data)
    prediction = np.argmax(probabilities)
    confidence = probabilities[0][prediction]
    if confidence < confidence_threshold:
        return 0
    return np.argmax(probabilities)

while os.path.exists(f'../../digits/digit_{image_num}.png'):
    img = cv2.imread(f'../../digits/digit_{image_num}.png', cv2.IMREAD_GRAYSCALE)
    prediction = get_prediction(img, model)
    print(f"digit_{image_num}.png: {prediction}")
    image_num += 1