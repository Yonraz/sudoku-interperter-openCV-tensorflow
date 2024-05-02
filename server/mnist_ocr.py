## loading mnist data set, and training the model using KNN
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

# Load mnist data set
mnist = tf.keras.datasets.mnist
# split mnist into training and testing data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# normalize the data (transform 0-255 values to 0-1, to make it easier for the model to learn)
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Create the model
model = tf.keras.models.Sequential() # sequential is the simplest model, a feed forward model which is just a stack of layers that are executed in order
model.add(tf.keras.layers.Flatten(input_shape=(28, 28))) # flatten the input, which is a 28x28 image, to a 1D array of 784 elements
model.add(tf.keras.layers.Dense(128, activation='relu')) # add a dense layer with 128 neurons, and relu activation function
model.add(tf.keras.layers.Dense(128, activation='relu')) # add another dense layer with 128 neurons, and relu activation function
model.add(tf.keras.layers.Dense(10, activation='softmax')) # add the output layer with 10 neurons, and softmax activation function

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) 

# Train the model
model.fit(x_train, y_train, epochs=10)

# Save the model
model.save('servermnist_model.keras')