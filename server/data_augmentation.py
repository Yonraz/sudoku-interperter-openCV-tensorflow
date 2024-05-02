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

# Data augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range=0.1,     # randomly zoom image
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False    # randomly flip images
)

datagen.fit(x_train.reshape(-1, 28, 28, 1))

# Create the model
model = tf.keras.models.Sequential() # sequential is the simplest model, a feed forward model which is just a stack of layers that are executed in order
model.add(tf.keras.layers.Flatten(input_shape=(28, 28))) # flatten the input, which is a 28x28 image, to a 1D array of 784 elements
model.add(tf.keras.layers.Dense(128, activation='relu')) # add a dense layer with 128 neurons, and relu activation function
model.add(tf.keras.layers.Dense(128, activation='relu')) # add another dense layer with 128 neurons, and relu activation function
model.add(tf.keras.layers.Dense(10, activation='softmax')) # add the output layer with 10 neurons, and softmax activation function

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) 

# Train the model with augmented data
model.fit(datagen.flow(x_train.reshape(-1, 28, 28, 1), y_train, batch_size=32),
          steps_per_epoch=int(len(x_train) / 32), epochs=10)

# Save the model
model.save('server/mnist_da_model.keras')