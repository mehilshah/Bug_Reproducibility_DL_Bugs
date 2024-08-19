# -*- coding: utf-8 -*-
################################################################################################
# reference : https://elitedatascience.com/keras-tutorial-deep-learning-in-python
################################################################################################
import numpy as np                  # NumPy
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential # Updated Keras import
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

np.random.seed(123)                 # Set random seed for reproducibility

# Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Reshape and normalize the data
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255

print(y_train[:10])
Y_train = to_categorical(y_train, 10)
print(Y_train.shape)
Y_test = to_categorical(y_test, 10)

# Build the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
print(model.output_shape)
model.add(Conv2D(32, (3, 3), activation='relu'))
print(model.output_shape)
model.add(MaxPooling2D(pool_size=(2, 2)))
print(model.output_shape)
model.add(Dropout(0.25))
print(model.output_shape)
model.add(Flatten())
print(model.output_shape)
model.add(Dense(128, activation='relu'))
print(model.output_shape)
model.add(Dropout(0.5))
print(model.output_shape)
model.add(Dense(10, activation='softmax'))
print(model.output_shape)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print(model.input_shape)
print(model.output_shape)

# Train the model
model.fit(X_train, Y_train,
          batch_size=32, epochs=10, verbose=2)  # Adjusted batch size for practical purposes
