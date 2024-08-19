#!/usr/bin/env python3
# coding=UTF-8

'''Trains a simple convnet on a custom dataset.

Gets to high accuracy with appropriate tuning.
'''

from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import numpy as np
import glob
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import json

batch_size = 200
num_classes = 2
epochs = 10

# Input image dimensions
img_rows, img_cols = 64, 64
img_channels = 3

# Load custom data
folder = "catsdogsdataset64"
dogs_folder = "dogs64"
cats_folder = "cats64"

def load_images_from_folder(folder):
    images = []
    for filename in glob.glob(folder + '/*'):
        img = load_img(filename, target_size=(img_rows, img_cols))
        img = img_to_array(img)
        images.append(img)
    return np.array(images)

train_dogs = load_images_from_folder(folder + '/' + dogs_folder)
train_dogs_y = np.zeros((len(train_dogs), 1))

test_dogs = load_images_from_folder(folder + '/' + dogs_folder + '_test')
test_dogs_y = np.zeros((len(test_dogs), 1))

train_cats = load_images_from_folder(folder + '/' + cats_folder)
train_cats_y = np.ones((len(train_cats), 1))

test_cats = load_images_from_folder(folder + '/' + cats_folder + '_test')
test_cats_y = np.ones((len(test_cats), 1))

train_both = np.concatenate((train_dogs, train_cats), axis=0)
train_both_y = np.concatenate((train_dogs_y, train_cats_y), axis=0)
test_both = np.concatenate((test_dogs, test_cats), axis=0)
test_both_y = np.concatenate((test_dogs_y, test_cats_y), axis=0)

# Shuffle data
np.random.seed(1)
indices = np.arange(train_both.shape[0])
np.random.shuffle(indices)
train_both = train_both[indices]
train_both_y = train_both_y[indices]

np.random.seed(2)
indices = np.arange(test_both.shape[0])
np.random.shuffle(indices)
test_both = test_both[indices]
test_both_y = test_both_y[indices]

# Data preprocessing
x_train = train_both
y_train = train_both_y
x_test = test_both
y_test = test_both_y

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices (one-hot encoding)
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Create and compile model
model = Sequential()
model.add(Conv2D(32, kernel_size=(7, 7), activation='relu', input_shape=(img_rows, img_cols, img_channels)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(20, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(20, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(20, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

# Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save results to JSON file
with open("result.json", mode="w") as file:
    model_accuracy = np.float64(score[1])
    res = {"accuracy": model_accuracy}
    json.dump(res, file)
