import json
import os
import numpy as np
import tensorflow as tf
from argparse import ArgumentParser
from tensorflow.keras import backend
from tensorflow.keras.optimizers import SGD, Adam
from PIL import Image
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Reshape, Conv2D

def save_model(model, name):
    """
    Save model as JSON.
    Args:
        model : tf.keras.Model
        name : str, file name to save
    """
    json_str = model.to_json()
    with open(name, 'w') as f:
        f.write(json_str)

def build_model(input_size):
    input_shape = (input_size[0], input_size[1], 3)
    model = Sequential([
        Conv2D(128, kernel_size=(3, 3), input_shape=input_shape, padding='same'),
        Activation('relu'),
        Conv2D(64, kernel_size=(3, 3), padding='same'),
        Activation('relu'),
        Conv2D(3, kernel_size=(3, 3), padding='same'),
        Activation('sigmoid')
    ])
    return model

def load_images(name, size, ext='.jpg'):
    """
    Load images into an array.
    Args:
        name : str, directory path
        size : tuple, image size
        ext : str, file extension
    Returns:
        images : numpy array, image data
    """
    x_images = []
    y_images = []
    for file in tqdm(os.listdir(name)):
        if os.path.splitext(file)[1] != ext:
            continue
        image = Image.open(os.path.join(name, file))
        if image.mode != "RGB":
            image = image.convert("RGB")
        x_image = image.resize((size[0]//2, size[1]//2))
        x_image = image.resize(size, Image.BICUBIC)
        x_image = np.array(x_image)
        y_image = image.resize(size)
        y_image = np.array(y_image)
        x_images.append(x_image)
        y_images.append(y_image)
    x_images = np.array(x_images) / 255.0
    y_images = np.array(y_images) / 255.0
    return x_images, y_images

def main():
    x_images, y_images = load_images('images/', (128, 128))
    model = build_model((128, 128))
    optimizer = Adam(learning_rate=0.0001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    save_model(model, 'model.json')
    model.fit(x_images, y_images, batch_size=64, epochs=200)
    model.save_weights('weights.h5')  # Updated extension to .h5 for TensorFlow

    x_test, y_test = load_images('images_sample/', (128, 128))
    eva = model.evaluate(x_test, y_test, batch_size=64, verbose=1)
    print(eva)

    loss = eva[0]
    with open("buggy/result.json", "w") as file:
        res = {"loss": float(loss)}
        json.dump(res, file)

if __name__ == '__main__':
    main()
