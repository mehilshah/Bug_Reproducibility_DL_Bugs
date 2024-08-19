import os
import numpy as np
from argparse import ArgumentParser
from tensorflow.keras.models import model_from_json
from PIL import Image
import tensorflow as tf

def load_image(name, size):
    """
    Load and preprocess an image.
    Args:
        name : str, file path
        size : tuple, target image size
    Returns:
        image : numpy array, preprocessed image data
    """
    image = Image.open(name)
    image = image.resize((size[0]//2, size[1]//2))
    image = image.resize(size, Image.NEAREST)
    image = np.array(image)
    image = image / 255.0
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image

def load_model(name):
    """
    Load a model from a JSON file.
    Args:
        name : str, file path of the JSON file
    Returns:
        tf.keras.Model
    """
    with open(name) as f:
        json_str = f.read()
    model = model_from_json(json_str)
    return model

def show(image):
    """
    Convert image data to a format suitable for displaying.
    Args:
        image : numpy array, image data
    """
    image = image.squeeze()  # Remove batch dimension
    image = image * 255.0
    image = image.astype(np.uint8)
    image = Image.fromarray(image)
    image.show()

def main():
    model = load_model('model.json')
    model.load_weights('weights.h5')  # Updated extension to .h5
    print('Enter the file name (*.jpg)')
    while True:
        values = input('>> ').rstrip()
        if not os.path.isfile(values):
            print('File does not exist')
            continue
        image = load_image(name=values, size=(128, 128))
        prediction = model.predict(image)
        show(prediction[0])

if __name__ == '__main__':
    main()
