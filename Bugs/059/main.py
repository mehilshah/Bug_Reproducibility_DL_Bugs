import tensorflow as tf
import numpy as np
from PIL import Image
import os

def detect_fn(input_tensor):
    return input_tensor

image_np = np.asarray(np.array(Image.open('sample.jpeg')))
image_np = image_np[..., :3]  # Remove alpha channel
input_tensor = tf.convert_to_tensor(image_np)
input_tensor = input_tensor[tf.newaxis, ...]
detections = detect_fn(input_tensor)