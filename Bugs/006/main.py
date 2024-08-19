import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import numpy as np

x = np.random.randn(1,19,)
y = np.ones((1,1))

def make_model():
    input_vec = tf.keras.layers.Input((19,))
    final = tf.keras.layers.Dense(12, activation='relu')(input_vec)
    final = tf.keras.layers.Dense(1, activation='sigmoid')(final)

    model = tf.keras.models.Model(inputs=input_vec, outputs=final)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

model = make_model()

model.fit(x, y, batch_size=1, epochs = 10, validation_data=[x,y])