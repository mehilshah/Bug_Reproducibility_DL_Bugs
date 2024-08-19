import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
import numpy as np

def set_gpu_config(device="0", fraction=0.25):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=int(fraction * 1000))])
        except RuntimeError as e:
            print(e)

class ODEBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(ODEBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv1 = Conv2D(self.filters, self.kernel_size, padding="same", activation="relu")
        self.conv2 = Conv2D(self.filters, self.kernel_size, padding="same", activation="relu")

    def call(self, inputs):
        def ode_func(x, t):
            return self.conv2(self.conv1(x))

        t = tf.constant([0.0, 1.0], dtype=tf.float32)
        # Using the Runge-Kutta 4th order solver for ODEs
        solver = tfa.metrics.CohenKappa()
        return solver(inputs, t)

def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = ODEBlock(64, (3, 3))(x)
    x = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs)

# Set GPU configuration
# set_gpu_config("0", 0.25)

batch_size = 128
num_classes = 10
epochs = 4
image_shape = (28, 28, 1)

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train = x_train.reshape((-1,) + image_shape)
x_test = x_test.reshape((-1,) + image_shape)

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Build and compile the model
model = build_model(image_shape, num_classes)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
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
