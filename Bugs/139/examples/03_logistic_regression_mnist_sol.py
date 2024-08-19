import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import time
import json

# Define parameters for the model
learning_rate = 0.01
batch_size = 128
n_epochs = 30

# Load and preprocess data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 784).astype('float32') / 255
X_test = X_test.reshape(-1, 784).astype('float32') / 255
Y_train = to_categorical(y_train, 10)
Y_test = to_categorical(y_test, 10)

# Create TensorFlow dataset objects
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(batch_size)

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01))
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Training loop
start_time = time.time()
history = model.fit(train_dataset, epochs=n_epochs, verbose=2)

# Calculate total training time
print(f'Total time: {time.time() - start_time} seconds')

# Evaluate the model
loss, accuracy = model.evaluate(test_dataset, verbose=2)
print(f'Accuracy: {accuracy}')

# Save the result to a JSON file
result = {"accuracy": float(accuracy)}
with open('fixed/result.json', 'w') as file:
    json.dump(result, file)
