import warnings
warnings.simplefilter("error")
import tensorflow as tf

# Set random seed for reproducibility
tf.random.set_seed(777)

# Data
x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_data, y_data, epochs=2001, verbose=0)

# Testing & One-hot encoding
a = model.predict([[1, 11, 7, 9]])
print(a, tf.argmax(a, axis=1).numpy())

print('--------------')

b = model.predict([[1, 3, 4, 3]])
print(b, tf.argmax(b, axis=1).numpy())

print('--------------')

c = model.predict([[1, 1, 0, 1]])
print(c, tf.argmax(c, axis=1).numpy())

print('--------------')

all = model.predict([[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]])
print(all, tf.argmax(all, axis=1).numpy())
