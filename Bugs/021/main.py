import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Dense

x = np.random.rand(10000, 28, 28)
y = np.random.rand(10000)
x = x.reshape(-1, 28, 28, 1)

model = Sequential([
    Conv2D(8, kernel_size=(3, 3), padding="same", activation="relu", input_shape=(28, 28, 1)),
    Dense(64, activation="relu"),
    Dense(64, activation="relu"),
    Dense(10, activation="softmax")
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x, y, epochs=5)