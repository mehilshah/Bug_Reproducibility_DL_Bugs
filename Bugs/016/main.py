from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, Dense, Dropout
import numpy as np
import random

in_shape = (5, 720)
number_of_classes = random.randint(1, 10) 
batch_size = random.choice([32, 64, 128])
epochs = random.randint(1, 10)

x_train = np.random.random((300, 5, 720))
y_train = np.random.randint(number_of_classes, size=(300, number_of_classes))
x_test = np.random.random((100, 5, 720))
y_test = np.random.randint(number_of_classes, size=(100, number_of_classes))

cnn = Sequential()

cnn.add(Conv2D(64, (5, 50),
              padding="same",
              activation="relu",
              data_format="channels_last",
              input_shape=in_shape))

cnn.add(MaxPooling2D(pool_size=(2,2),data_format="channels_last"))
cnn.add(Flatten())
cnn.add(Dense(number_of_classes, activation="softmax"))
cnn.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

cnn.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        shuffle=True)