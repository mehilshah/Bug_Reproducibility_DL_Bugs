import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD

# Optional: import plotting library if you want to visualize training history
# import matplotlib.pyplot as plt

if __name__ == '__main__':

    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    outputs = np.array([[0], [1], [1], [0]])

    model = Sequential()
    model.add(Dense(units=8, input_dim=2, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=SGD(learning_rate=0.1),
                  metrics=['accuracy'])

    history = model.fit(inputs, outputs, batch_size=1, epochs=1000, verbose=1)

    # Optional: Plot training history
    # plt.plot(history.history['accuracy'])
    # plt.title('Model accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['Train'], loc='upper left')
    # plt.savefig('history.png')

    print(model.predict(inputs))

    # Optional: Save the model
    # model.save('XOR/XOR_MODEL.h5')
