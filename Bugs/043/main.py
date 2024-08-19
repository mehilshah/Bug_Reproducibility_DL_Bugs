import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense

n_steps = 10
n_features = 1
X = np.random.rand(4000, n_steps, n_features)
y = np.random.randint(2, size=(4000, 1))

model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)