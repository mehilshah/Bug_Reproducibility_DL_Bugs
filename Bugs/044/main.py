import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation, Flatten
from keras.optimizers import Adam

# I try to train a model, input is (3000,1) vector that is consist of negative numbers mostly, inormalize input. Output is binary image which is represented as vector (2500,1).
x_train = np.random.rand(3000, 1)
y_train = np.random.randint(2, size=(3000, 1))

model = Sequential()
model.add(Dense(3000, input_shape=(x_train.shape[1:]), activation='linear'))
model.add(Dense(2500, activation='relu'))
model.add(Dense(2500, activation='relu'))
model.add(Dense(2500, activation='relu'))
model.add(Dense(2500, activation='relu'))
model.add(Dense(y_train.shape[1], activation='sigmoid'))
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Just 1 vector in the epoch
model.fit(x_train, y_train, epochs=1, batch_size=1, validation_split=0.2)