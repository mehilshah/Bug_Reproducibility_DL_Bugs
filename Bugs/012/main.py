from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

# Generate random input data
X_train = np.random.rand(118, 45, 2)

# Generate target data with the wrong shape
y_train = np.random.rand(118, 1)

model = Sequential()
model.add( LSTM( 512, input_shape=(45, 2), return_sequences=True))
model.add( LSTM( 512, return_sequences=True))
model.add( (Dense(1)))
model.compile(loss='mse', optimizer='adam')
history = model.fit( X_train, y_train, batch_size = 10, epochs=2, shuffle = False)