import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D

# create model
model = Sequential()

# Scenario 1
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(None, 286, 384, 1)))

data = np.random.random((100, 286, 384))

model.predict(data)