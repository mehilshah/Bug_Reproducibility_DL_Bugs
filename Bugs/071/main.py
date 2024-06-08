import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

model = Sequential()
model.add(Dense(78, activation='relu', input_shape = (3,)))
model.add(Dense(54, activation='relu'))
model.add(Dense(54, activation='relu'))
model.add(Dense(5))

print (model.summary())

inputs = keras.layers.Input(shape=3) #(X.shape[1],)
out = keras.layers.Dense(78, activation='relu')(inputs)
out = keras.layers.Dense(54, activation='relu')(out)
out = keras.layers.Dense(54, activation='relu')(out)
out = keras.layers.Dense(5, activation='relu')(out)

# Summary of the model
model = keras.models.Model(inputs=inputs, outputs=out)
print (model.summary())

