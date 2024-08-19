import numpy as np
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense

dataframe = read_csv("sonar.all-data", header=None)
dataset = dataframe.values

X = dataset[:,0:60].astype(float)
Y = dataset[:,60]
Y[Y=='R'] = 0
Y[Y=='M'] = 1

X_list = X.tolist()
Y_list = Y.tolist()

model  = Sequential()
model.add(Dense(5, input_dim=len(X_list[0]), activation='sigmoid' ))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc'])
model.fit(X_list, Y_list, epochs=20, batch_size=10)