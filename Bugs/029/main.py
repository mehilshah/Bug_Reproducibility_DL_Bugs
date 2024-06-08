import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import matplotlib.pyplot as plt

#datapoints
X = np.arange(0.0, 5.0, 0.1, dtype='float32').reshape(-1,1)
y = 5 * np.power(X,2) + np.power(np.random.randn(50).reshape(-1,1),3)

#model
model = Sequential()
model.add(Dense(50, activation='relu', input_dim=1))
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='linear'))

#training
sgd = SGD(lr=0.1)
model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])
model.fit(X, y, epochs=1000)

#predictions
predictions = model.predict(X)

#plot
plt.scatter(X, y,edgecolors='g')
plt.plot(X, predictions,'r')
plt.legend([ 'Predictated Y' ,'Actual Y'])
plt.show()