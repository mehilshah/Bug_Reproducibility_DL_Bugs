from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD
import pandas as pd
import numpy as np

dataset = pd.read_csv('data/dataset.csv')
numerical_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()
dataset = dataset[numerical_cols + ['Winner']]
X_train = dataset.iloc[:, :-1].values
y_train = dataset.iloc[:, -1].values
X_train = X_train.astype('float32')
y_train = np.array([0 if x == 'Barack Obama' else 1 for x in y_train])

model = Sequential()
model.add(Dense(64, input_dim=6,))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)
model.fit(X_train, y_train, epochs=20, batch_size=16)