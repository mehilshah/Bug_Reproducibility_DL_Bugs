import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import pandas as pd


batch_size = 32
epochs = 2
alpha = 0.0001
lambda_ = 0
h1 = 50

train = pd.read_csv('data/mnist_train.csv')
test = pd.read_csv('data/mnist_test.csv')

train = train.loc['1':'5000', :]
test = test.loc['1':'2000', :]

train = train.sample(frac=1).reset_index(drop=True)
test = test.sample(frac=1).reset_index(drop=True)

x_train = train.loc[:, '1x1':'28x28']
y_train = train.loc[:, 'label']

x_test = test.loc[:, '1x1':'28x28']
y_test = test.loc[:, 'label']

x_train = x_train.values
y_train = y_train.values

x_test = x_test.values
y_test = y_test.values

nb_classes = 10
targets = y_train.reshape(-1)
y_train_onehot = np.eye(nb_classes)[targets]

nb_classes = 10
targets = y_test.reshape(-1)
y_test_onehot = np.eye(nb_classes)[targets]

model = tf.keras.Sequential()
model.add(layers.Dense(784, input_shape=(784,)))
model.add(layers.Dense(h1, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(lambda_)))
model.add(layers.Dense(10, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(lambda_)))

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1), 
             loss = 'categorical_crossentropy',
             metrics = ['accuracy'])

model.fit(x_train, y_train_onehot, epochs=epochs, batch_size=batch_size)