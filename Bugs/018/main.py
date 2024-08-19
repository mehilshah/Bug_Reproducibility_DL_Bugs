import numpy as np
from keras.layers import Dense, Input
from keras import Model
import keras.backend as K

input_layer = Input(shape=(5,))
out = Dense(5)(input_layer)
model = Model(input_layer, out)
train_X = np.random.randn(100, 5)
train_Y = np.random.randn(100, 5)*0.01 + train_X

weights = np.random.randn(*train_X.shape)

def custom_loss_1(y_true, y_pred):
    return K.mean(K.abs(y_true-y_pred)*weights)

model.compile('adam','mean_absolute_error')
model.fit(train_X, train_Y, epochs=1)

model.compile('adam',custom_loss_1)
model.fit(train_X, train_Y, epochs=10)