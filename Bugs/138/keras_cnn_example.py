# -*- coding: utf-8 -*-
################################################################################################
# reference : https://elitedatascience.com/keras-tutorial-deep-learning-in-python
################################################################################################
import numpy as np                  # NumPy
from matplotlib import pyplot as plt
from keras.models import Sequential # 이하 Keras 모듈들입니다.
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D
from keras.utils import np_utils
from keras.datasets import mnist

np.random.seed(123)                 # 랜덤시드를 지정하면, 재실행시에도 같은 랜덤값을 추출합니다(reproducibility)
# Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
#plt.imshow(X_train[0])
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
print(X_train.shape)
#plt.imshow(X_train[0, 0])
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print(y_train[:10])
Y_train = np_utils.to_categorical(y_train, 10)
print(Y_train.shape)
Y_test = np_utils.to_categorical(y_test, 10)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(1,28,28)))
print(model.output_shape)
model.add(Conv2D(32, (3, 3), activation='relu'))
print(model.output_shape)
model.add(MaxPool2D(pool_size=(2,2)))
print(model.output_shape)
model.add(Dropout(0.25))
print(model.output_shape)
model.add(Flatten())
print(model.output_shape)
model.add(Dense(128, activation='relu'))
print(model.output_shape)
model.add(Dropout(0.5))
print(model.output_shape)
model.add(Dense(10, activation='softmax'))
print(model.output_shape)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print(model.input_shape)
print(model.output_shape)

model.fit(X_train, Y_train,
          batch_size=2, epochs=10, verbose=2)
dummy = 0

