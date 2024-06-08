import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils
from keras.datasets import mnist

# generate some data
dummyX, dummyY, p_c, p_w_c = make_multilabel_classification(n_samples=4000, n_features=20, n_classes=3, return_distributions=True)

# neural network
model = Sequential()
model.add(Dense(20, input_dim=20))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(20))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(loss='mean_squared_error',
          optimizer='sgd',
          metrics=['accuracy'])

X_train, X_test, y_train, y_test = train_test_split(dummyX, dummyY, test_size=0.20, random_state=42)

model.fit(X_train, y_train,epochs=20, batch_size=30, validation_data=(X_test, y_test))
# Epoch 20/20
# 3200/3200 [==============================] - 0s - loss: 0.2469 - acc: 0.4366 - val_loss: 0.2468 - val_acc: 0.4063
# Out[460]:


# SVM - note that y_train and test are binary label. I haven't included the multi class converter code here for brevity
svm = SVC()
y_train = np.argmax(y_train, axis=1)
y_test = np.argmax(y_test, axis=1)
svm.fit(X_train, y_train)
print (svm.score(X_test, y_test))
# 0.891249