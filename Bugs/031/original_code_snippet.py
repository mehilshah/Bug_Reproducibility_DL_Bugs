# keras.datasets import mnist
(x_tr, y_tr), (x_te, y_te) = mnist.load_data()
print (x_tr.shape)

X_train = numpy.array([[1] * 128] * (10 ** 4) + [[0] * 128] * (10 ** 4))
X_test = numpy.array([[1] * 128] * (10 ** 2) + [[0] * 128] * (10 ** 2))

Y_train = numpy.array([True] * (10 ** 4) + [False] * (10 ** 4))
Y_test = numpy.array([True] * (10 ** 2) + [False] * (10 ** 2))

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

Y_train = Y_train.astype("bool")
Y_test = Y_test.astype("bool")

model = Sequential()
model.add(Dense(128, 50))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(50, 50))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(50, 1))
model.add(Activation('softmax'))

rms = RMSprop()
model.compile(loss='binary_crossentropy', optimizer=rms)

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          show_accuracy=True, verbose=2, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])