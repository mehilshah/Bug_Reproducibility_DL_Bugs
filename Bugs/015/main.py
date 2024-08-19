import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D, BatchNormalization, LeakyReLU, Dropout, Flatten, Dense, Activation
from keras.callbacks import ReduceLROnPlateau

X_train = np.random.rand(100, 10, 4)  # Reshape X_train to have shape (100, 10, 4)
y_train = np.random.randint(0, 2, size=100)
y_train = y_train.reshape(-1, 1)
X_test = np.random.rand(*X_train.shape)
y_test = np.random.randint(0, 2, size=100)
y_test = y_test.reshape(-1, 1)

model = Sequential()
model.add(Conv1D(input_shape = (10, 4),
                        filters=16,
                        kernel_size=4))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(0.2))

model.add(Conv1D(filters=8,
                        kernel_size=4))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(64))
model.add(BatchNormalization())
model.add(LeakyReLU())

model.add(Dense(1))
model.add(Activation('softmax'))

reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=30, min_lr=0.000001, verbose=0)

model.compile(optimizer='adam', 
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, 
          epochs = 100, 
          batch_size = 128, 
          verbose=0,
          callbacks=[reduce_lr],
          shuffle=True)

y_pred = model.predict(X_test)
print (y_pred)