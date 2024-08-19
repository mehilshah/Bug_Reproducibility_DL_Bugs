import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from dataset import getData

img_rows, img_cols = 48, 48

# Define the model
model = Sequential()
model.add(Conv2D(64, (5, 5), padding='valid', input_shape=(img_rows, img_cols, 1)))
model.add(tf.keras.layers.PReLU())
model.add(ZeroPadding2D(padding=(2, 2)))
model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Conv2D(64, (3, 3)))
model.add(tf.keras.layers.PReLU())
model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Conv2D(64, (3, 3)))
model.add(tf.keras.layers.PReLU())
model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Conv2D(128, (3, 3)))
model.add(tf.keras.layers.PReLU())
model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Conv2D(128, (3, 3)))
model.add(tf.keras.layers.PReLU())

model.add(ZeroPadding2D(padding=(1, 1)))
model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(1024))
model.add(tf.keras.layers.PReLU())
model.add(Dropout(0.2))
model.add(Dense(1024))
model.add(tf.keras.layers.PReLU())
model.add(Dropout(0.2))
model.add(Dense(7))
model.add(Activation('softmax'))

# Compile the model
ada = Adadelta(learning_rate=0.1, rho=0.95, epsilon=1e-08)
model.compile(loss='categorical_crossentropy',
              optimizer=ada,
              metrics=['accuracy'])

model.summary()

# Data preparation
batch_size = 32
nb_classes = 7
nb_epoch = 50
img_channels = 1

X, y = getData()

Train_x, Val_x, Train_y, Val_y = train_test_split(X, y, test_size=0.33, random_state=42)

Train_x = np.asarray(Train_x)
Train_x = Train_x.reshape(Train_x.shape[0], img_rows, img_cols)

Val_x = np.asarray(Val_x)
Val_x = Val_x.reshape(Val_x.shape[0], img_rows, img_cols)

Train_x = Train_x.reshape(Train_x.shape[0], img_rows, img_cols, 1)
Val_x = Val_x.reshape(Val_x.shape[0], img_rows, img_cols, 1)

Train_x = Train_x.astype('float32')
Val_x = Val_x.astype('float32')

Train_y = to_categorical(Train_y, nb_classes)
Val_y = to_categorical(Val_y, nb_classes)

# Callbacks
filepath = 'Model.{epoch:02d}-{val_accuracy:.4f}.h5'
checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=False)

datagen.fit(Train_x)

# Train the model
model.fit(datagen.flow(Train_x, Train_y, batch_size=batch_size),
          steps_per_epoch=Train_x.shape[0] // batch_size,
          epochs=nb_epoch,
          validation_data=(Val_x, Val_y),
          callbacks=[checkpointer])
