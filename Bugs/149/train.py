# from keras.models import Sequential
# # from keras.layers.normalization import BatchNormalization
# # from keras.layers.convolutional import Conv2D
# # from keras.layers.convolutional import MaxPooling2D
# # from keras.layers.core import Activation
# # from keras.layers.core import Flatten
# # from keras.layers.core import Dropout
# # from keras.layers.core import Dense
# # from keras import backend as K

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adadelta
from keras.utils import np_utils
from keras.regularizers import l2 #, activity_l2
import keras
from dataset import getData
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import numpy
img_rows, img_cols = 48, 48
model = Sequential()
model.add(Convolution2D(64, 5, 5, border_mode='valid',
                        input_shape=(img_rows, img_cols, 1)))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(2, 2), dim_ordering='tf'))
model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
model.add(Convolution2D(64, 3, 3))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
model.add(Convolution2D(64, 3, 3))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(keras.layers.convolutional.AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
model.add(Convolution2D(128, 3, 3))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
model.add(Convolution2D(128, 3, 3))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))

model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
model.add(keras.layers.convolutional.AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(1024))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(Dropout(0.2))
model.add(Dense(1024))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(Dropout(0.2))

model.add(Dense(7))

model.add(Activation('softmax'))

ada = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
model.compile(loss='categorical_crossentropy',
              optimizer=ada,
              metrics=['accuracy'])
model.summary()
batch_size = 32
nb_classes = 7
nb_epoch = 50
img_channels = 1

X,y = getData()


Train_x, Val_x, Train_y, Val_y  = train_test_split(X, y, test_size=0.33, random_state=42)

Train_x = numpy.asarray(Train_x)
Train_x = Train_x.reshape(Train_x.shape[0],img_rows,img_cols)

Val_x = numpy.asarray(Val_x)
Val_x = Val_x.reshape(Val_x.shape[0],img_rows,img_cols)

Train_x = Train_x.reshape(Train_x.shape[0], img_rows, img_cols,1)
Val_x = Val_x.reshape(Val_x.shape[0], img_rows, img_cols,1)

Train_x = Train_x.astype('float32')
Val_x = Val_x.astype('float32')


Train_y = np_utils.to_categorical(Train_y, nb_classes)
Val_y = np_utils.to_categorical(Val_y, nb_classes)




filepath='Model.{epoch:02d}-{val_acc:.4f}.hdf5'
checkpointer = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=40,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

datagen.fit(Train_x)

model.fit_generator(datagen.flow(Train_x, Train_y,
                    batch_size=batch_size),
                    samples_per_epoch=Train_x.shape[0],
                    nb_epoch=nb_epoch,
                    validation_data=(Val_x, Val_y),
                    callbacks=[checkpointer],
                    )