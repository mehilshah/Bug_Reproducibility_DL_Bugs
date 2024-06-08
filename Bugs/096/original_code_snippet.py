#Import Libraries
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.optimizers import SGD

#model details
vgg19 = Sequential()
vgg19.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
vgg19.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
vgg19.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
vgg19.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
vgg19.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
vgg19.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
vgg19.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
vgg19.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
vgg19.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
vgg19.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
vgg19.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
vgg19.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
vgg19.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
vgg19.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
vgg19.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
vgg19.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
vgg19.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
vgg19.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
vgg19.add(Flatten())
vgg19.add(Dense(units=4096,activation="relu"))
vgg19.add(Dense(units=4096,activation="relu"))
vgg19.add(Dense(units=10, activation="softmax"))

#Preparing Dataset
from keras.datasets import cifar10
from keras.utils import to_categorical

(X, Y), (tsX, tsY) = cifar10.load_data() 
# Use a one-hot-encoding
Y = to_categorical(Y)
tsY = to_categorical(tsY)
# Change datatype to float
X = X.astype('float32')
tsX = tsX.astype('float32')
 
# Scale X and tsX so each entry is between 0 and 1
X = X / 255.0
tsX = tsX / 255.0

#training
optimizer = SGD(lr=0.001, momentum=0.9)
vgg19.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
history = vgg19.fit(X, Y, epochs=100, batch_size=64, validation_data=(tsX, tsY), verbose=0)