model = Sequential()
from keras.layers import Dropout
model.add(Conv2D(96, kernel_size=11, padding="same", input_shape=(300, 300, 1), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

model.add(Conv2D(128, kernel_size=3, padding="same", activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(128, kernel_size=3, padding="same", activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

from keras.layers.core import Activation

model.add(Flatten())
# model.add(Dense(units=1000, activation='relu'  ))
model.add(Dense(units= 300, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1))
model.add(Activation("softmax"))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    featurewise_center=True,
    rotation_range=90,
    fill_mode='nearest',
    validation_split = 0.2
    )

datagen.fit(train)

train_generator = datagen.flow(train, train_labels, batch_size=8)


# # fits the model on batches with real-time data augmentation:
history = model.fit_generator(generator=train_generator,
                    use_multiprocessing=True,
                    steps_per_epoch = len(train_generator) / 8,
                    epochs = 5,
                    workers=20)