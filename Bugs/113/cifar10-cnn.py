'''Train a simple deep CNN on the CIFAR10 small images dataset.

It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import optimizers
import os
import random
import json

import sys, getopt, time


def main(argumentList):
    trues = ['true', '1', 't', 'y', 'yes', 'yeah', 'yup', 'certainly', 'uh-huh']

    unixOptions = "a:b:m:k:e:t:c:f:n:"
    gnuOptions = ["augmentation=", "batch_size=", "model_name=", "model_key=", "epochs=", "test_mode=",
                  "conv_layers=", "full_layers=", "neurons_map="]

    try:
        arguments, values = getopt.getopt(argumentList, unixOptions, gnuOptions)
    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))
        sys.exit(2)

    argumentsDict = dict(arguments)

    test_mode = (getValue(argumentsDict, '-t', '--test_mode', "False")).lower() in trues

    batch_size = int(getValue(argumentsDict, '-b', '--batch_size', 32))

    epochs = int(argumentsDict.get('-e', argumentsDict.get('--epochs', 100)))

    data_augmentation = bool(argumentsDict.get('-a', argumentsDict.get('--augmentation', True)))

    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = argumentsDict.get('-m', argumentsDict.get('--model_name', 'keras_cifar10_trained_model.h5'))

    # ensure model name unicity
    model_key = argumentsDict.get('-k', argumentsDict.get('--model_key', str(time.time())))
    model_name = model_name + '_' + model_key

    num_classes = 10
    print('Model Name:', model_name)

    # Parameters tuning

    # Number of convolutional layers from 3 to 50
    conv_layers = int(getValue(argumentsDict, '-c', '--conv_layers', 3))
    print('Conv Layers:', conv_layers)

    # Number of fully connected layers from 1 to 4
    full_layers = int(getValue(argumentsDict, '-f', '--full_layers', 1))
    print('Full Layers:', full_layers)

    # Map of neurons for each layers:
    # Number of maps in a convolutional layer from 8 to 512
    # Number of fully connected layers from 1 to 4
    neurons_map = getValue(argumentsDict, '-n', '--neurons_map', "32,32,32&512")
    print('Neurons Map:', neurons_map)

    [conv_map, full_map] = neurons_map.split("&")
    conv_map = conv_map.split(',')
    full_map = full_map.split(',')

    # The data, shuffled and split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    if test_mode:

        dumb_accuracy = random.uniform(0, 100)

        # Score trained model.
        print('Test loss:', random.uniform(0, 10))
        print('Test accuracy:', dumb_accuracy)

        sys.stdout.write(str(dumb_accuracy))
        sys.stdout.flush()
        sys.exit(0)

    else:

        # Convert class vectors to binary class matrices.
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        model = Sequential()

        model.add(Conv2D(int(conv_map[0]), (3, 3), padding='same',
                         input_shape=x_train.shape[1:]))
        model.add(Activation('relu'))

        for i in range(1, conv_layers):
            model.add(Conv2D(int(conv_map[i]), (3, 3)))
            model.add(Activation('relu'))

        model.add(Flatten())

        for i in range(0, full_layers):
            model.add(Dense(int(full_map[i])))
            model.add(Activation('relu'))
            model.add(Dropout(0.5))

        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

        # initiate RMSprop optimizer
        opt = optimizers.RMSprop(lr=0.0001, decay=1e-6)

        # Let's train the model using RMSprop
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        if not data_augmentation:
            print('Not using data augmentation.')
            model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      validation_data=(x_test, y_test),
                      shuffle=True)
        else:
            print('Using real-time data augmentation.')
            # This will do preprocessing and realtime data augmentation:
            datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False)  # randomly flip images

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            epochs=epochs,
                            validation_data=(x_test, y_test),
                            workers=4)

        # Save model and weights
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        model_path = os.path.join(save_dir, model_name)
        model.save(model_path)
        print('Saved trained model at %s ' % model_path)

        # Score trained model.
        scores = model.evaluate(x_test, y_test, verbose=1)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])

        sys.stdout.write(str(scores[1]))
        sys.stdout.flush()
        sys.exit(0)

        # ///////// Added By @Mehdi
        file = open(file="buggy/result.json", mode="w")  
        model_acc = np.float64(scores[1])
        res = {"accuracy" : model_acc}
        json.dump(res, file)
        file.close()


def getValue(dictionary, shortKey, longKey, default):
    return dictionary.get(shortKey, dictionary.get(longKey, default))


if __name__ == "__main__":
    main(sys.argv[1:])
