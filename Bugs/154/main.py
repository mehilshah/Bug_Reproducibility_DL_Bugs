import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.callbacks import LearningRateScheduler
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from datetime import datetime

import helper_funcs
import settings

TESTING = settings.TESTING
steps_per_epoch = settings.steps_per_epoch
validation_steps = settings.validation_steps
epochs = settings.epochs
decay = settings.decay

train_path = 'dataset/train'
valid_path = 'dataset/valid'

train_batches = ImageDataGenerator().flow_from_directory(train_path, color_mode='grayscale', batch_size=30, target_size=(224,224))
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, color_mode='grayscale', batch_size=30, target_size=(224,224))

callbacks = [LearningRateScheduler(helper_funcs.PolynomialDecay(maxEpochs = epochs, initAlpha=1e-1, power=5))]
opt = Adam(learning_rate=1e-1)

model = helper_funcs.create_new_model()
# model = helper_funcs.load_model()

model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])

if not TESTING:
    print("----------------- TRAINING -----------------")
    try:
        history = model.fit_generator(train_batches,
                            steps_per_epoch = steps_per_epoch,
                            callbacks = callbacks,
                            validation_data = valid_batches,
                            validation_steps = validation_steps,
                            epochs = epochs,
                            verbose = 1,
                            shuffle = True)

        curr_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Plot training & validation data
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('training data')
        plt.ylabel('Accuracy/Loss')
        plt.xlabel('Epoch')
        plt.legend(['accuracy', 'val_accuracy', 'loss', 'val_loss'], loc='upper left')

        fname = "model_graphs/" + curr_datetime + "_class_model.jpg"
        plt.savefig(fname)

    except KeyboardInterrupt:
        helper_funcs.save_model(model, "class_model")

        curr_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Plot training & validation data
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('training data')
        plt.ylabel('Accuracy/Loss')
        plt.xlabel('Epoch')
        plt.legend(['accuracy', 'val_accuracy', 'loss', 'val_loss'], loc='upper left')

        fname = "model_graphs/" + curr_datetime + "_class_model.jpg"
        plt.savefig(fname)
    else:
        helper_funcs.save_model(model, "class_model")
else:
    print("----------------- TESTING -----------------")
    history = model.fit_generator(train_batches,
                        steps_per_epoch = steps_per_epoch,
                        callbacks = callbacks,
                        validation_data = valid_batches,
                        validation_steps = validation_steps,
                        epochs = epochs,
                        verbose = 1,
                        shuffle = True)

