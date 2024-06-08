from keras.models import Sequential
from keras.layers import Conv2D, Activation, Dropout
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.preprocessing import image
import pandas as pd
from PIL import Image
import IPython

import cv2
import numpy as np
from matplotlib import pyplot as plt

import datetime
import os
import platform
import keras
import random
import shutil

if platform.system()=='Windows':
    dir = './data/'
else:
    dir= './data/'

csv_train_data=dir+'train.csv'
model_dir=dir

image_size=28
epochs=20
num_classes=10
batch_size=32

train_data=pd.read_csv(csv_train_data)

num_images=train_data.shape[0]

images=[]
labels=[]
for x in range(0, num_images):
    img_arr=train_data.iloc[x,1:].values.reshape(28,28).astype(dtype='uint8')
    img_label=train_data.iloc[x].values[0]
    images.append(img_arr)
    labels.append(img_label)

train_x=np.array(images)
train_y=np.array(labels)

train_x=train_x.reshape(num_images,image_size,image_size,1)
train_y=pd.get_dummies(train_y)
print(train_x.shape, train_y.shape)

def build_model(image_size,num_classes):
    model = Sequential()
    model.add(Conv2D(128, (3, 3), activation='relu' ,input_shape=( image_size, image_size, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
   # model.add(Dropout(0.5))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
   # model.add(Dropout(0.5))
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
   # model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
   # model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))


    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy','categorical_accuracy'])

    model.summary()

    return model;


model = build_model(image_size,num_classes);
checkpoint = ModelCheckpoint(model_dir+'/checkpoint_model_v2.h5',
                             monitor='val_acc', verbose=1, save_best_only=True, mode='max')
model.fit(x=train_x, y=train_y, batch_size=batch_size, verbose=2,
          epochs=epochs, validation_split=0.2, callbacks=[checkpoint] )