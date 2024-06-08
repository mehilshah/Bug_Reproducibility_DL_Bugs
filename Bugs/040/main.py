from keras.models import Sequential,load_model,Model
from keras.layers import Conv2D,MaxPool2D,MaxPooling2D,Dense,Dropout,BatchNormalization,Flatten,Input
from keras.layers import *
import numpy as np
#model creation 
# input_shape = (128, 128, 3)
# inputs = Input((input_shape))

input = Input(shape=(128,128,3))
conv1 = Conv2D(32,(3,3),activation="relu")(input)
pool1 = MaxPool2D((2,2))(conv1)
conv2 = Conv2D(64,(3,3),activation="relu")(pool1)
pool2 = MaxPool2D((2,2))(conv2)
conv3 = Conv2D(128,(3,3),activation="relu")(pool2)
pool3 = MaxPool2D((2,2))(conv3)
flt = Flatten()(pool3)
#age
age_l = Dense(128,activation="relu")(flt)
age_l = Dense(64,activation="relu")(age_l)
age_l = Dense(32,activation="relu")(age_l)
age_l = Dense(1,activation="relu")(age_l)

#gender
gender_l = Dense(128,activation="relu")(flt)
gender_l = Dense(80,activation="relu")(gender_l)
gender_l = Dense(64,activation="relu")(gender_l)
gender_l = Dense(32,activation="relu")(gender_l)
gender_l = Dropout(0.5)(gender_l)
gender_l = Dense(2,activation="softmax")(gender_l)

modelA = Model(inputs=input,outputs=[age_l,gender_l])


modelA.compile(loss=['mse', 'sparse_categorical_crossentropy'], optimizer='adam', metrics=['accuracy', 'mae'])

modelA.summary() 

X_train = np.random.rand(4000, 128, 128, 3)
X_test = np.random.rand(1000, 128, 128, 3)

# Generate y_train, y_train2 with same label, i.e 1
y_train = np.ones((4000, 1))
y_train2 = np.ones((4000, 1))
y_test = np.ones((1000, 1))
y_test2 = np.ones((1000, 1))

save = modelA.fit(X_train, [y_train, y_train2],
      validation_data = (X_test, [y_test, y_test2]),
        batch_size = 128,
      epochs = 30)