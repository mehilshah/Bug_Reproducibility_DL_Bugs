# Multiclass Classification with the Iris Flowers Dataset 
import numpy 
import pandas
from sklearn.metrics import confusion_matrix 
import tensorflow as tf
from keras.models import Sequential 
from keras.layers import Dense 
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder 

# load dataset 
dataframe = pandas.read_csv("./iris/iris.data", header=None) 
dataset = dataframe.values 
X = dataset[:,0:4].astype(float) 
Y = dataset[:,4] 
# encode class values as integers 
encoder = LabelEncoder() 
encoder.fit(Y) 
encoded_Y = encoder.transform(Y) 

# convert integers to dummy variables (i.e. one hot encoded) 
dummy_y = to_categorical(encoded_Y) 

X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size = 0.2)

# Uncomment the model before running the file

# Network 1
# model = Sequential()
# model.add(Dense(4, input_dim=4, activation="relu", kernel_initializer="normal"))
# model.add(Dense(3, activation="sigmoid", kernel_initializer="normal"))

# # Model 2
# model = Sequential()
# model.add(Dense(8, input_dim=4, activation="relu", kernel_initializer="normal"))
# model.add(Dense(6, activation="relu", kernel_initializer="normal"))
# model.add(Dense(3, activation="softmax", kernel_initializer="normal"))

model.compile(loss= 'categorical_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])
model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1, validation_data=(X_test, y_test)) 

predictions = model.predict(X_test)
y_pred = (predictions > 0.5)

confusion_matrix(y_test, y_pred)