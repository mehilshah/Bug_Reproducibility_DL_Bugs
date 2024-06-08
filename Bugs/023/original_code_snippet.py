import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline
import keras

dataset_size = 200

class_1 = np.random.uniform(low=0.2, high=0.4, size=(dataset_size,))
class_2 = np.random.uniform(low=0.7, high=0.5, size=(dataset_size,))

dataset = []
for i in range(0, dataset_size, 2):
    dataset.append([class_1[i],class_1[i+1],1])
    dataset.append([class_2[i],class_2[i+1],2])
    
df_train = pd.DataFrame(data=dataset, columns=['x','y','class'])
    
df_train.head()

from sklearn.preprocessing import LabelEncoder

x_train = df_train.iloc[:,0:-1].values
y_train = df_train.iloc[:, -1]

nr_feats = x_train.shape[1]
nr_classes = y_train.nunique()

label_enc = LabelEncoder()
label_enc.fit(y_train)

y_train = keras.utils.to_categorical(label_enc.transform(y_train), nr_classes)

from keras.models import Sequential
from keras.layers import Dense


model = Sequential()
model.add(Dense(units=2, activation='sigmoid', input_shape= (nr_feats,)))
model.add(Dense(units=nr_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=True)
accuracy_score(model.predict_classes(x_train),  df_train.iloc[:, -1].values)

from sklearn.neural_network import MLPClassifier

sk_model = MLPClassifier(hidden_layer_sizes=(3,),activation='logistic')

sk_model.fit(x_train, df_train.iloc[:, -1].values)

from sklearn.metrics import accuracy_score

accuracy_score(sk_model.predict(x_train), df_train.iloc[:, -1].values)