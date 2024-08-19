import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import keras
from keras.models import Sequential
from keras.layers import Dense

# Load the dataset
dataset = np.array([[1.0, 2.0, 0], [2.0, 1.0, 1], [2.0, 3.0, 0], [3.0, 2.0, 1], [4.0, 2.0, 0], [3.0, 3.0, 1]])
df_train = pd.DataFrame(data=dataset, columns=['x', 'y', 'class'])

# Prepare the input and output data
x_train = df_train.iloc[:, 0:-1].values
y_train = df_train.iloc[:, -1]

nr_feats = x_train.shape[1]
nr_classes = y_train.nunique()

label_enc = LabelEncoder()
label_enc.fit(y_train)

y_train = keras.utils.to_categorical(label_enc.transform(y_train), nr_classes)

# Define the model
model = Sequential()
model.add(Dense(units=2, activation='sigmoid', input_shape=(nr_feats,)))
model.add(Dense(units=nr_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=True)

# Compute the accuracy score
y_pred_prob = model.predict(x_train)
y_pred = np.argmax(y_pred_prob, axis=-1)
y_true = label_enc.transform(df_train.iloc[:, -1])
acc = accuracy_score(y_true, y_pred)
print(f"Accuracy score: {acc}")