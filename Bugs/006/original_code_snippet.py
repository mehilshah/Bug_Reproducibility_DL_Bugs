import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
from sklearn.model_selection import train_test_split

train_data_path = "data/train.csv" #path where data is stored
train_data = pd.read_csv(train_data_path, header=None) #load data in dataframe using pandas

X_train, X_val = train_test_split(train_data, test_size=0.2, random_state=42)

X_train, y_train = X_train.iloc[:, :-1], X_train.iloc[:, -1]
X_val, y_val = X_val.iloc[:, :-1], X_val.iloc[:, -1]
print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

def make_model():
    input_vec = tf.keras.layers.Input((19,))
    final = tf.keras.layers.Dense(12, activation='relu')(input_vec)
    final = tf.keras.layers.Dense(1, activation='sigmoid')(final)

    model = tf.keras.models.Model(inputs=[input_vec], outputs=[final])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

model = make_model()
model.summary()
tf.keras.utils.plot_model(model, 'model.png')

model.fit(X_train, y_train, validation_data=[X_val, y_val], epochs=10, batch_size=64, validation_batch_size=64)

print(model.evaluate(X_train, y_train))
print(model.evaluate(X_val, y_val))