import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

dirname = os.path.dirname(os.path.realpath(__file__))
# Place the .csv files one level up and within data/kaggle-facial-keypoint-detection folder.
FTRAIN = os.path.join(dirname, '../data/training.csv')
FTEST = os.path.join(dirname, '../data/test.csv')

def load(test=False, cols=None):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """
    fname = FTEST if test else FTRAIN
    df = pd.read_csv(os.path.expanduser(fname))  # load pandas dataframe

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # get a subset of columns
        df = df[list(cols) + ['Image']]

    print(df.count())  # prints the number of values for each column
    df = df.dropna()  # drop all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.float32)
    else:
        y = None

    return X, y

X, y = load()
print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
    X.shape, X.min(), X.max()))
print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
    y.shape, y.min(), y.max()))

# Create the TensorFlow Keras model
model = Sequential()
model.add(Dense(50, input_shape=(9216,), activation='relu')) # Input Batches of 96x96 input pixels per batch
model.add(Dense(100, activation='relu'))
model.add(Dense(30)) # 30 to represent 30 facial keypoints.

# Compile the model. Minimize mean squared error
model.compile(optimizer=Adam(learning_rate=0.1), loss='mean_squared_error')

# Fit the model with the data
model.fit(X, y, epochs=100, verbose=1)

# Get loss on the data
eval_result = model.evaluate(X, y)
print("\n\nTest loss:", eval_result)

# Save the model
model.save('1_single_hidden_layer.h5')

# Save the result
import json
with open("buggy/result.json", "w") as file:
    model_loss = np.float64(eval_result)
    res = {"loss": model_loss}
    json.dump(res, file)
