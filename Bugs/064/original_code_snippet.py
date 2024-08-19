dataset = df.values
X = dataset[:, 1:8]
Y = dataset[:,0]

## Normalize X-Values
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)
X_scale

##Partition Data
from sklearn.model_selection import train_test_split
X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)
print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)

from keras.models import Sequential
from keras.layers import Dense

model = Sequential(
    Dense(32, activation='relu', input_shape=(7,)),
    Dense(1, activation='linear'))

model.compile(optimizer='sgd',
              loss='mse',
              metrics=['mean_squared_error'])

model.evaluate(X_test, Y_test)[1] ##Type Error is here!