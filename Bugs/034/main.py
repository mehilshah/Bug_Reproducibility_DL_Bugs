from keras.models import Sequential
from keras.layers import Dense
import numpy as np
#fix random seed for reproducibility
np.random.seed(7)

#load and read dataset
X = np.random.rand(100, 2)

# Generate random Y array with 100 samples and 1 label per sample
Y = np.random.rand(100, 1)

print ("Variables: \n", X)
print ("Target_outputs: \n", Y)
# create model
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
#model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='relu'))
model.summary()
# Compile model
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['MSE'])
# Fit the model
model.fit(X, Y, epochs=500, batch_size=10)
#make predictions (test)
F = model.predict(X)
print ("Predicted values: \n", F)