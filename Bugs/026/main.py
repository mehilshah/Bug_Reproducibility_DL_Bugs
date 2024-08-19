import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

x = pd.Series([i/10 for i in range(-100, 101)])
y = x**2 + 3
dataset = pd.DataFrame({'x': x, 'y': y})

X_train, X_test, Y_train, Y_test = train_test_split(dataset['x'], dataset['y'], 
                                                    test_size=0.25,)
# Now we build the model
neural_network = Sequential() # create model
neural_network.add(Dense(5, input_dim=1, activation='sigmoid')) # hidden layer
neural_network.add(Dense(1, activation='sigmoid')) # output layer
neural_network.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
neural_network_fitted = neural_network.fit(X_train, Y_train, epochs=1000, verbose="0", 
                                           batch_size=X_train.shape[0], initial_epoch=0)

# Make predictions
Y_predicted = neural_network.predict(X_test)
print (Y_predicted)