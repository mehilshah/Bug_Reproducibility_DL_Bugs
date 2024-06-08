dataset = pd.read_csv("simple_network/Linear Data.csv", header=None).values
X_train, X_test, Y_train, Y_test = train_test_split(dataset[:,0:1], dataset[:,1], 
                                                    test_size=0.25,)
# Now we build the model
neural_network = Sequential() # create model
neural_network.add(Dense(5, input_dim=1, activation='sigmoid')) # hidden layer
neural_network.add(Dense(1, activation='sigmoid')) # output layer
neural_network.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
neural_network_fitted = neural_network.fit(X_train, Y_train, epochs=1000, verbose=0, 
                                           batch_size=X_train.shape[0], initial_epoch=0)