#datapoints
X = np.arange(0.0, 5.0, 0.1, dtype='float32').reshape(-1,1)
y = 5 * np.power(X,2) + np.power(np.random.randn(50).reshape(-1,1),3)

#model
model = Sequential()
model.add(Dense(50, activation='relu', input_dim=1))
model.add(Dense(30, activation='relu', init='uniform'))
model.add(Dense(output_dim=1, activation='linear'))

#training
sgd = SGD(lr=0.1);
model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])
model.fit(X, y, nb_epoch=1000)

#predictions
predictions = model.predict(X)

#plot
plt.scatter(X, y,edgecolors='g')
plt.plot(X, predictions,'r')
plt.legend([ 'Predictated Y' ,'Actual Y'])
plt.show()