from keras.models import Sequential
from keras.layers import Dense

# Use Iris Dataset
from sklearn import datasets
X, y = datasets.load_iris(return_X_y=True)

model = Sequential()
model.add(Dense(4, input_dim=4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, y, epochs=10, batch_size=4, verbose=0)

# Evaluate the model
scores = model.evaluate(X, y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

model = Sequential()
model.add(Dense(8, input_dim=4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Fit the model
model.fit(X, y, epochs=10, batch_size=4, verbose=0)

# Evaluate the model
scores = model.evaluate(X, y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
