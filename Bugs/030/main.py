from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# Create a dataframe with columns 0, 1, 2 with random values between 65, and 70
# Create a column 3 with values 0 or 1
dataset = numpy.random.randint(65, 70, size=(100, 3))
dataset = numpy.append(dataset, numpy.random.randint(0, 2, size=(100, 1)), axis=1)

# split into input (X) and output (Y) variables
X = dataset[:,0:3]
Y = dataset[:,3]
# split into 67% for train and 33% for test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=seed)
# create model
model = Sequential()
model.add(Dense(12, input_dim=3, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=150, batch_size=10)