import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

def generator(nb_samples, matrix_size = 2, entries_range = (0,1), determinant = None):
    ''' 
    Generate nb_samples random matrices of size matrix_size with float 
    entries in interval entries_range and of determinant determinant 
    ''' 
    matrices = []
    if determinant:
        inverses = []
        for i in range(nb_samples):
            matrix = np.random.uniform(entries_range[0], entries_range[1], (matrix_size,matrix_size))
            matrix[0] *= determinant/np.linalg.det(matrix)
            matrices.append(matrix.reshape(matrix_size**2,))
            inverses.append(np.array(np.linalg.inv(matrix)).reshape(matrix_size**2,))
        return np.array(matrices), np.array(inverses)
    else:
        determinants = []
        for i in range(nb_samples):
            matrix = np.random.uniform(entries_range[0], entries_range[1], (matrix_size,matrix_size))
            determinants.append(np.array(np.linalg.det(matrix)).reshape(1,))
            matrices.append(matrix.reshape(matrix_size**2,))    
        return np.array(matrices), np.array(determinants)

### Select number of samples, matrix size and range of entries in matrices
nb_samples = 10000
matrix_size = 3
entries_range = (0, 100)
determinant = 1

### Generate random matrices and determinants
matrices, inverses = generator(nb_samples, matrix_size = matrix_size, entries_range = entries_range, determinant = determinant)

### Select number of layers and neurons
nb_hidden_layers = 1
nb_neurons = matrix_size**2
activation = 'relu'

### Create dense neural network with nb_hidden_layers hidden layers having nb_neurons neurons each
model = Sequential()
model.add(Dense(nb_neurons, input_dim = matrix_size**2, activation = activation))
for i in range(nb_hidden_layers):
    model.add(Dense(nb_neurons, activation = activation))
model.add(Dense(matrix_size**2))
model.compile(loss='mse', optimizer='adam')

### Train and save model using train size of 0.66
history = model.fit(matrices, inverses, epochs = 400, batch_size = 100, verbose = 0, validation_split = 0.33)

### Get validation loss from object 'history'
rmse = np.sqrt(history.history['val_loss'][-1])

### Print RMSE and parameter values
print('''
Validation RMSE: {}
Number of hidden layers: {}
Number of neurons: {}
Number of samples: {}
Matrices size: {}
Range of entries: {}
Determinant: {}
'''.format(rmse,nb_hidden_layers,nb_neurons,nb_samples,matrix_size,entries_range,determinant))