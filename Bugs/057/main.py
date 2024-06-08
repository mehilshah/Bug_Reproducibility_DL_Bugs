from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Input, LSTM
from tensorflow.keras.losses import MeanSquaredError
import numpy as np

# Once again, the shape of feature_vec is (68,1) and the shape of output_prob_matrix is (?,10)
lstm_input_sequences = np.random.rand(68,1,10)
output_prob_matrix = np.random.rand(68,10,10)

lstm = Sequential()
lstm.add(LSTM(10, input_shape=(1,10)))
lstm.add(Dense(1))
feature_vec = lstm(lstm_input_sequences)
feature_vec = np.array(feature_vec)

vnn = Sequential()
vnn.add(Input(1,68))
vnn.add(Dense(units=10,activation='sigmoid'))
loss_fn = MeanSquaredError()
vnn.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
print(vnn.summary())
vnn.fit(feature_vec,output_prob_matrix,32,100)