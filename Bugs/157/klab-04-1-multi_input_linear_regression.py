import tensorflow as tf
import numpy as np

# Data
x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]
y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]

# Define the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_dim=3, activation='linear'))

# Compile the model
rmsprop = tf.keras.optimizers.RMSprop(learning_rate=1e-10)
model.compile(loss='mse', optimizer=rmsprop)

# Train the model
model.fit(np.array(x_data), np.array(y_data), epochs=1000, verbose=0)

# Predict
y_predict = model.predict(np.array([[95., 100., 80]]))
print(y_predict)
