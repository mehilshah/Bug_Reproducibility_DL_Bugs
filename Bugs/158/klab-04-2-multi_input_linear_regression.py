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
model.add(tf.keras.layers.Dense(units=1, input_dim=3))

# Compile the model
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
model.compile(loss='mse', optimizer=optimizer)

# Train the model
model.fit(np.array(x_data), np.array(y_data), epochs=1000, verbose=0)

# Predict
y_predict = model.predict(np.array([[0, 2, 1]]))
print(y_predict)
