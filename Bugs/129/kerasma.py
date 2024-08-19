import numpy as np
import tensorflow as tf
import json

# Fix random seed for reproducibility
np.random.seed(7)

# Load Pima Indians dataset
dataset = np.loadtxt("pima-indians-diabetes.data", delimiter=",")
# Split into input (X) and output (Y) variables
X = dataset[:, 0:8]
Y = dataset[:, 8]

# Create model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(12, input_dim=8, activation='relu'))
model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, epochs=150, batch_size=10)

# Evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

# Save results to JSON file
with open("result.json", mode="w") as file:
    model_accuracy = np.float64(scores[1])
    res = {"accuracy": model_accuracy}
    json.dump(res, file)
